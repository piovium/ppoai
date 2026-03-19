from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from gitcg_world_model.action_hierarchy import low_level_spec_for_code


ACTION_KIND_TEXT = {
    "action_use_skill": "使用技能",
    "action_play_card": "打出卡牌",
    "action_switch_active": "切换出战角色",
    "action_elemental_tuning": "元素调和",
    "action_declare_end": "结束回合",
    "choose_active": "选择出战角色",
    "select_card": "选牌",
    "reroll_dice": "重投骰子",
    "switch_hands": "换手牌",
}

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "packages" / "data" / "src"

DICE_TEXT = {
    0: "无色",
    1: "冰",
    2: "水",
    3: "火",
    4: "雷",
    5: "风",
    6: "岩",
    7: "草",
    8: "万能",
    9: "充能",
    10: "秘传",
}

AURA_TEXT = {
    0: "无附着",
    1: "冰附着",
    2: "水附着",
    3: "火附着",
    4: "雷附着",
    5: "草附着",
    6: "冰草附着",
}

ID_NAME_PATTERN = re.compile(r"@id\s+(\d+)\s*[\r\n]+\s*\*\s+@name\s+(.+)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_definition_names() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for source in DATA_ROOT.rglob("*.ts"):
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = source.read_text(encoding="utf-8", errors="ignore")
        for match in ID_NAME_PATTERN.finditer(text):
            definition_id = int(match.group(1))
            name = match.group(2).strip()
            mapping.setdefault(definition_id, name)
    return mapping


DEFINITION_NAMES = _load_definition_names()


def _name_for_definition(definition_id: int | None) -> str:
    if definition_id is None:
        return "未知"
    return DEFINITION_NAMES.get(int(definition_id), f"def={definition_id}")


def _chosen_spec(step: dict[str, Any]):
    action_code = int(step.get("choice", {}).get("action_code", -1))
    if action_code < 0:
        return None
    try:
        return low_level_spec_for_code(action_code)
    except KeyError:
        return None


def _format_targets(metadata: dict[str, Any]) -> str:
    return ""


def _decode_state_json(raw_json: str) -> dict[str, Any]:
    payload = json.loads(raw_json)
    store = payload["store"]
    cache: dict[int, Any] = {}

    def resolve_index(index: int) -> Any:
        if index in cache:
            return cache[index]
        cache[index] = None
        cache[index] = resolve_node(store[index])
        return cache[index]

    def resolve_node(node: Any) -> Any:
        if isinstance(node, dict):
            if "$" in node and len(node) == 1:
                return resolve_index(int(node["$"]))
            if node.get("__type") == "map":
                return {
                    resolve_node(entry[0]): resolve_node(entry[1])
                    for entry in node.get("entries", [])
                }
            if node.get("__type") == "set":
                return [resolve_node(value) for value in node.get("values", [])]
            return {key: resolve_node(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve_node(value) for value in node]
        return node

    return resolve_index(len(store) - 1)


def _build_target_lookup(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for owner, player in enumerate(state.get("players", [])):
        active_character_id = player.get("activeCharacterId")
        for character in player.get("characters", []):
            character_id = int(character.get("id", 0))
            definition_id = int(character.get("definition", {}).get("id", 0))
            lookup[character_id] = {
                "owner": owner,
                "kind": "character",
                "definition_id": definition_id,
                "is_active": active_character_id is not None and character_id == int(active_character_id),
            }
            for entity in character.get("entities", []):
                lookup[int(entity.get("id", 0))] = {
                    "owner": owner,
                    "kind": "character_entity",
                    "definition_id": int(entity.get("definition", {}).get("id", 0)),
                    "is_active": active_character_id is not None and character_id == int(active_character_id),
                }
        for key, kind in (
            ("combatStatuses", "combat_status"),
            ("summons", "summon"),
            ("supports", "support"),
            ("hands", "hand_card"),
            ("pile", "pile_card"),
        ):
            for entity in player.get(key, []):
                lookup[int(entity.get("id", 0))] = {
                    "owner": owner,
                    "kind": kind,
                    "definition_id": int(entity.get("definition", {}).get("id", 0)),
                    "is_active": False,
                }
    return lookup


def _entity_display_name(kind: str, definition_id: int, owner: int) -> str:
    prefix = f"P{owner}"
    name = _name_for_definition(definition_id)
    if kind == "character":
        return f"{prefix}{name}"
    if kind == "support":
        return f"{prefix}支援「{name}」"
    if kind == "summon":
        return f"{prefix}召唤物「{name}」"
    if kind == "combat_status":
        return f"{prefix}战斗状态「{name}」"
    if kind == "character_entity":
        return f"{prefix}角色状态「{name}」"
    return f"{prefix}{name}"


def _format_target_names(before_root: dict[str, Any] | None, spec) -> str:
    names: list[str] = []
    if before_root is not None and spec is not None:
        player_count = len(before_root.get("players", []))
        for target in getattr(spec, "target_slots", ()):
            if player_count < 2:
                continue
            owner_index = 0 if target.owner == "self" else 1
            player = before_root.get("players", [])[owner_index]
            zone = str(target.zone)
            index = int(target.index)
            if zone == "character":
                collection = player.get("characters", [])
            elif zone == "character_entity":
                collection = [entity for character in player.get("characters", []) for entity in character.get("entities", [])]
            elif zone == "combat_status":
                collection = player.get("combatStatuses", [])
            elif zone == "summon":
                collection = player.get("summons", [])
            elif zone == "support":
                collection = player.get("supports", [])
            else:
                collection = []
            if 0 <= index < len(collection):
                entity = collection[index]
                definition_id = int(entity.get("definition", {}).get("id", 0))
                names.append(_entity_display_name(zone, definition_id, owner_index))
    if not names:
        return ""
    joined = "、".join(names)
    return f"，目标={joined}"


def _format_definition(spec) -> str:
    if spec is None:
        return ""
    definition_id = int(
        getattr(spec, "select_card_definition_id", 0)
        or getattr(spec, "subject_definition_id", 0)
        or getattr(spec, "discarded_card_definition_id", 0)
    )
    if definition_id <= 0:
        return ""
    return f"「{_name_for_definition(definition_id)}」"


def _format_dice(spec) -> str:
    used_dice = list(getattr(spec, "used_dice", ()) or ())
    if not used_dice:
        return ""
    names = "、".join(DICE_TEXT.get(int(value), str(value)) for value in used_dice)
    return f"，耗骰={names}"


def _format_dice_pool(dice_values: list[Any] | None) -> str:
    if not dice_values:
        return "无"
    return "、".join(DICE_TEXT.get(int(value), str(value)) for value in dice_values)


def _format_extra(spec) -> str:
    if spec is None:
        return ""
    kind = str(getattr(spec, "kind", ""))
    parts: list[str] = []
    if kind == "action_elemental_tuning":
        removed = int(getattr(spec, "discarded_card_definition_id", 0))
        if removed > 0:
            parts.append("弃牌=" + _name_for_definition(removed))
        used = list(getattr(spec, "used_dice", ()) or ())
        target = int(getattr(spec, "target_dice", 0))
        if used and target > 0:
            parts.append(
                f"调和：{DICE_TEXT.get(int(used[0]), str(used[0]))} -> {DICE_TEXT.get(int(target), str(target))}"
            )
    return ("，" + "，".join(parts)) if parts else ""


def _state_delta_summary(pre_state: dict[str, Any], post_state: dict[str, Any]) -> str:
    events: list[str] = []
    for player_index in range(2):
        pre_player = pre_state.get("players", [])[player_index]
        post_player = post_state.get("players", [])[player_index]
        pre_by_id = {
            int(character["id"]): {
                "definition_id": int(character["definition_id"]),
                "health": int(character["health"]),
                "max_health": int(character["max_health"]),
                "energy": int(character["energy"]),
                "max_energy": int(character["max_energy"]),
            }
            for character in pre_player.get("characters", [])
        }
        for character in post_player.get("characters", []):
            char_id = int(character["id"])
            definition_id = int(character["definition_id"])
            name = _name_for_definition(definition_id)
            post_hp = int(character["health"])
            post_max_hp = int(character["max_health"])
            post_energy = int(character["energy"])
            post_max_energy = int(character["max_energy"])
            previous = pre_by_id.get(char_id)
            if previous is None:
                continue
            hp_changed = post_hp != previous["health"]
            energy_changed = post_energy != previous["energy"]
            if not hp_changed and not energy_changed:
                continue
            parts: list[str] = [f"P{player_index}{name}"]
            if hp_changed:
                if post_hp < previous["health"]:
                    parts.append(f"HP {previous['health']}/{previous['max_health']} -> {post_hp}/{post_max_hp}")
                else:
                    parts.append(f"HP {previous['health']}/{previous['max_health']} -> {post_hp}/{post_max_hp}")
            if energy_changed:
                parts.append(
                    f"能量 {previous['energy']}/{previous['max_energy']} -> {post_energy}/{post_max_energy}"
                )
            events.append("，".join(parts))
    return ("；变化：" + "；".join(events)) if events else ""


def _aura_delta_summary(step: dict[str, Any]) -> str:
    raw_before = step.get("full_state_json_before")
    raw_after = step.get("full_state_json_after")
    if not raw_before or not raw_after:
        return ""
    before_root = _decode_state_json(str(raw_before))
    after_root = _decode_state_json(str(raw_after))
    events: list[str] = []
    for player_index in range(2):
        before_chars = {
            int(character.get("id", 0)): character
            for character in before_root.get("players", [])[player_index].get("characters", [])
        }
        for character in after_root.get("players", [])[player_index].get("characters", []):
            char_id = int(character.get("id", 0))
            previous = before_chars.get(char_id)
            if previous is None:
                continue
            before_aura = int(previous.get("variables", {}).get("aura", 0))
            after_aura = int(character.get("variables", {}).get("aura", 0))
            if before_aura == after_aura:
                continue
            definition_id = int(character.get("definition", {}).get("id", 0))
            name = _name_for_definition(definition_id)
            events.append(
                f"P{player_index}{name} 元素附着 {AURA_TEXT.get(before_aura, str(before_aura))} -> {AURA_TEXT.get(after_aura, str(after_aura))}"
            )
    return ("；附着变化：" + "；".join(events)) if events else ""


def _variable_delta_text(before: dict[str, Any], after: dict[str, Any]) -> str:
    keys = sorted(set(before) | set(after))
    changes: list[str] = []
    for key in keys:
        old = int(before.get(key, 0))
        new = int(after.get(key, 0))
        if old != new:
            changes.append(f"{key} {old}->{new}")
    return "、".join(changes)


def _zone_delta_summary(before_root: dict[str, Any] | None, after_root: dict[str, Any] | None) -> str:
    if before_root is None or after_root is None:
        return ""
    zone_specs = (
        ("supports", "支援区", "support"),
        ("summons", "召唤区", "summon"),
    )
    events: list[str] = []
    for player_index in range(2):
        before_player = before_root.get("players", [])[player_index]
        after_player = after_root.get("players", [])[player_index]
        for key, label, kind in zone_specs:
            before_map = {
                int(entity.get("id", 0)): entity
                for entity in before_player.get(key, [])
            }
            after_map = {
                int(entity.get("id", 0)): entity
                for entity in after_player.get(key, [])
            }
            added_ids = [entity_id for entity_id in after_map if entity_id not in before_map]
            removed_ids = [entity_id for entity_id in before_map if entity_id not in after_map]
            changed_ids = [
                entity_id
                for entity_id in after_map
                if entity_id in before_map
                and dict(before_map[entity_id].get("variables", {})) != dict(after_map[entity_id].get("variables", {}))
            ]
            parts: list[str] = []
            if added_ids:
                parts.append(
                    "新增="
                    + "、".join(
                        f"「{_name_for_definition(int(after_map[entity_id].get('definition', {}).get('id', 0)))}」"
                        for entity_id in added_ids
                    )
                )
            if removed_ids:
                parts.append(
                    "移除="
                    + "、".join(
                        f"「{_name_for_definition(int(before_map[entity_id].get('definition', {}).get('id', 0)))}」"
                        for entity_id in removed_ids
                    )
                )
            for entity_id in changed_ids:
                before_entity = before_map[entity_id]
                after_entity = after_map[entity_id]
                name = _name_for_definition(int(after_entity.get("definition", {}).get("id", 0)))
                delta = _variable_delta_text(
                    dict(before_entity.get("variables", {})),
                    dict(after_entity.get("variables", {})),
                )
                if delta:
                    parts.append(f"「{name}」{delta}")
            if parts:
                events.append(f"P{player_index}{label}变化：" + "；".join(parts))
    return ("；区域变化：" + "；".join(events)) if events else ""


def _hand_delta_summary(
    before_root: dict[str, Any] | None,
    after_root: dict[str, Any] | None,
    *,
    actor: int,
    kind: str,
) -> str:
    if before_root is None or after_root is None:
        return ""
    events: list[str] = []
    for player_index in range(2):
        before_player = before_root.get("players", [])[player_index]
        after_player = after_root.get("players", [])[player_index]
        before_map = {
            int(card.get("id", 0)): card
            for card in before_player.get("hands", [])
        }
        after_map = {
            int(card.get("id", 0)): card
            for card in after_player.get("hands", [])
        }
        added_ids = [card_id for card_id in after_map if card_id not in before_map]
        removed_ids = [card_id for card_id in before_map if card_id not in after_map]
        if not added_ids and not removed_ids:
            continue
        parts: list[str] = []
        if added_ids:
            parts.append(
                "抽到="
                + "、".join(
                    _name_for_definition(int(after_map[card_id].get("definition", {}).get("id", 0)))
                    for card_id in added_ids
                )
            )
        if removed_ids:
            if player_index == actor and kind == "action_play_card":
                removed_label = "打出"
            elif player_index == actor and kind == "switch_hands":
                removed_label = "换出"
            elif player_index == actor and kind == "action_elemental_tuning":
                removed_label = "调和弃掉"
            else:
                removed_label = "失去"
            parts.append(
                removed_label
                + "="
                + "、".join(
                    _name_for_definition(int(before_map[card_id].get("definition", {}).get("id", 0)))
                    for card_id in removed_ids
                )
            )
        remaining = "、".join(
            _name_for_definition(int(card.get("definition", {}).get("id", 0)))
            for card in after_player.get("hands", [])
        )
        parts.append(f"手牌剩余={len(after_player.get('hands', []))}张" + (f"（{remaining}）" if remaining else ""))
        events.append(f"P{player_index}手牌变化：" + "；".join(parts))
    return ("；手牌变化：" + "；".join(events)) if events else ""


def _actor_name(pre_state: dict[str, Any], acting_player: int) -> str:
    player = pre_state.get("players", [])[acting_player]
    active_id = player.get("active_character_id")
    for character in player.get("characters", []):
        if (active_id is not None and int(character["id"]) == int(active_id)) or bool(character.get("is_active")):
            return _name_for_definition(int(character["definition_id"]))
    return f"P{acting_player}角色"


def _human_line(step: dict[str, Any], index: int) -> str:
    spec = _chosen_spec(step)
    kind = str(getattr(spec, "kind", "unknown"))
    action_text = ACTION_KIND_TEXT.get(kind, kind)
    actor = int(step.get("acting_player", -1))
    pre_state = dict(step.get("pre_state", {}) or {})
    post_state = dict(step.get("post_state", {}) or {})
    before_root = (
        _decode_state_json(str(step["full_state_json_before"]))
        if step.get("full_state_json_before")
        else None
    )
    after_root = (
        _decode_state_json(str(step["full_state_json_after"]))
        if step.get("full_state_json_after")
        else None
    )
    round_number = int(pre_state.get("round_number", -1))
    definition = _format_definition(spec)
    targets = _format_target_names(before_root, spec)
    dice = _format_dice(spec)
    extra = _format_extra(spec)
    actor_name = _actor_name(pre_state, actor)
    result = _state_delta_summary(pre_state, post_state)
    aura_result = _aura_delta_summary(step)
    zone_result = _zone_delta_summary(before_root, after_root)
    hand_result = _hand_delta_summary(before_root, after_root, actor=actor, kind=kind)
    special = ""
    if kind == "reroll_dice":
        pre_player = pre_state.get("players", [])[actor]
        post_player = post_state.get("players", [])[actor]
        special = (
            f"，重投前={_format_dice_pool(pre_player.get('dice'))}"
            f"，重投后={_format_dice_pool(post_player.get('dice'))}"
        )
    terminal = ""
    if bool(step.get("done")):
        winner = post_state.get("winner")
        terminal = f"，终局 winner={winner}"
    return (
        f"{index}. 回合{round_number}，P{actor}当前出战「{actor_name}」{action_text}"
        f"{definition}{targets}{dice}{extra}{special}{terminal}{result}{aura_result}{zone_result}{hand_result}"
    )


def _summarize_round(round_number: int, steps: list[dict[str, Any]]) -> list[str]:
    lines = [f"## 回合 {round_number}", ""]
    for index, step in enumerate(steps, start=1):
        lines.append(_human_line(step, index))
    lines.append("")
    return lines


def render_episode(path: Path, output_path: Path) -> None:
    payload = _load_json(path)
    steps = list(payload.get("steps", []))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        round_number = int(step.get("pre_state", {}).get("round_number", -1))
        grouped.setdefault(round_number, []).append(step)

    lines = [
        f"# {path.stem}",
        "",
        f"- 对局: `{payload.get('matchup_key') or payload.get('matchup')}`",
        f"- 随机种子: `{payload.get('seed')}`",
        f"- 胜者: `player{payload.get('winner')}`" if payload.get("winner") is not None else "- 胜者: `draw`",
        f"- 总步数: `{len(steps)}`",
        f"- 最终回合: `{payload.get('final_state', {}).get('round_number')}`",
        f"- 来源模型: `{payload.get('source_checkpoint_copy') or payload.get('source_checkpoint')}`",
        "",
        "下面是按回合整理的动作时间线。",
        "",
    ]

    for round_number in sorted(grouped):
        lines.extend(_summarize_round(round_number, grouped[round_number]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a human-readable recap from an episode JSON.")
    parser.add_argument("inputs", nargs="+", help="Input episode JSON files.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered markdown files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for raw_input in args.inputs:
        input_path = Path(raw_input)
        render_episode(input_path, output_dir / f"{input_path.stem}.md")


if __name__ == "__main__":
    main()
