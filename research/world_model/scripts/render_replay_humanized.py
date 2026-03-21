
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def _load_recap_module(recap_path: Path):
    if not recap_path.exists():
        raise FileNotFoundError(f"render_episode_recap.py not found: {recap_path}")

    # Make the repo importable for render_episode_recap.py
    try:
        repo_root = recap_path.resolve().parents[3]
    except IndexError:
        repo_root = recap_path.resolve().parent

    candidates = [
        repo_root,
        repo_root / "research" / "world_model" / "src",
        repo_root / "research",
    ]
    for candidate in candidates:
        text = str(candidate)
        if candidate.exists() and text not in sys.path:
            sys.path.insert(0, text)

    spec = importlib.util.spec_from_file_location("render_episode_recap_runtime", recap_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import recap module from: {recap_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_jsonl(path: str | Path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse {path} line {lineno}: {exc}") from exc


def load_episodes(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def sanitize_filename(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "episode"


def choose_replay_path(args) -> Path:
    if args.replay:
        return Path(args.replay)
    if not args.round_dir:
        raise SystemExit("Please provide --replay or --round-dir")
    round_dir = Path(args.round_dir)
    name = {
        "self_play": "self_play.jsonl",
        "br_self_play": "br_self_play.jsonl",
        "bootstrap": "bootstrap.jsonl",
    }[args.source]
    return round_dir / name


def filter_episodes(episodes: list[dict[str, Any]], matchup: str | None) -> list[dict[str, Any]]:
    if matchup:
        episodes = [ep for ep in episodes if str(ep.get("matchup")) == matchup]
    return episodes


def _winner_text(winner: Any) -> str:
    if winner is None:
        return "平局/未结束"
    try:
        return f"玩家{int(winner)}"
    except Exception:
        return str(winner)


def _replace_player_tokens(text: str) -> str:
    text = text.replace("P0", "玩家0").replace("P1", "玩家1")
    text = text.replace("winner=0", "胜者=玩家0").replace("winner=1", "胜者=玩家1")
    return text


def _dice_pool_text(values: list[int] | tuple[int, ...], recap) -> str:
    if not values:
        return "无"
    return "、".join(recap.DICE_TEXT.get(int(v), str(v)) for v in values)


def _entity_name(definition_id: int | None, recap) -> str:
    try:
        return recap._name_for_definition(definition_id)
    except Exception:
        if definition_id is None:
            return "未知"
        return f"def={definition_id}"


def _character_summary(ch: dict[str, Any], recap) -> str:
    name = _entity_name(ch.get("definition_id"), recap)
    aura = int(ch.get("aura", 0) or 0)
    aura_text = ""
    if aura:
        aura_text = f"，{recap.AURA_TEXT.get(aura, str(aura))}"
    status = []
    if ch.get("is_active"):
        status.append("前台")
    if ch.get("defeated"):
        status.append("已倒下")
    suffix = f"（{'、'.join(status)}）" if status else ""
    return (
        f"「{name}」{suffix}："
        f"生命 {ch.get('health')}/{ch.get('max_health')}，"
        f"能量 {ch.get('energy')}/{ch.get('max_energy')}{aura_text}"
    )


def _zone_names(items: list[dict[str, Any]], recap) -> str:
    if not items:
        return "无"
    names = []
    for item in items:
        names.append(_entity_name(item.get("definition_id"), recap))
    return "、".join(names)


def _player_board_text(player: dict[str, Any], player_idx: int, recap) -> str:
    active_id = player.get("active_character_id")
    active = None
    for ch in player.get("characters", []):
        if ch.get("is_active") or (active_id is not None and ch.get("id") == active_id):
            active = ch
            break
    if active is None and player.get("characters"):
        active = player["characters"][0]

    active_name = _entity_name(active.get("definition_id"), recap) if active else "未知"
    bench = []
    for ch in player.get("characters", []):
        if active is not None and ch.get("id") == active.get("id"):
            continue
        bench.append(_character_summary(ch, recap))

    return (
        f"- 玩家{player_idx}：前台是「{active_name}」；"
        f"手牌 {len(player.get('hand_cards', []))} 张；"
        f"骰子={_dice_pool_text(player.get('dice', []), recap)}；"
        f"召唤物={_zone_names(player.get('summons', []), recap)}；"
        f"支援={_zone_names(player.get('supports', []), recap)}；"
        f"战斗状态={_zone_names(player.get('combat_statuses', []), recap)}；"
        f"后台：" + ("；".join(bench) if bench else "无")
    )


def _board_section(title: str, state: dict[str, Any], recap) -> list[str]:
    phase = state.get("phase")
    round_number = state.get("round_number")
    winner = state.get("winner")
    lines = [title, ""]
    lines.append(f"- 阶段：{phase}；回合：{round_number}；胜者：{_winner_text(winner)}")
    for idx, player in enumerate(state.get("players", [])):
        lines.append(_player_board_text(player, idx, recap))
    lines.append("")
    return lines


def _prepare_episode_payload(episode: dict[str, Any]) -> dict[str, Any]:
    payload = dict(episode)
    steps = [dict(step) for step in payload.get("steps", [])]
    payload["steps"] = steps
    for index, step in enumerate(steps):
        if step.get("full_state_json_after"):
            continue
        if index + 1 < len(steps):
            step["full_state_json_after"] = steps[index + 1].get("full_state_json_before")
        else:
            step["full_state_json_after"] = payload.get("final_state_json")
    return payload


def _human_line(step: dict[str, Any], idx: int, recap) -> str:
    line = recap._human_line(step, idx)
    line = _replace_player_tokens(line)
    line = line.replace("，终局 胜者=玩家", "，对局结束，胜者=玩家")
    return line


def render_episode_markdown(episode: dict[str, Any], episode_no: int, recap) -> str:
    payload = _prepare_episode_payload(episode)
    steps = list(payload.get("steps", []))
    first_state = steps[0].get("pre_state") if steps else payload.get("final_state", {})
    final_state = payload.get("final_state", {})
    grouped: dict[int, list[dict[str, Any]]] = {}
    for step in steps:
        round_number = int(step.get("pre_state", {}).get("round_number", -1))
        grouped.setdefault(round_number, []).append(step)

    lines: list[str] = []
    lines.append(f"# Episode {episode_no:04d}")
    lines.append("")
    lines.append("## 基本信息")
    lines.append(f"- 对局：`{payload.get('matchup')}`")
    lines.append(f"- 随机种子：`{payload.get('seed')}`")
    lines.append(f"- 胜者：`{_winner_text(payload.get('winner'))}`")
    lines.append(f"- 总步数：`{len(steps)}`")
    lines.append("")

    lines.extend(_board_section("## 开局盘面", first_state or {}, recap))
    lines.append("## 按回合时间线")
    lines.append("")
    for round_number in sorted(grouped):
        lines.append(f"### 第 {round_number} 回合")
        lines.append("")
        for idx, step in enumerate(grouped[round_number], 1):
            lines.append(_human_line(step, idx, recap))
        lines.append("")
    lines.extend(_board_section("## 终局盘面", final_state or {}, recap))
    return "\n".join(lines)


def write_one_episode(episode: dict[str, Any], out_path: Path, episode_no: int, recap) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_episode_markdown(episode, episode_no, recap), encoding="utf-8")


def write_all_episodes(episodes: list[dict[str, Any]], out_dir: Path, recap) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_lines = ["# Replay Index", ""]
    for i, ep in enumerate(episodes, 1):
        matchup = str(ep.get("matchup", "unknown"))
        winner = _winner_text(ep.get("winner"))
        seed = ep.get("seed")
        filename = f"episode_{i:04d}_{sanitize_filename(matchup)}.md"
        write_one_episode(ep, out_dir / filename, i, recap)
        index_lines.append(
            f"- [{filename}]({filename}) | 对局=`{matchup}` | 胜者=`{winner}` | seed=`{seed}` | steps={len(ep.get('steps', []))}"
        )
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert replay jsonl into human-readable Chinese markdown recaps."
    )
    p.add_argument("--replay", type=str, default=None, help="Path to *.jsonl replay file")
    p.add_argument("--round-dir", type=str, default=None, help="Path to round_XXXX directory")
    p.add_argument(
        "--source",
        type=str,
        default="self_play",
        choices=["self_play", "br_self_play", "bootstrap"],
        help="Which replay file to use under --round-dir",
    )
    p.add_argument("--matchup", type=str, default=None, help="Filter by exact matchup string")
    p.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="1-based index after filtering. Example: 32",
    )
    p.add_argument("--all", action="store_true", help="Render all filtered episodes")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for single episode, or output dir when --all is used",
    )
    p.add_argument(
        "--recap-script",
        type=str,
        default=None,
        help="Path to render_episode_recap.py; defaults to sibling file",
    )
    return p.parse_args()


def main():
    args = parse_args()
    replay_path = choose_replay_path(args)
    recap_path = Path(args.recap_script) if args.recap_script else Path(__file__).with_name("render_episode_recap.py")
    recap = _load_recap_module(recap_path)

    episodes = load_episodes(replay_path)
    episodes = filter_episodes(episodes, matchup=args.matchup)
    if not episodes:
        raise SystemExit(f"No episodes matched in {replay_path}")

    if args.all:
        out_dir = Path(args.output) if args.output else replay_path.with_suffix("")
        write_all_episodes(episodes, out_dir, recap)
        print(f"Wrote {len(episodes)} episode markdown files to: {out_dir}")
        return

    episode_index = args.episode_index or 1
    if episode_index < 1 or episode_index > len(episodes):
        raise SystemExit(f"--episode-index must be in [1, {len(episodes)}]")

    episode = episodes[episode_index - 1]
    out_path = Path(args.output) if args.output else replay_path.with_name(
        replay_path.stem + f".episode_{episode_index:04d}.md"
    )
    write_one_episode(episode, out_path, episode_index, recap)
    print(f"Wrote episode {episode_index} to: {out_path}")


if __name__ == "__main__":
    main()
