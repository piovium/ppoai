
from pathlib import Path
import re
import sys

ROOT = Path.cwd()
ENV_PATH = ROOT / "research/world_model/src/gitcg_world_model/env.py"
TRAIN_PATH = ROOT / "research/world_model/src/gitcg_world_model/ppo_training.py"

def backup(path: Path, suffix: str):
    bak = path.with_name(path.name + suffix)
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return bak

def patch_env(text: str) -> str:
    old_block = """    def step(self, choice: ActionChoice) -> tuple[DecisionContext | None, TrajectoryStep, bool]:
        if self._current_context is None or self._current_context.terminal:
            raise RuntimeError("environment is not awaiting an action")
        chosen_action_code = int(choice.action_code)
        if chosen_action_code < 0:
            chosen_action_code = _fallback_action_code(self._current_context)
        self._response_queue.put(chosen_action_code)
        event = self._next_event()
"""
    new_block = """    def step(self, choice: ActionChoice) -> tuple[DecisionContext | None, TrajectoryStep, bool]:
        if self._current_context is None or self._current_context.terminal:
            raise RuntimeError("environment is not awaiting an action")
        current_context = self._current_context
        current_built_context = self._current_built_context
        chosen_action_code = int(choice.action_code)
        if chosen_action_code < 0:
            chosen_action_code = _fallback_action_code(current_context)
        self._response_queue.put(chosen_action_code)
        event = self._next_event()
"""
    if old_block not in text:
        raise RuntimeError("env.py: missing step prologue block")
    text = text.replace(old_block, new_block, 1)

    replacements = {
        "self._current_context.acting_player": "current_context.acting_player",
        "self._current_context.request_type": "current_context.request_type",
        "pre_state=self._current_context.full_state": "pre_state=current_context.full_state",
        "legal_low_level_codes=self._current_context.legal_low_level_codes": "legal_low_level_codes=current_context.legal_low_level_codes",
        "legal_low_level_specs=self._current_context.legal_low_level_specs": "legal_low_level_specs=current_context.legal_low_level_specs",
        "legal_low_level_mask=self._current_context.legal_low_level_mask": "legal_low_level_mask=current_context.legal_low_level_mask",
        "legal_high_level_codes=self._current_context.legal_high_level_codes": "legal_high_level_codes=current_context.legal_high_level_codes",
        "high_to_low_map=self._current_context.high_to_low_map": "high_to_low_map=current_context.high_to_low_map",
        "player_view=self._current_context.player_view": "player_view=current_context.player_view",
        "full_state_json_before=self._current_context.full_state_json": "full_state_json_before=current_context.full_state_json",
        "next_context.full_state.winner,\n                self._current_context.acting_player,": "next_context.full_state.winner,\n                current_context.acting_player,",
        "for candidate_code, candidate_spec in zip(\n            self._current_context.legal_low_level_codes,\n            self._current_context.legal_low_level_specs,": "for candidate_code, candidate_spec in zip(\n            current_context.legal_low_level_codes,\n            current_context.legal_low_level_specs,",
        "self._current_built_context.low_to_high_code.get(chosen_action_code)\n                if self._current_built_context is not None": "current_built_context.low_to_high_code.get(chosen_action_code)\n                if current_built_context is not None",
        "\"action_label\": _action_label(self._current_built_context, chosen_action_code),": "\"action_label\": _action_label(current_built_context, chosen_action_code),",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"env.py: missing snippet: {old[:60]!r}")
        text = text.replace(old, new, 1)
    return text

def patch_training(text: str) -> str:
    old_sample = """                    chosen_high_level_code=int(
                        transition.source_step.chosen_high_level_code
                        if transition.source_step.chosen_high_level_code is not None
                        else encoded.low_to_high_codes[transition.action_index]
                    ),
"""
    new_sample = """                    chosen_high_level_code=int(
                        transition.source_step.chosen_high_level_code
                        if (
                            transition.source_step.chosen_high_level_code is not None
                            and int(transition.source_step.chosen_high_level_code)
                            in {int(code) for code in transition.source_step.legal_high_level_codes}
                        )
                        else encoded.low_to_high_codes[transition.action_index]
                    ),
"""
    if old_sample not in text:
        raise RuntimeError("ppo_training.py: missing chosen_high_level_code sample block")
    text = text.replace(old_sample, new_sample, 1)

    old_valid = """        step_high_codes = chosen_high_level_code.index_select(0, loss_indices)[:, step_index]
        valid_high = step_high_codes >= 0
"""
    new_valid = """        step_high_codes = chosen_high_level_code.index_select(0, loss_indices)[:, step_index]
        high_dim = step_high_log_probs.shape[-1]
        valid_high = (
            (step_high_codes >= 0)
            & (step_high_codes < high_dim)
            & torch.gather(
                current_high_mask.to(dtype=torch.bool),
                1,
                step_high_codes.clamp_min(0).unsqueeze(-1),
            ).squeeze(-1)
        )
"""
    if old_valid not in text:
        raise RuntimeError("ppo_training.py: missing valid_high block")
    text = text.replace(old_valid, new_valid, 1)
    return text

def main():
    for path in (ENV_PATH, TRAIN_PATH):
        if not path.exists():
            raise SystemExit(f"missing file: {path}")
        backup(path, ".bak_high_policy_fix")

    env_text = ENV_PATH.read_text(encoding="utf-8")
    train_text = TRAIN_PATH.read_text(encoding="utf-8")

    env_text = patch_env(env_text)
    train_text = patch_training(train_text)

    ENV_PATH.write_text(env_text, encoding="utf-8")
    TRAIN_PATH.write_text(train_text, encoding="utf-8")
    print(f"patched: {ENV_PATH}")
    print(f"patched: {TRAIN_PATH}")
    print("backup suffix: .bak_high_policy_fix")

if __name__ == "__main__":
    main()
