
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT_CANDIDATES = [
    Path("."),
    Path("research/world_model/src/gitcg_world_model"),
]

ENV_REL = Path("research/world_model/src/gitcg_world_model/env.py")
PPO_REL = Path("research/world_model/src/gitcg_world_model/ppo_training.py")


def find_repo_file(rel: Path) -> Path:
    for base in REPO_ROOT_CANDIDATES:
        candidate = base / rel
        if candidate.exists():
            return candidate
    # fallback: allow running from repo root or from src dir
    if Path(rel.name).exists():
        return Path(rel.name)
    raise FileNotFoundError(f"Could not find {rel}")


def backup(path: Path, suffix: str) -> None:
    bak = path.with_name(path.name + suffix)
    if not bak.exists():
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def patch_env(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    # 1) snapshot current context/built_context before waiting for next event
    old = """        chosen_action_code = int(choice.action_code)\n        if chosen_action_code < 0:\n            chosen_action_code = _fallback_action_code(self._current_context)\n        self._response_queue.put(chosen_action_code)\n        event = self._next_event()\n"""
    new = """        chosen_action_code = int(choice.action_code)\n        current_context_for_step = self._current_context\n        current_built_context_for_step = self._current_built_context\n        if chosen_action_code < 0:\n            chosen_action_code = _fallback_action_code(current_context_for_step)\n        self._response_queue.put(chosen_action_code)\n        event = self._next_event()\n"""
    if old not in text:
        raise RuntimeError("env.py: could not find step() pre-event block")
    text = text.replace(old, new, 1)

    # 2) use snapshotted current context, not possibly-overwritten next-step context
    text = text.replace(
        "                self._current_context.acting_player,\n",
        "                current_context_for_step.acting_player,\n",
        1,
    )

    old = """        for candidate_code, candidate_spec in zip(\n            self._current_context.legal_low_level_codes,\n            self._current_context.legal_low_level_specs,\n            strict=False,\n        ):\n"""
    new = """        for candidate_code, candidate_spec in zip(\n            current_context_for_step.legal_low_level_codes,\n            current_context_for_step.legal_low_level_specs,\n            strict=False,\n        ):\n"""
    if old not in text:
        raise RuntimeError("env.py: could not find chosen_low_level_spec loop")
    text = text.replace(old, new, 1)

    old = """        trajectory_step = TrajectoryStep(\n            acting_player=self._current_context.acting_player,\n            request_type=self._current_context.request_type,\n            choice=normalized_choice,\n            pre_state=self._current_context.full_state,\n            post_state=terminal_context.full_state,\n            reward=reward,\n            done=done,\n            legal_low_level_codes=self._current_context.legal_low_level_codes,\n            legal_low_level_specs=self._current_context.legal_low_level_specs,\n            legal_low_level_mask=self._current_context.legal_low_level_mask,\n            legal_high_level_codes=self._current_context.legal_high_level_codes,\n            high_to_low_map=self._current_context.high_to_low_map,\n            chosen_high_level_code=(\n                self._current_built_context.low_to_high_code.get(chosen_action_code)\n                if self._current_built_context is not None\n                else None\n            ),\n            chosen_low_level_spec=chosen_low_level_spec,\n            player_view=self._current_context.player_view,\n            full_state_json_before=self._current_context.full_state_json,\n            metadata={\n                \"matchup\": self._matchup.key,\n                \"action_label\": _action_label(self._current_built_context, chosen_action_code),\n                **terminal_context.metadata,\n            },\n        )\n"""
    new = """        trajectory_step = TrajectoryStep(\n            acting_player=current_context_for_step.acting_player,\n            request_type=current_context_for_step.request_type,\n            choice=normalized_choice,\n            pre_state=current_context_for_step.full_state,\n            post_state=terminal_context.full_state,\n            reward=reward,\n            done=done,\n            legal_low_level_codes=current_context_for_step.legal_low_level_codes,\n            legal_low_level_specs=current_context_for_step.legal_low_level_specs,\n            legal_low_level_mask=current_context_for_step.legal_low_level_mask,\n            legal_high_level_codes=current_context_for_step.legal_high_level_codes,\n            high_to_low_map=current_context_for_step.high_to_low_map,\n            chosen_high_level_code=(\n                current_built_context_for_step.low_to_high_code.get(chosen_action_code)\n                if current_built_context_for_step is not None\n                else None\n            ),\n            chosen_low_level_spec=chosen_low_level_spec,\n            player_view=current_context_for_step.player_view,\n            full_state_json_before=current_context_for_step.full_state_json,\n            metadata={\n                \"matchup\": self._matchup.key,\n                \"action_label\": _action_label(current_built_context_for_step, chosen_action_code),\n                **terminal_context.metadata,\n            },\n        )\n"""
    if old not in text:
        raise RuntimeError("env.py: could not find trajectory_step block")
    text = text.replace(old, new, 1)

    if text == original:
        print("env.py: already patched?")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"Patched {path}")


def patch_ppo(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    # 1) when building samples, trust source_step chosen_high only if legal in that step
    old = """                    chosen_high_level_code=int(\n                        transition.source_step.chosen_high_level_code\n                        if transition.source_step.chosen_high_level_code is not None\n                        else encoded.low_to_high_codes[transition.action_index]\n                    ),\n"""
    new = """                    chosen_high_level_code=int(\n                        transition.source_step.chosen_high_level_code\n                        if (\n                            transition.source_step.chosen_high_level_code is not None\n                            and int(transition.source_step.chosen_high_level_code)\n                            in {int(code) for code in transition.source_step.legal_high_level_codes}\n                        )\n                        else encoded.low_to_high_codes[transition.action_index]\n                    ),\n"""
    if old not in text:
        raise RuntimeError("ppo_training.py: could not find chosen_high_level_code sample construction block")
    text = text.replace(old, new, 1)

    # 2) loss side: require mask-valid high code, not just >= 0
    old = """        step_high_log_probs = masked_log_softmax(output.high_policy_logits[local_indices], current_high_mask)\n        step_high_codes = chosen_high_level_code.index_select(0, loss_indices)[:, step_index]\n        valid_high = step_high_codes >= 0\n        if torch.any(valid_high):\n            selected_high_logprob = torch.gather(\n                step_high_log_probs[valid_high],\n                1,\n                step_high_codes[valid_high].unsqueeze(-1),\n            ).squeeze(-1)\n            high_policy_loss = -_weighted_mean(\n                selected_high_logprob,\n                step_weights[valid_high],\n            )\n        else:\n            high_policy_loss = selected_logprob.new_zeros(())\n"""
    new = """        step_high_log_probs = masked_log_softmax(output.high_policy_logits[local_indices], current_high_mask)\n        step_high_codes = chosen_high_level_code.index_select(0, loss_indices)[:, step_index]\n        valid_high = (\n            (step_high_codes >= 0)\n            & (step_high_codes < current_high_mask.shape[1])\n            & torch.gather(current_high_mask, 1, step_high_codes.clamp_min(0).unsqueeze(-1)).squeeze(-1)\n        )\n        if torch.any(valid_high):\n            selected_high_logprob = torch.gather(\n                step_high_log_probs[valid_high],\n                1,\n                step_high_codes[valid_high].unsqueeze(-1),\n            ).squeeze(-1)\n            high_policy_loss = -_weighted_mean(\n                selected_high_logprob,\n                step_weights[valid_high],\n            )\n        else:\n            high_policy_loss = selected_logprob.new_zeros(())\n"""
    if old not in text:
        raise RuntimeError("ppo_training.py: could not find high policy loss block")
    text = text.replace(old, new, 1)

    if text == original:
        print("ppo_training.py: already patched?")
    else:
        path.write_text(text, encoding="utf-8")
        print(f"Patched {path}")


def main() -> None:
    env_path = find_repo_file(ENV_REL)
    ppo_path = find_repo_file(PPO_REL)
    backup(env_path, ".bak_high_policy_fix")
    backup(ppo_path, ".bak_high_policy_fix")
    patch_env(env_path)
    patch_ppo(ppo_path)
    print("Done.")


if __name__ == "__main__":
    main()
