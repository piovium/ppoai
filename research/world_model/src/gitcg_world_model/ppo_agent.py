from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Sequence

import torch

from .inference_server import GpuInferenceServer
from .schema import EnvConfig
from .public_state import PublicStateTracker
from .ppo_features import TokenObservationEncoder
from .ppo_model import PpoTransformerPolicy, masked_log_softmax
from .sepot_search import SePotSearchController, should_trigger_search
from .schema import ActionChoice, DecisionContext, TrajectoryStep


@dataclass
class PpoAgent:
    model: PpoTransformerPolicy
    encoder: TokenObservationEncoder
    device: str = "cpu"
    sample: bool = True
    temperature: float = 1.0
    player_id: int | None = None
    max_history: int = 32
    opponent_tag: str = "unknown"
    opponent_entity_key: str = "unknown"
    opponent_deck_name: str = "unknown"
    inference_server: GpuInferenceServer | None = None
    model_key: str | None = None
    sepot_controller: SePotSearchController | None = None
    env_config: EnvConfig | None = None
    _device: torch.device = field(init=False, repr=False)
    _history: deque[TrajectoryStep] = field(init=False, repr=False)
    _last_metadata: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _recurrent_state: torch.Tensor | None = field(init=False, default=None, repr=False)
    _public_tracker: PublicStateTracker | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._device = torch.device(self.device)
        self.model.to(self._device)
        self.model.eval()
        self._history = deque(maxlen=self.max_history)
        if self.player_id is not None:
            self._public_tracker = PublicStateTracker(player_id=self.player_id)
        if self.sepot_controller is None:
            self.sepot_controller = SePotSearchController(card_vocabulary=self.encoder.config.card_vocabulary)

    def choose_action(self, context: DecisionContext) -> ActionChoice:
        if not context.legal_low_level_codes:
            raise RuntimeError("no legal low-level actions available")
        if context.player_view is None:
            raise ValueError("ppo agent requires player_view and must not act from full_state")
        if self.player_id is not None and context.acting_player != self.player_id:
            raise ValueError("ppo agent received a context for the wrong player")

        history = tuple(self._history)
        encoded = self._encode_context(context, history=history)
        search_triggered = (
            self.sepot_controller is not None
            and should_trigger_search(context, config=self.sepot_controller.config)
        )

        evaluation = self._evaluate_encoded(
            encoded,
            recurrent_state=self._recurrent_state,
            sample=self.sample,
            temperature=self.temperature,
            use_inference_server=not search_triggered,
        )
        action_index = int(evaluation["action_index"])
        search_latency_ms = 0.0
        search_decision = None
        search_fallback_reason = "not_triggered"
        if search_triggered and self.sepot_controller is not None:
            started_at = time.perf_counter()
            search_decision = self.sepot_controller.choose_action(
                context=context,
                encoded=encoded,
                history=history,
                tracker=self._public_tracker,
                opponent_deck_name=self.opponent_deck_name,
                policy_log_probs=evaluation["log_probs"],
                belief_histogram=evaluation["belief_histogram"],
                belief_remaining_deck_histogram=evaluation["belief_remaining_deck_histogram"],
                model=self.model,
                encoder=self.encoder,
                env_config=self.env_config,
                search_option_logits_fn=self.model.search_option_logits,
                search_state_value_fn=self.model.search_state_value,
                sample=self.sample,
                temperature=self.temperature,
                device=self._device,
            )
            search_latency_ms = (time.perf_counter() - started_at) * 1000.0
            if search_decision is not None:
                action_index = int(search_decision.selected_action_index)
                search_fallback_reason = str(search_decision.fallback_reason or "")
            else:
                search_fallback_reason = "search_unavailable"
        selected_action_code = int(context.legal_low_level_codes[action_index])
        executed_logprob = float(evaluation["log_probs"][action_index])
        if search_decision is not None and search_decision.transformed_policy:
            probability = max(1.0e-8, float(search_decision.transformed_policy[action_index]))
            executed_logprob = float(torch.log(torch.tensor(probability)).item())

        self._last_metadata = {
            "selected_logprob": executed_logprob,
            "partial_value": float(evaluation["partial_value"]),
            "policy_entropy": float(evaluation["policy_entropy"]),
            "chosen_action_code": selected_action_code,
            "chosen_high_action_code": int(encoded.low_to_high_codes[action_index]),
            "chosen_action_index": action_index,
            "policy_action_index": int(action_index),
            "sampled_policy_action_index": int(action_index),
            "opponent_tag": self.opponent_tag,
            "opponent_entity_key": self.opponent_entity_key,
            "opponent_deck_name": self.opponent_deck_name,
            "inference_batch_size": int(evaluation["batch_size"]),
            "sepot_triggered": bool(search_triggered),
            "sepot_fallback_reason": search_fallback_reason,
            "sepot_root_candidate_count": (
                int(search_decision.root_candidate_count) if search_decision is not None else 0
            ),
            "sepot_belief_sample_count": (
                int(search_decision.belief_sample_count) if search_decision is not None else 0
            ),
            "sepot_phase_depth": (int(search_decision.phase_depth) if search_decision is not None else 0),
            "sepot_root_value": (float(search_decision.root_value) if search_decision is not None else 0.0),
            "sepot_latency_ms": float(search_latency_ms),
        }
        if search_decision is not None and not search_decision.fallback_reason:
            self._last_metadata["search_teacher_policy"] = list(search_decision.transformed_policy)
            self._last_metadata["search_teacher_value"] = float(search_decision.root_value)
            self._last_metadata["search_teacher_weight"] = 1.0
            self._last_metadata["search_acted"] = 1.0
            self._last_metadata["sampled_policy_action_index"] = int(search_decision.selected_action_index)
        else:
            self._last_metadata["search_teacher_weight"] = 0.0
            self._last_metadata["search_acted"] = 0.0

        next_recurrent_state = evaluation.get("recurrent_state")
        if next_recurrent_state is not None:
            self._recurrent_state = next_recurrent_state

        return ActionChoice(action_code=selected_action_code)

    def observe_step(self, step: TrajectoryStep) -> None:
        if self._public_tracker is not None:
            self._public_tracker.observe_step(step)
        self._history.append(step)

    def reset_history(self) -> None:
        self._history.clear()
        self._last_metadata = {}
        self._recurrent_state = None
        if self._public_tracker is not None:
            self._public_tracker.reset()

    def pop_last_decision_metadata(self) -> dict[str, Any]:
        payload = dict(self._last_metadata)
        self._last_metadata = {}
        return payload

    def _encode_context(
        self,
        context: DecisionContext,
        *,
        history: Sequence[TrajectoryStep],
    ) -> Any:
        opponent_public_dice_count = (
            self._public_tracker.opponent_public_dice_count(context)
            if self._public_tracker is not None
            else None
        )
        return self.encoder.encode_context(
            context,
            history=history,
            opponent_tag=self.opponent_tag,
            opponent_entity_key=self.opponent_entity_key,
            opponent_deck_name=self.opponent_deck_name,
            opponent_public_dice_count=opponent_public_dice_count,
            include_training_targets=False,
        )

    def _evaluate_encoded(
        self,
        encoded: Any,
        *,
        recurrent_state: torch.Tensor | None,
        sample: bool,
        temperature: float,
        use_inference_server: bool,
    ) -> dict[str, Any]:
        if self.inference_server is not None and use_inference_server:
            model_key = self.model_key or f"server:{id(self.model)}"
            result = self.inference_server.infer(
                model_key=model_key,
                token_features=encoded.token_features,
                token_mask=encoded.token_mask,
                opponent_token_mask=encoded.opponent_token_mask,
                option_features=encoded.option_features,
                option_mask=encoded.option_mask,
                opponent_tag_id=encoded.opponent_tag_id,
                opponent_entity_id=encoded.opponent_entity_id,
                opponent_deck_id=encoded.opponent_deck_id,
                recurrent_state=(
                    tuple(float(value) for value in recurrent_state.detach().cpu().tolist())
                    if recurrent_state is not None
                    else None
                ),
                sample=sample,
                temperature=temperature,
            )
            return {
                "action_index": int(result.action_index),
                "log_probs": tuple(float(value) for value in result.log_probs),
                "partial_value": float(result.partial_value),
                "policy_entropy": float(result.policy_entropy),
                "batch_size": int(result.batch_size),
                "belief_histogram": tuple(float(value) for value in result.belief_histogram),
                "belief_remaining_deck_histogram": tuple(
                    float(value) for value in result.belief_remaining_deck_histogram
                ),
                "high_policy_logits": tuple(float(value) for value in result.high_policy_logits),
                "recurrent_state": (
                    torch.as_tensor(result.recurrent_state, dtype=torch.float32, device=self._device)
                    if result.recurrent_state is not None
                    else None
                ),
            }

        token_features = torch.as_tensor(encoded.token_features, dtype=torch.float32, device=self._device).unsqueeze(0)
        token_mask = torch.as_tensor(encoded.token_mask, dtype=torch.bool, device=self._device).unsqueeze(0)
        opponent_token_mask = torch.as_tensor(encoded.opponent_token_mask, dtype=torch.bool, device=self._device).unsqueeze(0)
        option_features = torch.as_tensor(encoded.option_features, dtype=torch.float32, device=self._device).unsqueeze(0)
        option_mask = torch.as_tensor(encoded.option_mask, dtype=torch.bool, device=self._device).unsqueeze(0)
        opponent_tag_ids = torch.as_tensor([encoded.opponent_tag_id], dtype=torch.long, device=self._device)
        opponent_entity_ids = torch.as_tensor([encoded.opponent_entity_id], dtype=torch.long, device=self._device)
        opponent_deck_ids = torch.as_tensor([encoded.opponent_deck_id], dtype=torch.long, device=self._device)

        with torch.no_grad():
            output = self.model(
                token_features,
                token_mask,
                option_features,
                option_mask,
                opponent_token_mask=opponent_token_mask,
                opponent_tag_ids=opponent_tag_ids,
                opponent_entity_ids=opponent_entity_ids,
                opponent_deck_ids=opponent_deck_ids,
                recurrent_state=(recurrent_state.unsqueeze(0) if recurrent_state is not None else None),
            )

        logits = output.policy_logits
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        log_probs = masked_log_softmax(logits, option_mask)
        probs = torch.exp(log_probs)
        if sample:
            action_index = int(torch.multinomial(probs[0], num_samples=1).item())
        else:
            action_index = int(torch.argmax(log_probs[0]).item())

        return {
            "action_index": action_index,
            "log_probs": tuple(float(value) for value in log_probs[0].detach().cpu().tolist()),
            "partial_value": float(output.partial_value[0].item()),
            "policy_entropy": float((-(probs * log_probs).sum(dim=-1)).item()),
            "batch_size": 1,
            "belief_histogram": tuple(float(value) for value in output.belief_histogram[0].detach().cpu().tolist()),
            "belief_remaining_deck_histogram": tuple(
                float(value) for value in output.belief_remaining_deck_histogram[0].detach().cpu().tolist()
            ),
            "high_policy_logits": tuple(float(value) for value in output.high_policy_logits[0].detach().cpu().tolist()),
            "recurrent_state": (
                output.recurrent_state[0].detach() if output.recurrent_state is not None else None
            ),
        }
