from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .ppo_features import TokenObservationEncoder
from .ppo_model import PpoModelConfig, PpoTransformerPolicy


def _model_config_from_payload(payload: dict[str, Any]) -> PpoModelConfig:
    config_payload = dict(payload["model_config"])
    if "belief_deck_histogram_dim" not in config_payload:
        config_payload["belief_deck_histogram_dim"] = int(config_payload.get("belief_histogram_dim", 0))
    if "belief_group_dim" not in config_payload:
        config_payload["belief_group_dim"] = 10
    if "opponent_tag_vocab_size" not in config_payload:
        encoder_payload = dict(payload.get("encoder_config", {}))
        config_payload["opponent_tag_vocab_size"] = 7 if bool(encoder_payload.get("include_opponent_tag", False)) else 0
    if "opponent_entity_vocab_size" not in config_payload:
        encoder_payload = dict(payload.get("encoder_config", {}))
        config_payload["opponent_entity_vocab_size"] = int(encoder_payload.get("opponent_entity_vocab_size", 257))
    if "deck_vocab_size" not in config_payload:
        encoder_payload = dict(payload.get("encoder_config", {}))
        config_payload["deck_vocab_size"] = int(encoder_payload.get("deck_vocab_size", 65))
    if "recurrent_hidden_dim" not in config_payload:
        config_payload["recurrent_hidden_dim"] = 0
    if "search_range_summary_dim" not in config_payload:
        belief_dim = int(config_payload.get("belief_histogram_dim", 0))
        config_payload["search_range_summary_dim"] = (belief_dim * 2) + 12
    if "search_context_dim" not in config_payload:
        config_payload["search_context_dim"] = 16
    if "high_action_dim" not in config_payload:
        encoder_payload = dict(payload.get("encoder_config", {}))
        config_payload["high_action_dim"] = int(encoder_payload.get("high_action_dim", 0))
    return PpoModelConfig(**config_payload)


def _current_model_config(
    payload: dict[str, Any],
    *,
    encoder: TokenObservationEncoder,
) -> PpoModelConfig:
    config_payload = dict(payload["model_config"])
    config_payload["token_dim"] = int(encoder.token_dim)
    config_payload["option_dim"] = int(encoder.option_dim)
    config_payload["high_action_dim"] = int(encoder.high_action_dim)
    config_payload["privileged_dim"] = int(encoder.privileged_state_dim)
    config_payload["belief_histogram_dim"] = int(encoder.belief_histogram_dim)
    config_payload["belief_deck_histogram_dim"] = int(encoder.belief_remaining_deck_histogram_dim)
    config_payload["max_tokens"] = int(encoder.max_tokens)
    config_payload["belief_group_dim"] = int(encoder.belief_group_dim)
    if "opponent_tag_vocab_size" not in config_payload:
        config_payload["opponent_tag_vocab_size"] = int(encoder.opponent_tag_vocab_size)
    if "opponent_entity_vocab_size" not in config_payload:
        config_payload["opponent_entity_vocab_size"] = int(encoder.opponent_entity_vocab_size)
    if "deck_vocab_size" not in config_payload:
        config_payload["deck_vocab_size"] = int(encoder.deck_vocab_size)
    if "recurrent_hidden_dim" not in config_payload:
        config_payload["recurrent_hidden_dim"] = 0
    if "search_range_summary_dim" not in config_payload:
        config_payload["search_range_summary_dim"] = (int(encoder.belief_histogram_dim) * 2) + 12
    if "search_context_dim" not in config_payload:
        config_payload["search_context_dim"] = 16
    return PpoModelConfig(**config_payload)


def _resize_tensor_like(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    resized = target.detach().clone()
    common_shape = tuple(min(int(a), int(b)) for a, b in zip(resized.shape, source.shape))
    if not common_shape:
        return resized
    target_slices = tuple(slice(0, dim) for dim in common_shape)
    resized[target_slices] = source[target_slices].to(dtype=resized.dtype)
    return resized


def _load_compatible_state_dict(
    model: PpoTransformerPolicy,
    incoming_state: dict[str, torch.Tensor],
) -> None:
    current_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    for key, current_value in current_state.items():
        loaded_value = incoming_state.get(key)
        if isinstance(current_value, torch.Tensor) and isinstance(loaded_value, torch.Tensor):
            if tuple(current_value.shape) == tuple(loaded_value.shape):
                compatible_state[key] = loaded_value
            else:
                compatible_state[key] = _resize_tensor_like(current_value, loaded_value)
        else:
            compatible_state[key] = current_value
    model.load_state_dict(compatible_state, strict=False)


def save_ppo_training_checkpoint(
    path: str | Path,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    *,
    metadata: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "ppo_training",
            "model_config": asdict(model.config),
            "encoder_config": encoder.to_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer_state,
            "metadata": metadata or {},
        },
        destination,
    )


def load_ppo_training_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[PpoTransformerPolicy, TokenObservationEncoder, dict[str, Any], dict[str, Any] | None]:
    payload = torch.load(Path(path), map_location=device)
    encoder = TokenObservationEncoder.from_dict(dict(payload["encoder_config"]))
    model = PpoTransformerPolicy(_current_model_config(payload, encoder=encoder))
    _load_compatible_state_dict(model, payload["state_dict"])
    model.to(device)
    return model, encoder, dict(payload.get("metadata", {})), payload.get("optimizer_state")


def save_ppo_deployment_checkpoint(
    path: str | Path,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    actor_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("oracle_")
    }
    torch.save(
        {
            "kind": "ppo_deployment",
            "model_config": asdict(model.config),
            "encoder_config": encoder.to_dict(),
            "actor_state_dict": actor_state,
            "metadata": metadata or {},
        },
        destination,
    )


def load_ppo_deployment_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[PpoTransformerPolicy, TokenObservationEncoder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    encoder = TokenObservationEncoder.from_dict(dict(payload["encoder_config"]))
    model = PpoTransformerPolicy(_current_model_config(payload, encoder=encoder))
    _load_compatible_state_dict(model, payload["actor_state_dict"])
    model.to(device)
    return model, encoder, dict(payload.get("metadata", {}))
