from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .action_quality import (
    declare_end_waste_ratio,
    premature_end_severity,
)
from .action_hierarchy import (
    aggregate_high_policy,
    high_action_vocab_size,
    legal_low_level_specs,
    selected_low_level_code,
)
from .lookahead_search import LookaheadSearchConfig, annotate_episodes_with_search_targets
from .public_state import PublicStateTracker
from .ppo_checkpoint import (
    load_ppo_deployment_checkpoint,
    load_ppo_training_checkpoint,
    save_ppo_deployment_checkpoint,
    save_ppo_training_checkpoint,
)
from .ppo_features import EncodedObservation, TokenObservationEncoder
from .ppo_model import PpoModelConfig, PpoTransformerPolicy, masked_log_softmax
from .progress import atomic_write_json
from .sepot_search import (
    SePotSearchConfig,
    build_public_belief_state,
    phase_budget_for_context,
    range_summary,
    state_search_context_summary,
)
from .schema import DecisionContext, EnvConfig, EpisodeRecord, TrajectoryStep


@dataclass(frozen=True)
class PpoTrainSample:
    token_features: tuple[tuple[float, ...], ...]
    token_mask: tuple[bool, ...]
    opponent_token_mask: tuple[bool, ...]
    option_features: tuple[tuple[float, ...], ...]
    option_mask: tuple[bool, ...]
    low_level_action_codes: tuple[int, ...]
    low_to_high_codes: tuple[int, ...]
    high_level_mask: tuple[bool, ...]
    opponent_tag_id: int
    opponent_entity_id: int
    opponent_deck_id: int
    action_index: int
    chosen_low_level_code: int
    chosen_high_level_code: int
    old_logprob: float
    return_target: float
    advantage: float
    privileged_state: tuple[float, ...]
    belief_histogram_target: tuple[float, ...]
    belief_hand_size_target: float
    belief_remaining_deck_histogram_target: tuple[float, ...]
    belief_burst_ready_count_target: float
    belief_group_hand_target: tuple[float, ...]
    belief_group_deck_target: tuple[float, ...]
    search_self_range_summary: tuple[float, ...] = ()
    search_opponent_range_summary: tuple[float, ...] = ()
    search_context_summary: tuple[float, ...] = ()
    search_teacher_policy_target: tuple[float, ...] = ()
    search_teacher_high_policy_target: tuple[float, ...] = ()
    search_teacher_value_target: float = 0.0
    search_teacher_weight: float = 0.0
    search_acted: float = 0.0
    sample_weight: float = 1.0
    round_number: int = 0
    terminal_draw: bool = False
    terminal_loss: bool = False
    close_game: bool = False


@dataclass(frozen=True)
class PpoTrainChunk:
    steps: tuple[PpoTrainSample, ...]
    burn_in: int
    loss_length: int
    has_late_round: bool
    has_terminal_draw: bool
    has_terminal_loss: bool
    has_close_game: bool


@dataclass(frozen=True)
class PpoTrainingMetrics:
    total_loss: float
    policy_loss: float
    high_policy_loss: float
    fresh_policy_loss: float
    reference_kl_loss: float
    value_loss: float
    replay_value_loss: float
    entropy: float
    belief_loss: float
    replay_belief_loss: float
    belief_hand_loss: float
    belief_deck_loss: float
    belief_burst_loss: float
    belief_group_hand_loss: float
    belief_group_deck_loss: float
    search_distill_loss: float
    search_policy_loss: float
    search_value_loss: float
    oracle_loss: float
    oracle_distill_loss: float
    approx_kl: float
    late_round_sample_ratio: float
    draw_chunk_ratio: float
    loss_chunk_ratio: float


@dataclass(frozen=True)
class PpoTrainArtifacts:
    output_dir: str
    training_checkpoint: str
    deployment_checkpoint: str
    summary_path: str
    best_epoch: int
    best_score: float
    device: str
    sample_count: int


PpoTrainingProgressCallback = Callable[[int, int, PpoTrainingMetrics, bool], None]


@dataclass(frozen=True)
class _DecisionTransition:
    encoded: EncodedObservation
    source_step: TrajectoryStep
    search_self_range_summary: tuple[float, ...]
    search_opponent_range_summary: tuple[float, ...]
    search_context_summary: tuple[float, ...]
    action_index: int
    old_logprob: float
    value: float
    reward: float
    discount: float
    next_value: float
    round_number: int
    terminal_draw: bool
    terminal_loss: bool
    close_game: bool


def build_ppo_samples(
    episodes: Sequence[EpisodeRecord],
    encoder: TokenObservationEncoder,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    include_old_logprob: bool = True,
) -> list[PpoTrainSample]:
    samples: list[PpoTrainSample] = []
    for episode in episodes:
        player_sequences = _build_player_sequences(
            episode=episode,
            encoder=encoder,
            gamma=gamma,
            gae_lambda=gae_lambda,
            include_old_logprob=include_old_logprob,
        )
        for sequence in player_sequences.values():
            samples.extend(sequence)
    return samples


def build_ppo_chunks(
    episodes: Sequence[EpisodeRecord],
    encoder: TokenObservationEncoder,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    include_old_logprob: bool = True,
    chunk_length: int = 16,
    burn_in: int = 4,
    train_length: int = 12,
) -> list[PpoTrainChunk]:
    chunks: list[PpoTrainChunk] = []
    for episode in episodes:
        player_sequences = _build_player_sequences(
            episode=episode,
            encoder=encoder,
            gamma=gamma,
            gae_lambda=gae_lambda,
            include_old_logprob=include_old_logprob,
        )
        for sequence in player_sequences.values():
            chunks.extend(
                _chunk_player_sequence(
                    sequence,
                    chunk_length=chunk_length,
                    burn_in=burn_in,
                    train_length=train_length,
                )
            )
    return chunks


def bootstrap_pretrain_from_episodes(
    *,
    episodes: Sequence[EpisodeRecord],
    output_dir: str | Path,
    init_checkpoint: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    micro_batch_size: int | None = None,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float | None = 1.0,
    use_amp: bool | None = None,
    chunk_length: int = 16,
    burn_in: int = 4,
    train_length: int = 12,
    progress_callback: PpoTrainingProgressCallback | None = None,
) -> PpoTrainArtifacts:
    encoder = _encoder_for_init_checkpoint(init_checkpoint)
    fresh_chunks = build_ppo_chunks(
        episodes,
        encoder,
        include_old_logprob=False,
        chunk_length=chunk_length,
        burn_in=burn_in,
        train_length=train_length,
    )
    return _fit_actor_critic(
        fresh_chunks=fresh_chunks,
        replay_chunks=(),
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        reference_checkpoint=None,
        device=device,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        use_amp=use_amp,
        objective="behavior_clone",
        chunk_length=chunk_length,
        burn_in=burn_in,
        train_length=train_length,
        progress_callback=progress_callback,
    )


def train_ppo_from_episodes(
    *,
    episodes: Sequence[EpisodeRecord],
    replay_episodes: Sequence[EpisodeRecord] | None = None,
    output_dir: str | Path,
    init_checkpoint: str | Path | None = None,
    env_config: EnvConfig | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    micro_batch_size: int | None = None,
    epochs: int = 12,
    replay_epochs: int = 2,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float | None = 1.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    policy_coef: float = 4.0,
    value_coef: float = 0.5,
    belief_coef: float = 0.2,
    oracle_coef: float = 0.2,
    use_amp: bool | None = None,
    chunk_length: int = 16,
    burn_in: int = 4,
    train_length: int = 12,
    reference_kl_coef: float = 0.005,
    target_kl_low: float = 0.01,
    target_kl_high: float = 0.05,
    search_teacher_config: LookaheadSearchConfig | None = None,
    progress_callback: PpoTrainingProgressCallback | None = None,
) -> PpoTrainArtifacts:
    encoder = _encoder_for_init_checkpoint(init_checkpoint)
    resolved_search_config = search_teacher_config or LookaheadSearchConfig()
    training_episodes = tuple(episodes)
    if resolved_search_config.enabled and env_config is not None:
        teacher_model = _load_search_teacher_model(
            init_checkpoint=init_checkpoint,
            encoder=encoder,
            device=device,
        )
        training_episodes = annotate_episodes_with_search_targets(
            training_episodes,
            model=teacher_model,
            encoder=encoder,
            device=torch.device(device),
            env_config=env_config,
            config=resolved_search_config,
        )
    fresh_chunks = build_ppo_chunks(
        training_episodes,
        encoder,
        gamma=gamma,
        gae_lambda=gae_lambda,
        include_old_logprob=True,
        chunk_length=chunk_length,
        burn_in=burn_in,
        train_length=train_length,
    )
    replay_chunks = _select_aux_replay_chunks(
        build_ppo_chunks(
            tuple(replay_episodes or ()),
            encoder,
            gamma=gamma,
            gae_lambda=gae_lambda,
            include_old_logprob=True,
            chunk_length=chunk_length,
            burn_in=burn_in,
            train_length=train_length,
        )
    )
    return _fit_actor_critic(
        fresh_chunks=fresh_chunks,
        replay_chunks=replay_chunks,
        output_dir=output_dir,
        init_checkpoint=init_checkpoint,
        reference_checkpoint=init_checkpoint,
        device=device,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        epochs=epochs,
        replay_epochs=replay_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        objective="ppo",
        entropy_coef=entropy_coef,
        policy_coef=policy_coef,
        value_coef=value_coef,
        belief_coef=belief_coef,
        oracle_coef=oracle_coef,
        use_amp=use_amp,
        chunk_length=chunk_length,
        burn_in=burn_in,
        train_length=train_length,
        reference_kl_coef=reference_kl_coef,
        target_kl_low=target_kl_low,
        target_kl_high=target_kl_high,
        progress_callback=progress_callback,
    )


def load_ppo_train_artifacts(output_dir: str | Path) -> PpoTrainArtifacts | None:
    destination = Path(output_dir)
    summary_path = destination / "training_summary.json"
    training_checkpoint = destination / "last.pt"
    deployment_checkpoint = destination / "deployment.pt"
    if not summary_path.exists() or not training_checkpoint.exists() or not deployment_checkpoint.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return PpoTrainArtifacts(
        output_dir=str(destination),
        training_checkpoint=str(training_checkpoint),
        deployment_checkpoint=str(deployment_checkpoint),
        summary_path=str(summary_path),
        best_epoch=int(payload["best_epoch"]),
        best_score=float(payload["best_score"]),
        device=str(payload["device"]),
        sample_count=int(payload["sample_count"]),
    )


def _encoder_for_init_checkpoint(init_checkpoint: str | Path | None) -> TokenObservationEncoder:
    if init_checkpoint is None:
        return TokenObservationEncoder()
    checkpoint_path = Path(init_checkpoint)
    if not checkpoint_path.exists():
        return TokenObservationEncoder()
    try:
        _, encoder, _, _ = load_ppo_training_checkpoint(checkpoint_path, device="cpu")
        return encoder
    except Exception:
        try:
            _, encoder, _ = load_ppo_deployment_checkpoint(checkpoint_path, device="cpu")
            return encoder
        except Exception:
            return TokenObservationEncoder()


def _load_search_teacher_model(
    *,
    init_checkpoint: str | Path | None,
    encoder: TokenObservationEncoder,
    device: str,
) -> PpoTransformerPolicy:
    if init_checkpoint is not None:
        checkpoint_path = Path(init_checkpoint)
        if checkpoint_path.exists():
            try:
                model, _, _, _ = load_ppo_training_checkpoint(checkpoint_path, device=device)
                model.eval()
                return model
            except Exception:
                try:
                    model, _, _ = load_ppo_deployment_checkpoint(checkpoint_path, device=device)
                    model.eval()
                    return model
                except Exception:
                    pass
    resolved_device = torch.device(device)
    model = PpoTransformerPolicy(
        PpoModelConfig(
            token_dim=encoder.token_dim,
            option_dim=encoder.option_dim,
            high_action_dim=encoder.high_action_dim,
            privileged_dim=encoder.privileged_state_dim,
            belief_histogram_dim=encoder.belief_histogram_dim,
            belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
            max_tokens=encoder.max_tokens,
            belief_group_dim=encoder.belief_group_dim,
            opponent_tag_vocab_size=encoder.opponent_tag_vocab_size if encoder.config.include_opponent_tag else 0,
            opponent_entity_vocab_size=encoder.opponent_entity_vocab_size,
            deck_vocab_size=encoder.deck_vocab_size,
            search_range_summary_dim=(encoder.belief_histogram_dim * 2) + 12,
            search_context_dim=16,
        )
    )
    model.to(resolved_device)
    model.eval()
    return model


def _build_player_sequences(
    *,
    episode: EpisodeRecord,
    encoder: TokenObservationEncoder,
    gamma: float,
    gae_lambda: float,
    include_old_logprob: bool,
) -> dict[int, list[PpoTrainSample]]:
    encoded_steps: dict[int, dict[str, object]] = {}
    trainable_by_player: dict[int, list[int]] = {0: [], 1: []}
    history: list[TrajectoryStep] = []
    public_trackers = {
        0: PublicStateTracker(player_id=0),
        1: PublicStateTracker(player_id=1),
    }
    search_config = SePotSearchConfig()
    zero_belief_histogram = tuple(0.0 for _ in range(encoder.belief_histogram_dim))
    zero_belief_deck_histogram = tuple(0.0 for _ in range(encoder.belief_remaining_deck_histogram_dim))
    for step_index, step in enumerate(episode.steps):
        context = _context_from_step(step, episode_metadata=episode.metadata)
        is_trainable = bool(step.metadata.get("ppo_trainable", True))
        if is_trainable:
            tracker = public_trackers[step.acting_player]
            opponent_deck_name = str(
                context.metadata.get("opponent_deck_name")
                or ""
            )
            encoded = encoder.encode_context(
                context,
                history=tuple(history[-encoder.config.max_history_tokens :]),
                opponent_tag=str(context.metadata.get("opponent_tag", "unknown")),
                opponent_entity_key=str(context.metadata.get("opponent_entity_key", "unknown")),
                opponent_deck_name=opponent_deck_name,
                opponent_public_dice_count=tracker.opponent_public_dice_count(context),
            )
            budget = phase_budget_for_context(context, config=search_config)
            public_belief = build_public_belief_state(
                context=context,
                tracker=tracker,
                card_vocabulary=encoder.config.card_vocabulary,
                opponent_deck_name=(opponent_deck_name or "unknown"),
                belief_histogram=zero_belief_histogram,
                belief_remaining_deck_histogram=zero_belief_deck_histogram,
                belief_samples=budget.belief_samples,
            )
            encoded_steps[step_index] = {
                "encoded": encoded,
                "action_index": _selected_action_index(step),
                "old_logprob": float(step.metadata.get("selected_logprob", 0.0))
                if include_old_logprob
                else 0.0,
                "value": float(step.metadata.get("partial_value", 0.0)),
                "search_self_range_summary": range_summary(
                    public_belief.self_range,
                    card_vocabulary=encoder.config.card_vocabulary,
                ),
                "search_opponent_range_summary": range_summary(
                    public_belief.opponent_range,
                    card_vocabulary=encoder.config.card_vocabulary,
                ),
                "search_context_summary": state_search_context_summary(
                    context=context,
                    budget=budget,
                ),
            }
            trainable_by_player[step.acting_player].append(step_index)
        for tracker in public_trackers.values():
            tracker.observe_step(step)
        history.append(step)

    result: dict[int, list[PpoTrainSample]] = {0: [], 1: []}
    for player, indices in trainable_by_player.items():
        if not indices:
            continue
        transitions: list[_DecisionTransition] = []
        value_lookup = {index: float(encoded_steps[index]["value"]) for index in indices}
        for offset, step_index in enumerate(indices):
            next_index = indices[offset + 1] if offset + 1 < len(indices) else None
            reward, discount = _accumulate_reward_until_next_trainable(
                episode.steps,
                start_index=step_index,
                next_trainable_index=next_index,
                perspective_player=player,
                gamma=gamma,
            )
            item = encoded_steps[step_index]
            source_step = episode.steps[step_index]
            transitions.append(
                _DecisionTransition(
                    encoded=item["encoded"],  # type: ignore[arg-type]
                    source_step=source_step,
                    search_self_range_summary=tuple(float(value) for value in item["search_self_range_summary"]),  # type: ignore[index]
                    search_opponent_range_summary=tuple(float(value) for value in item["search_opponent_range_summary"]),  # type: ignore[index]
                    search_context_summary=tuple(float(value) for value in item["search_context_summary"]),  # type: ignore[index]
                    action_index=int(item["action_index"]),
                    old_logprob=float(item["old_logprob"]),
                    value=float(item["value"]),
                    reward=reward,
                    discount=discount,
                    next_value=(value_lookup.get(next_index, 0.0) if next_index is not None else 0.0),
                    round_number=source_step.pre_state.round_number,
                    terminal_draw=bool(source_step.done and episode.winner is None),
                    terminal_loss=bool(source_step.done and episode.winner is not None and episode.winner != player),
                    close_game=abs(_potential(source_step.pre_state, perspective_player=player)) <= 0.10,
                )
            )
        returns, advantages = _gae_returns(transitions, gae_lambda=gae_lambda)
        for transition, return_target, advantage in zip(transitions, returns, advantages):
            encoded = transition.encoded
            transition_context = _context_from_step(transition.source_step, episode_metadata=episode.metadata)
            search_teacher_policy_target = tuple(
                float(value) for value in transition.source_step.metadata.get("search_teacher_policy", ())
            )
            result[player].append(
                PpoTrainSample(
                    token_features=encoded.token_features,
                    token_mask=encoded.token_mask,
                    opponent_token_mask=encoded.opponent_token_mask,
                    option_features=encoded.option_features,
                    option_mask=encoded.option_mask,
                    low_level_action_codes=encoded.low_level_action_codes,
                    low_to_high_codes=encoded.low_to_high_codes,
                    high_level_mask=encoded.high_level_mask,
                    opponent_tag_id=encoded.opponent_tag_id,
                    opponent_entity_id=encoded.opponent_entity_id,
                    opponent_deck_id=encoded.opponent_deck_id,
                    action_index=transition.action_index,
                    chosen_low_level_code=int(
                        transition.source_step.choice.action_code
                        if transition.source_step.choice.action_code >= 0
                        else encoded.low_level_action_codes[transition.action_index]
                    ),
                    chosen_high_level_code=int(
                        transition.source_step.chosen_high_level_code
                        if transition.source_step.chosen_high_level_code is not None
                        else encoded.low_to_high_codes[transition.action_index]
                    ),
                    old_logprob=transition.old_logprob,
                    return_target=float(return_target),
                    advantage=float(advantage),
                    privileged_state=encoded.privileged_state,
                    belief_histogram_target=encoded.belief_target.hand_histogram,
                    belief_hand_size_target=encoded.belief_target.hand_size,
                    belief_remaining_deck_histogram_target=encoded.belief_target.remaining_deck_histogram,
                    belief_burst_ready_count_target=encoded.belief_target.burst_ready_count,
                    belief_group_hand_target=encoded.belief_target.group_hand_presence,
                    belief_group_deck_target=encoded.belief_target.group_remaining_deck_presence,
                    search_self_range_summary=transition.search_self_range_summary,
                    search_opponent_range_summary=transition.search_opponent_range_summary,
                    search_context_summary=transition.search_context_summary,
                    search_teacher_policy_target=search_teacher_policy_target,
                    search_teacher_high_policy_target=aggregate_high_policy(
                        context=transition_context,
                        low_level_policy=search_teacher_policy_target,
                    )
                    if search_teacher_policy_target
                    else tuple(0.0 for _ in range(high_action_vocab_size())),
                    search_teacher_value_target=float(transition.source_step.metadata.get("search_teacher_value", 0.0)),
                    search_teacher_weight=float(transition.source_step.metadata.get("search_teacher_weight", 0.0)),
                    search_acted=float(transition.source_step.metadata.get("search_acted", 0.0)),
                    sample_weight=_sample_weight(
                        episode=episode,
                        perspective_player=player,
                        round_number=transition.round_number,
                    ),
                    round_number=transition.round_number,
                    terminal_draw=transition.terminal_draw,
                    terminal_loss=transition.terminal_loss,
                    close_game=transition.close_game,
                )
            )
    return result


def _chunk_player_sequence(
    sequence: Sequence[PpoTrainSample],
    *,
    chunk_length: int,
    burn_in: int,
    train_length: int,
) -> list[PpoTrainChunk]:
    if not sequence:
        return []
    chunks: list[PpoTrainChunk] = []
    stride = max(1, train_length)
    total = len(sequence)
    start = 0
    while start < total:
        end = min(total, start + chunk_length)
        steps = tuple(sequence[start:end])
        if not steps:
            break
        local_burn_in = min(burn_in, max(0, len(steps) - 1))
        loss_length = max(1, len(steps) - local_burn_in)
        chunks.append(
            PpoTrainChunk(
                steps=steps,
                burn_in=local_burn_in,
                loss_length=loss_length,
                has_late_round=any(step.round_number >= 13 for step in steps),
                has_terminal_draw=any(step.terminal_draw for step in steps),
                has_terminal_loss=any(step.terminal_loss for step in steps),
                has_close_game=any(step.close_game for step in steps),
            )
        )
        start += stride
    return chunks


def _select_aux_replay_chunks(chunks: Sequence[PpoTrainChunk]) -> tuple[PpoTrainChunk, ...]:
    return tuple(
        chunk
        for chunk in chunks
        if chunk.has_late_round or chunk.has_terminal_draw or chunk.has_terminal_loss or chunk.has_close_game
    )


def _fit_actor_critic(
    *,
    fresh_chunks: Sequence[PpoTrainChunk],
    replay_chunks: Sequence[PpoTrainChunk],
    output_dir: str | Path,
    init_checkpoint: str | Path | None,
    reference_checkpoint: str | Path | None,
    device: str,
    batch_size: int,
    micro_batch_size: int | None,
    epochs: int,
    replay_epochs: int = 0,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float | None,
    use_amp: bool | None,
    objective: str,
    entropy_coef: float = 0.01,
    policy_coef: float = 4.0,
    value_coef: float = 0.5,
    belief_coef: float = 0.2,
    oracle_coef: float = 0.2,
    chunk_length: int = 16,
    burn_in: int = 4,
    train_length: int = 12,
    reference_kl_coef: float = 0.005,
    target_kl_low: float = 0.01,
    target_kl_high: float = 0.05,
    progress_callback: PpoTrainingProgressCallback | None = None,
) -> PpoTrainArtifacts:
    if not fresh_chunks:
        raise RuntimeError("no PPO chunks available")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if init_checkpoint is not None:
        try:
            model, encoder, parent_metadata, optimizer_state = load_ppo_training_checkpoint(
                init_checkpoint,
                device=device,
            )
        except Exception:
            model, encoder, parent_metadata = load_ppo_deployment_checkpoint(init_checkpoint, device=device)
            optimizer_state = None
    else:
        encoder = TokenObservationEncoder()
        model = PpoTransformerPolicy(
            PpoModelConfig(
                token_dim=encoder.token_dim,
                option_dim=encoder.option_dim,
                high_action_dim=encoder.high_action_dim,
                privileged_dim=encoder.privileged_state_dim,
                belief_histogram_dim=encoder.belief_histogram_dim,
                belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
                max_tokens=encoder.max_tokens,
                belief_group_dim=encoder.belief_group_dim,
                opponent_tag_vocab_size=encoder.opponent_tag_vocab_size if encoder.config.include_opponent_tag else 0,
                opponent_entity_vocab_size=encoder.opponent_entity_vocab_size,
                deck_vocab_size=encoder.deck_vocab_size,
                search_range_summary_dim=(encoder.belief_histogram_dim * 2) + 12,
                search_context_dim=16,
            )
        )
        model.to(resolved_device)
        parent_metadata = {}
        optimizer_state = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    resolved_use_amp = bool(use_amp) if use_amp is not None else resolved_device.type == "cuda"
    resolved_micro_batch_size = _resolve_micro_batch_size(
        batch_size=batch_size,
        requested_micro_batch_size=micro_batch_size,
        param_count=sum(parameter.numel() for parameter in model.parameters()),
        device=resolved_device,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=resolved_use_amp and resolved_device.type == "cuda")
    reference_model = _load_reference_model(reference_checkpoint, device=device) if objective == "ppo" else None
    fresh_loader = _build_chunk_loader(fresh_chunks, batch_size=batch_size, objective=objective)
    replay_loader = (
        _build_chunk_loader(replay_chunks, batch_size=batch_size, objective="aux_replay")
        if replay_chunks
        else None
    )

    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    latest_checkpoint = destination / "last.pt"
    deployment_checkpoint = destination / "deployment.pt"
    candidate_dir = destination / "candidates"
    candidate_epochs: list[int] = []
    if objective == "ppo":
        candidate_dir.mkdir(parents=True, exist_ok=True)
    replay_chunk_count = len(replay_chunks)
    static_ratios = _dataset_ratios(fresh_chunks=fresh_chunks, replay_chunks=replay_chunks)
    effective_epochs = int(epochs)

    for epoch_index in range(epochs):
        main_metrics = _run_training_epoch(
            model=model,
            loader=fresh_loader,
            optimizer=optimizer,
            device=resolved_device,
            grad_clip_norm=grad_clip_norm,
            scaler=scaler,
            use_amp=resolved_use_amp,
            micro_batch_size=resolved_micro_batch_size,
            objective=objective,
            entropy_coef=entropy_coef,
            policy_coef=policy_coef if objective == "ppo" else 1.0,
            value_coef=value_coef,
            belief_coef=belief_coef,
            oracle_coef=oracle_coef,
            reference_model=reference_model,
            reference_kl_coef=reference_kl_coef if objective == "ppo" else 0.0,
        )
        replay_value_loss = 0.0
        replay_belief_loss = 0.0
        if replay_loader is not None and replay_epochs > 0:
            replay_value_total = 0.0
            replay_belief_total = 0.0
            for _ in range(replay_epochs):
                aux_metrics = _run_training_epoch(
                    model=model,
                    loader=replay_loader,
                    optimizer=optimizer,
                    device=resolved_device,
                    grad_clip_norm=grad_clip_norm,
                    scaler=scaler,
                    use_amp=resolved_use_amp,
                    micro_batch_size=resolved_micro_batch_size,
                    objective="aux_replay",
                    entropy_coef=0.0,
                    policy_coef=1.0,
                    value_coef=value_coef,
                    belief_coef=belief_coef,
                    oracle_coef=oracle_coef,
                    reference_model=None,
                    reference_kl_coef=0.0,
                )
                replay_value_total += aux_metrics["value_loss"]
                replay_belief_total += aux_metrics["belief_loss"]
            replay_value_loss = replay_value_total / replay_epochs
            replay_belief_loss = replay_belief_total / replay_epochs

        metrics = PpoTrainingMetrics(
            total_loss=main_metrics["total_loss"] + 0.5 * replay_value_loss + 0.2 * replay_belief_loss,
            policy_loss=main_metrics["policy_loss"],
            high_policy_loss=main_metrics.get("high_policy_loss", 0.0),
            fresh_policy_loss=main_metrics["policy_loss"],
            reference_kl_loss=main_metrics["reference_kl_loss"],
            value_loss=main_metrics["value_loss"],
            replay_value_loss=replay_value_loss,
            entropy=main_metrics["entropy"],
            belief_loss=main_metrics["belief_loss"],
            replay_belief_loss=replay_belief_loss,
            belief_hand_loss=main_metrics["belief_hand_loss"],
            belief_deck_loss=main_metrics["belief_deck_loss"],
            belief_burst_loss=main_metrics["belief_burst_loss"],
            belief_group_hand_loss=main_metrics["belief_group_hand_loss"],
            belief_group_deck_loss=main_metrics["belief_group_deck_loss"],
            search_distill_loss=main_metrics.get("search_distill_loss", 0.0),
            search_policy_loss=main_metrics.get("search_policy_loss", 0.0),
            search_value_loss=main_metrics.get("search_value_loss", 0.0),
            oracle_loss=main_metrics["oracle_loss"],
            oracle_distill_loss=main_metrics["oracle_distill_loss"],
            approx_kl=main_metrics["approx_kl"],
            late_round_sample_ratio=static_ratios["late_round_sample_ratio"],
            draw_chunk_ratio=static_ratios["draw_chunk_ratio"],
            loss_chunk_ratio=static_ratios["loss_chunk_ratio"],
        )
        history.append(metrics.__dict__)
        is_best = False
        if metrics.total_loss < best_score:
            best_score = metrics.total_loss
            best_epoch = epoch_index + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            is_best = True
        if progress_callback is not None:
            progress_callback(epoch_index + 1, epochs, metrics, is_best)
        save_ppo_training_checkpoint(
            latest_checkpoint,
            model,
            encoder,
            metadata={
                "objective": objective,
                "epoch": epoch_index + 1,
                "parent_metadata": parent_metadata,
                "history": history,
                "chunk_length": chunk_length,
                "burn_in": burn_in,
                "train_length": train_length,
                "replay_chunk_count": replay_chunk_count,
                "micro_batch_size": resolved_micro_batch_size,
                "use_amp": resolved_use_amp,
            },
            optimizer_state=optimizer.state_dict(),
        )
        if objective == "ppo":
            epoch_number = epoch_index + 1
            candidate_epochs.append(epoch_number)
            save_ppo_deployment_checkpoint(
                candidate_dir / f"epoch_{epoch_number:03d}.pt",
                model,
                encoder,
                metadata={
                    "objective": objective,
                    "epoch": epoch_number,
                    "metrics": metrics.__dict__,
                },
            )
            if metrics.approx_kl > target_kl_high:
                effective_epochs = epoch_number
                break
    model.load_state_dict(best_state)
    save_ppo_deployment_checkpoint(
        deployment_checkpoint,
        model,
        encoder,
        metadata={
            "objective": objective,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
    )
    summary_path = destination / "training_summary.json"
    atomic_write_json(
        summary_path,
        {
            "objective": objective,
            "device": device,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "sample_count": sum(len(chunk.steps) for chunk in fresh_chunks),
            "history": history,
            "candidate_epochs": candidate_epochs,
            "safe_epochs": (
                [index + 1 for index, item in enumerate(history) if all(math.isfinite(float(value)) for value in item.values())]
                if objective != "ppo"
                else []
            ),
            "selection_method": "min_total_loss" if objective != "ppo" else None,
            "selected_epoch": best_epoch if objective != "ppo" else None,
            "chunk_length": chunk_length,
            "burn_in": burn_in,
            "train_length": train_length,
            "replay_chunk_count": replay_chunk_count,
            "replay_epochs": replay_epochs if objective == "ppo" else 0,
            "policy_coef": policy_coef if objective == "ppo" else 1.0,
            "reference_kl_coef": reference_kl_coef if objective == "ppo" else 0.0,
            "target_kl_low": target_kl_low if objective == "ppo" else 0.0,
            "target_kl_high": target_kl_high if objective == "ppo" else 0.0,
            "effective_epochs": effective_epochs,
            "micro_batch_size": resolved_micro_batch_size,
            "use_amp": resolved_use_amp,
            **static_ratios,
        },
    )
    return PpoTrainArtifacts(
        output_dir=str(destination),
        training_checkpoint=str(latest_checkpoint),
        deployment_checkpoint=str(deployment_checkpoint),
        summary_path=str(summary_path),
        best_epoch=best_epoch,
        best_score=best_score,
        device=device,
        sample_count=sum(len(chunk.steps) for chunk in fresh_chunks),
    )


def _build_chunk_loader(
    chunks: Sequence[PpoTrainChunk],
    *,
    batch_size: int,
    objective: str,
) -> DataLoader:
    chunk_list = list(chunks)
    if objective == "ppo":
        weights = [_chunk_sampling_weight(chunk) for chunk in chunk_list]
        sampler = WeightedRandomSampler(weights, num_samples=len(chunk_list), replacement=True)
        return DataLoader(chunk_list, batch_size=batch_size, sampler=sampler, collate_fn=_collate_chunks)
    return DataLoader(chunk_list, batch_size=batch_size, shuffle=True, collate_fn=_collate_chunks)


def _chunk_sampling_weight(chunk: PpoTrainChunk) -> float:
    weight = 1.0
    if chunk.has_late_round:
        weight *= 2.0
    if chunk.has_terminal_draw or chunk.has_terminal_loss:
        weight *= 2.0
    if chunk.has_close_game:
        weight *= 1.5
    return min(3.0, weight)


def _run_training_epoch(
    *,
    model: PpoTransformerPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    micro_batch_size: int,
    objective: str,
    entropy_coef: float,
    policy_coef: float,
    value_coef: float,
    belief_coef: float,
    oracle_coef: float,
    reference_model: PpoTransformerPolicy | None,
    reference_kl_coef: float,
) -> dict[str, float]:
    model.train()
    aggregate = {
        "total_loss": 0.0,
        "policy_loss": 0.0,
        "high_policy_loss": 0.0,
        "reference_kl_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "belief_loss": 0.0,
        "belief_hand_loss": 0.0,
        "belief_deck_loss": 0.0,
        "belief_burst_loss": 0.0,
        "belief_group_hand_loss": 0.0,
        "belief_group_deck_loss": 0.0,
        "search_distill_loss": 0.0,
        "search_policy_loss": 0.0,
        "search_value_loss": 0.0,
        "oracle_loss": 0.0,
        "oracle_distill_loss": 0.0,
        "approx_kl": 0.0,
        "batches": 0,
    }
    for batch in loader:
        batch_loss_steps = int(batch["loss_mask"].sum().item())
        if batch_loss_steps <= 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        batch_metrics = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "high_policy_loss": 0.0,
            "reference_kl_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "belief_loss": 0.0,
            "belief_hand_loss": 0.0,
            "belief_deck_loss": 0.0,
            "belief_burst_loss": 0.0,
            "belief_group_hand_loss": 0.0,
            "belief_group_deck_loss": 0.0,
            "search_distill_loss": 0.0,
            "search_policy_loss": 0.0,
            "search_value_loss": 0.0,
            "oracle_loss": 0.0,
            "oracle_distill_loss": 0.0,
            "approx_kl": 0.0,
        }
        batch_items = int(batch["step_mask"].shape[0])
        for start in range(0, batch_items, micro_batch_size):
            stop = min(batch_items, start + micro_batch_size)
            micro_batch = _slice_chunk_batch(batch, start, stop)
            micro_loss_steps = int(micro_batch["loss_mask"].sum().item())
            if micro_loss_steps <= 0:
                continue
            amp_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                loss, metrics, loss_steps = _compute_chunk_batch_loss(
                    model=model,
                    batch=micro_batch,
                    device=device,
                    objective=objective,
                    entropy_coef=entropy_coef,
                    policy_coef=policy_coef,
                    value_coef=value_coef,
                    belief_coef=belief_coef,
                    oracle_coef=oracle_coef,
                    reference_model=reference_model,
                    reference_kl_coef=reference_kl_coef,
                )
            scale = float(loss_steps) / float(batch_loss_steps)
            scaled_loss = loss * scale
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            for key in batch_metrics:
                batch_metrics[key] += metrics[key] * scale
        if grad_clip_norm is not None:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        for key in aggregate:
            if key == "batches":
                continue
            aggregate[key] += batch_metrics[key]
        aggregate["batches"] += 1
    batches = max(1, int(aggregate["batches"]))
    return {
        key: (aggregate[key] / batches if key != "batches" else float(batches))
        for key in aggregate
        if key != "batches"
    }


def _compute_chunk_batch_loss(
    *,
    model: PpoTransformerPolicy,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    objective: str,
    entropy_coef: float,
    policy_coef: float,
    value_coef: float,
    belief_coef: float,
    oracle_coef: float,
    reference_model: PpoTransformerPolicy | None,
    reference_kl_coef: float,
) -> tuple[torch.Tensor, dict[str, float], int]:
    token_features = batch["token_features"].to(device)
    token_mask = batch["token_mask"].to(device)
    opponent_token_mask = batch["opponent_token_mask"].to(device)
    option_features = batch["option_features"].to(device)
    option_mask = batch["option_mask"].to(device)
    opponent_tag_id = batch["opponent_tag_id"].to(device)
    opponent_entity_id = batch["opponent_entity_id"].to(device)
    opponent_deck_id = batch["opponent_deck_id"].to(device)
    privileged_state = batch["privileged_state"].to(device)
    belief_histogram_target = batch["belief_histogram_target"].to(device)
    belief_hand_size_target = batch["belief_hand_size_target"].to(device)
    belief_remaining_deck_histogram_target = batch["belief_remaining_deck_histogram_target"].to(device)
    belief_burst_ready_count_target = batch["belief_burst_ready_count_target"].to(device)
    belief_group_hand_target = batch["belief_group_hand_target"].to(device)
    belief_group_deck_target = batch["belief_group_deck_target"].to(device)
    search_self_range_summary = batch["search_self_range_summary"].to(device)
    search_opponent_range_summary = batch["search_opponent_range_summary"].to(device)
    search_context_summary = batch["search_context_summary"].to(device)
    search_teacher_policy_target = batch["search_teacher_policy_target"].to(device)
    search_teacher_high_policy_target = batch["search_teacher_high_policy_target"].to(device)
    search_teacher_value_target = batch["search_teacher_value_target"].to(device)
    search_teacher_weight = batch["search_teacher_weight"].to(device)
    search_acted = batch["search_acted"].to(device)
    high_level_mask = batch["high_level_mask"].to(device)
    action_index = batch["action_index"].to(device)
    chosen_high_level_code = batch["chosen_high_level_code"].to(device)
    old_logprob = batch["old_logprob"].to(device)
    return_target = batch["return_target"].to(device)
    advantage = batch["advantage"].to(device)
    sample_weight = batch["sample_weight"].to(device)
    step_mask = batch["step_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    batch_size, seq_len = step_mask.shape
    recurrent_state = (
        token_features.new_zeros((batch_size, model.config.recurrent_hidden_dim))
        if model.config.recurrent_hidden_dim > 0
        else None
    )
    reference_recurrent = (
        token_features.new_zeros((batch_size, reference_model.config.recurrent_hidden_dim))
        if reference_model is not None and reference_model.config.recurrent_hidden_dim > 0
        else None
    )
    metrics = {
        "total_loss": 0.0,
        "policy_loss": 0.0,
        "high_policy_loss": 0.0,
        "reference_kl_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "belief_loss": 0.0,
        "belief_hand_loss": 0.0,
        "belief_deck_loss": 0.0,
        "belief_burst_loss": 0.0,
        "belief_group_hand_loss": 0.0,
        "belief_group_deck_loss": 0.0,
        "search_distill_loss": 0.0,
        "search_policy_loss": 0.0,
        "search_value_loss": 0.0,
        "oracle_loss": 0.0,
        "oracle_distill_loss": 0.0,
        "approx_kl": 0.0,
    }
    total_loss = token_features.new_zeros(())
    loss_steps = 0
    for step_index in range(seq_len):
        active = step_mask[:, step_index]
        if not torch.any(active):
            continue
        active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
        active_recurrent = recurrent_state.index_select(0, active_indices) if recurrent_state is not None else None
        step_kwargs = {
            "token_features": token_features.index_select(0, active_indices)[:, step_index],
            "token_mask": token_mask.index_select(0, active_indices)[:, step_index],
            "option_features": option_features.index_select(0, active_indices)[:, step_index],
            "option_mask": option_mask.index_select(0, active_indices)[:, step_index],
            "opponent_token_mask": opponent_token_mask.index_select(0, active_indices)[:, step_index],
            "opponent_tag_ids": opponent_tag_id.index_select(0, active_indices)[:, step_index],
            "opponent_entity_ids": opponent_entity_id.index_select(0, active_indices)[:, step_index],
            "opponent_deck_ids": opponent_deck_id.index_select(0, active_indices)[:, step_index],
            "recurrent_state": active_recurrent,
        }
        active_loss = loss_mask[:, step_index].index_select(0, active_indices)
        if not torch.any(active_loss):
            with torch.no_grad():
                burn_output = model(**step_kwargs)
            if recurrent_state is not None and burn_output.recurrent_state is not None:
                recurrent_state = recurrent_state.clone()
                recurrent_state[active_indices] = burn_output.recurrent_state.detach().to(recurrent_state.dtype)
            if reference_model is not None:
                reference_kwargs = dict(step_kwargs)
                if reference_recurrent is not None:
                    reference_kwargs["recurrent_state"] = reference_recurrent.index_select(0, active_indices)
                with torch.no_grad():
                    reference_output = reference_model(**reference_kwargs)
                if reference_recurrent is not None and reference_output.recurrent_state is not None:
                    reference_recurrent = reference_recurrent.clone()
                    reference_recurrent[active_indices] = reference_output.recurrent_state.detach().to(reference_recurrent.dtype)
            continue
        output = model(**step_kwargs)
        if recurrent_state is not None and output.recurrent_state is not None:
            recurrent_state = recurrent_state.clone()
            recurrent_state[active_indices] = output.recurrent_state.to(recurrent_state.dtype)
        local_indices = torch.nonzero(active_loss, as_tuple=False).squeeze(-1)
        loss_indices = active_indices[local_indices]
        current_option_mask = step_kwargs["option_mask"][local_indices]
        step_log_probs = masked_log_softmax(output.policy_logits[local_indices], current_option_mask)
        step_action_index = action_index.index_select(0, loss_indices)[:, step_index].unsqueeze(-1)
        selected_logprob = torch.gather(step_log_probs, 1, step_action_index).squeeze(-1)
        step_old_logprob = old_logprob.index_select(0, loss_indices)[:, step_index]
        step_returns = return_target.index_select(0, loss_indices)[:, step_index]
        step_advantages = advantage.index_select(0, loss_indices)[:, step_index]
        step_weights = sample_weight.index_select(0, loss_indices)[:, step_index]
        step_search_acted = search_acted.index_select(0, loss_indices)[:, step_index] > 0.0
        current_high_mask = high_level_mask.index_select(0, loss_indices)[:, step_index]
        step_high_log_probs = masked_log_softmax(output.high_policy_logits[local_indices], current_high_mask)
        step_high_codes = chosen_high_level_code.index_select(0, loss_indices)[:, step_index]
        valid_high = step_high_codes >= 0
        if torch.any(valid_high):
            selected_high_logprob = torch.gather(
                step_high_log_probs[valid_high],
                1,
                step_high_codes[valid_high].unsqueeze(-1),
            ).squeeze(-1)
            high_policy_loss = -_weighted_mean(
                selected_high_logprob,
                step_weights[valid_high],
            )
        else:
            high_policy_loss = selected_logprob.new_zeros(())
        step_advantages = (step_advantages - step_advantages.mean()) / step_advantages.std(unbiased=False).clamp_min(1e-6)
        if objective == "ppo":
            ppo_active = ~step_search_acted
            if torch.any(ppo_active):
                ratio = torch.exp(selected_logprob[ppo_active] - step_old_logprob[ppo_active])
                clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
                policy_loss = -_weighted_mean(
                    torch.min(
                        ratio * step_advantages[ppo_active],
                        clipped_ratio * step_advantages[ppo_active],
                    ),
                    step_weights[ppo_active],
                )
                approx_kl = 0.5 * _weighted_mean(
                    (selected_logprob[ppo_active] - step_old_logprob[ppo_active]) ** 2,
                    step_weights[ppo_active],
                )
            else:
                policy_loss = selected_logprob.new_zeros(())
                approx_kl = selected_logprob.new_zeros(())
        else:
            policy_loss = -_weighted_mean(selected_logprob, step_weights) if objective == "behavior_clone" else selected_logprob.new_zeros(())
            approx_kl = selected_logprob.new_zeros(())

        reference_kl_loss = selected_logprob.new_zeros(())
        if reference_model is not None and objective == "ppo":
            reference_kwargs = dict(step_kwargs)
            if reference_recurrent is not None:
                reference_kwargs["recurrent_state"] = reference_recurrent.index_select(0, active_indices)
            with torch.no_grad():
                reference_output = reference_model(**reference_kwargs)
                reference_log_probs = masked_log_softmax(
                    reference_output.policy_logits[local_indices],
                    current_option_mask,
                )
            if reference_recurrent is not None and reference_output.recurrent_state is not None:
                reference_recurrent = reference_recurrent.clone()
                reference_recurrent[active_indices] = reference_output.recurrent_state.detach().to(reference_recurrent.dtype)
            ppo_active = ~step_search_acted
            if torch.any(ppo_active):
                current_probs = torch.exp(step_log_probs[ppo_active])
                reference_kl_loss = _weighted_mean(
                    (current_probs * (step_log_probs[ppo_active] - reference_log_probs[ppo_active])).sum(dim=-1),
                    step_weights[ppo_active],
                )

        current_values = output.partial_value[local_indices]
        value_loss = _weighted_mean((current_values - step_returns) ** 2, step_weights)
        oracle_values = model.oracle_value(privileged_state.index_select(0, loss_indices)[:, step_index])
        oracle_loss = _weighted_mean((oracle_values - step_returns) ** 2, step_weights)
        oracle_distill_loss = _weighted_mean((current_values - oracle_values.detach()) ** 2, step_weights)
        current_belief_hist = torch.sigmoid(output.belief_histogram[local_indices])
        current_belief_deck = torch.sigmoid(output.belief_remaining_deck_histogram[local_indices])
        target_hist = belief_histogram_target.index_select(0, loss_indices)[:, step_index]
        target_hand = belief_hand_size_target.index_select(0, loss_indices)[:, step_index]
        target_deck = belief_remaining_deck_histogram_target.index_select(0, loss_indices)[:, step_index]
        target_burst = belief_burst_ready_count_target.index_select(0, loss_indices)[:, step_index]
        target_group_hand = belief_group_hand_target.index_select(0, loss_indices)[:, step_index]
        target_group_deck = belief_group_deck_target.index_select(0, loss_indices)[:, step_index]
        current_search_self_range = search_self_range_summary.index_select(0, loss_indices)[:, step_index]
        current_search_opponent_range = search_opponent_range_summary.index_select(0, loss_indices)[:, step_index]
        current_search_context = search_context_summary.index_select(0, loss_indices)[:, step_index]
        target_search_policy = search_teacher_policy_target.index_select(0, loss_indices)[:, step_index]
        target_search_value = search_teacher_value_target.index_select(0, loss_indices)[:, step_index]
        target_search_weight = search_teacher_weight.index_select(0, loss_indices)[:, step_index]
        belief_hist_loss = _weighted_mean(((current_belief_hist - target_hist) ** 2).mean(dim=-1), step_weights)
        belief_hand_loss = _weighted_mean(
            F.smooth_l1_loss(output.belief_hand_size[local_indices], target_hand, reduction="none"),
            step_weights,
        )
        belief_deck_loss = _weighted_mean(((current_belief_deck - target_deck) ** 2).mean(dim=-1), step_weights)
        belief_burst_loss = _weighted_mean(
            F.smooth_l1_loss(output.belief_burst_ready_count[local_indices], target_burst, reduction="none"),
            step_weights,
        )
        belief_group_hand_loss = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output.belief_group_hand_logits[local_indices],
                target_group_hand,
                reduction="none",
            ).mean(dim=-1),
            step_weights,
        )
        belief_group_deck_loss = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output.belief_group_deck_logits[local_indices],
                target_group_deck,
                reduction="none",
            ).mean(dim=-1),
            step_weights,
        )
        belief_loss = (
            belief_hist_loss
            + belief_hand_loss
            + belief_deck_loss
            + belief_burst_loss
            + belief_group_hand_loss
            + belief_group_deck_loss
        )
        search_distill_loss = selected_logprob.new_zeros(())
        search_policy_loss = selected_logprob.new_zeros(())
        search_value_loss = selected_logprob.new_zeros(())
        if objective == "ppo":
            active_search = target_search_weight > 0.0
            if torch.any(active_search):
                search_weights = step_weights[active_search] * target_search_weight[active_search]
                search_distill_loss = _weighted_mean(
                    (
                        -target_search_policy[active_search, : current_option_mask.shape[1]]
                        * step_log_probs[active_search]
                    ).sum(dim=-1),
                    search_weights,
                )
                search_option_logits = model.search_option_logits(
                    option_features=step_kwargs["option_features"][local_indices][active_search],
                    self_range_summary=current_search_self_range[active_search],
                    opponent_range_summary=current_search_opponent_range[active_search],
                    search_context_summary=current_search_context[active_search],
                )
                search_log_probs = masked_log_softmax(
                    search_option_logits,
                    current_option_mask[active_search],
                )
                search_policy_loss = _weighted_mean(
                    (
                        -target_search_policy[active_search, : current_option_mask.shape[1]]
                        * search_log_probs
                    ).sum(dim=-1),
                    search_weights,
                )
                search_state_values = model.search_state_value(
                    self_range_summary=current_search_self_range[active_search],
                    opponent_range_summary=current_search_opponent_range[active_search],
                    search_context_summary=current_search_context[active_search],
                )
                search_value_loss = _weighted_mean(
                    (search_state_values - target_search_value[active_search]) ** 2,
                    search_weights,
                )
                target_search_high_policy = search_teacher_high_policy_target.index_select(0, loss_indices)[:, step_index]
                high_policy_loss = high_policy_loss + _weighted_mean(
                    (
                        -target_search_high_policy[active_search]
                        * step_high_log_probs[active_search]
                    ).sum(dim=-1),
                    search_weights,
                )
        entropy = _weighted_mean((-(torch.exp(step_log_probs) * step_log_probs).sum(dim=-1)), step_weights)
        step_total = (
            (policy_coef * policy_loss)
            + high_policy_loss
            + (value_coef * value_loss)
            + (belief_coef * belief_loss)
            + search_distill_loss
            + search_policy_loss
            + search_value_loss
            + (oracle_coef * (oracle_loss + oracle_distill_loss))
            + (reference_kl_coef * reference_kl_loss)
            - (entropy_coef * entropy if objective == "ppo" else 0.0)
        )
        total_loss = total_loss + step_total
        metrics["policy_loss"] += float(policy_loss.item())
        metrics["high_policy_loss"] += float(high_policy_loss.item())
        metrics["reference_kl_loss"] += float(reference_kl_loss.item())
        metrics["value_loss"] += float(value_loss.item())
        metrics["entropy"] += float(entropy.item())
        metrics["belief_loss"] += float(belief_loss.item())
        metrics["belief_hand_loss"] += float(belief_hand_loss.item())
        metrics["belief_deck_loss"] += float(belief_deck_loss.item())
        metrics["belief_burst_loss"] += float(belief_burst_loss.item())
        metrics["belief_group_hand_loss"] += float(belief_group_hand_loss.item())
        metrics["belief_group_deck_loss"] += float(belief_group_deck_loss.item())
        metrics["search_distill_loss"] += float(search_distill_loss.item())
        metrics["search_policy_loss"] += float(search_policy_loss.item())
        metrics["search_value_loss"] += float(search_value_loss.item())
        metrics["oracle_loss"] += float(oracle_loss.item())
        metrics["oracle_distill_loss"] += float(oracle_distill_loss.item())
        metrics["approx_kl"] += float(approx_kl.item())
        loss_steps += 1
    if loss_steps <= 0:
        raise RuntimeError("no valid loss steps found in chunk batch")
    for key in metrics:
        metrics[key] /= loss_steps
    metrics["total_loss"] = float(total_loss.item()) / loss_steps
    return total_loss / loss_steps, metrics, loss_steps


def _slice_chunk_batch(batch: dict[str, torch.Tensor], start: int, stop: int) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in batch.items()}


def _resolve_micro_batch_size(
    *,
    batch_size: int,
    requested_micro_batch_size: int | None,
    param_count: int,
    device: torch.device,
) -> int:
    if batch_size <= 0:
        return 1
    if requested_micro_batch_size is not None:
        return max(1, min(batch_size, int(requested_micro_batch_size)))
    if device.type != "cuda":
        return batch_size
    if param_count >= 30_000_000:
        return min(batch_size, 8)
    if param_count >= 20_000_000:
        return min(batch_size, 12)
    if param_count >= 10_000_000:
        return min(batch_size, 16)
    return batch_size


def _load_reference_model(
    checkpoint_path: str | Path | None,
    *,
    device: str,
) -> PpoTransformerPolicy | None:
    if checkpoint_path is None:
        return None
    reference_path = Path(checkpoint_path)
    if not reference_path.exists():
        return None
    try:
        model, _, _, _ = load_ppo_training_checkpoint(reference_path, device=device)
    except Exception:
        model, _, _ = load_ppo_deployment_checkpoint(reference_path, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _collate_chunks(chunks: Sequence[PpoTrainChunk]) -> dict[str, torch.Tensor]:
    batch_size = len(chunks)
    max_seq_len = max(len(chunk.steps) for chunk in chunks)
    first = chunks[0].steps[0]
    max_tokens = max(len(step.token_features) for chunk in chunks for step in chunk.steps)
    max_options = max(len(step.option_features) for chunk in chunks for step in chunk.steps)
    token_dim = len(first.token_features[0])
    option_dim = len(first.option_features[0])
    privileged_dim = len(first.privileged_state)
    belief_dim = len(first.belief_histogram_target)
    belief_deck_dim = len(first.belief_remaining_deck_histogram_target)
    belief_group_dim = len(first.belief_group_hand_target)
    search_range_dim = len(first.search_self_range_summary)
    search_context_dim = len(first.search_context_summary)
    high_action_dim = len(first.high_level_mask)

    token_features = torch.zeros((batch_size, max_seq_len, max_tokens, token_dim), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_seq_len, max_tokens), dtype=torch.bool)
    opponent_token_mask = torch.zeros((batch_size, max_seq_len, max_tokens), dtype=torch.bool)
    option_features = torch.zeros((batch_size, max_seq_len, max_options, option_dim), dtype=torch.float32)
    option_mask = torch.zeros((batch_size, max_seq_len, max_options), dtype=torch.bool)
    privileged_state = torch.zeros((batch_size, max_seq_len, privileged_dim), dtype=torch.float32)
    belief_histogram_target = torch.zeros((batch_size, max_seq_len, belief_dim), dtype=torch.float32)
    belief_hand_size_target = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    belief_remaining_deck_histogram_target = torch.zeros((batch_size, max_seq_len, belief_deck_dim), dtype=torch.float32)
    belief_burst_ready_count_target = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    belief_group_hand_target = torch.zeros((batch_size, max_seq_len, belief_group_dim), dtype=torch.float32)
    belief_group_deck_target = torch.zeros((batch_size, max_seq_len, belief_group_dim), dtype=torch.float32)
    search_self_range_summary = torch.zeros((batch_size, max_seq_len, search_range_dim), dtype=torch.float32)
    search_opponent_range_summary = torch.zeros((batch_size, max_seq_len, search_range_dim), dtype=torch.float32)
    search_context_summary = torch.zeros((batch_size, max_seq_len, search_context_dim), dtype=torch.float32)
    search_teacher_policy_target = torch.zeros((batch_size, max_seq_len, max_options), dtype=torch.float32)
    search_teacher_high_policy_target = torch.zeros((batch_size, max_seq_len, high_action_dim), dtype=torch.float32)
    search_teacher_value_target = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    search_teacher_weight = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    search_acted = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    high_level_mask = torch.zeros((batch_size, max_seq_len, high_action_dim), dtype=torch.bool)
    opponent_tag_id = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    opponent_entity_id = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    opponent_deck_id = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    action_index = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    chosen_high_level_code = torch.full((batch_size, max_seq_len), -1, dtype=torch.long)
    old_logprob = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    return_target = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    advantage = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
    sample_weight = torch.ones((batch_size, max_seq_len), dtype=torch.float32)
    step_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for batch_index, chunk in enumerate(chunks):
        for step_index, step in enumerate(chunk.steps):
            token_count = len(step.token_features)
            option_count = len(step.option_features)
            token_features[batch_index, step_index, :token_count] = torch.as_tensor(step.token_features, dtype=torch.float32)
            token_mask[batch_index, step_index, :token_count] = True
            opponent_token_mask[batch_index, step_index, :token_count] = torch.as_tensor(
                step.opponent_token_mask,
                dtype=torch.bool,
            )
            option_features[batch_index, step_index, :option_count] = torch.as_tensor(
                step.option_features,
                dtype=torch.float32,
            )
            option_mask[batch_index, step_index, :option_count] = True
            privileged_state[batch_index, step_index] = torch.as_tensor(step.privileged_state, dtype=torch.float32)
            belief_histogram_target[batch_index, step_index] = torch.as_tensor(
                step.belief_histogram_target,
                dtype=torch.float32,
            )
            belief_hand_size_target[batch_index, step_index] = float(step.belief_hand_size_target)
            belief_remaining_deck_histogram_target[batch_index, step_index] = torch.as_tensor(
                step.belief_remaining_deck_histogram_target,
                dtype=torch.float32,
            )
            belief_burst_ready_count_target[batch_index, step_index] = float(step.belief_burst_ready_count_target)
            belief_group_hand_target[batch_index, step_index] = torch.as_tensor(
                step.belief_group_hand_target,
                dtype=torch.float32,
            )
            belief_group_deck_target[batch_index, step_index] = torch.as_tensor(
                step.belief_group_deck_target,
                dtype=torch.float32,
            )
            search_self_range_summary[batch_index, step_index] = torch.as_tensor(
                step.search_self_range_summary,
                dtype=torch.float32,
            )
            search_opponent_range_summary[batch_index, step_index] = torch.as_tensor(
                step.search_opponent_range_summary,
                dtype=torch.float32,
            )
            search_context_summary[batch_index, step_index] = torch.as_tensor(
                step.search_context_summary,
                dtype=torch.float32,
            )
            if step.search_teacher_policy_target:
                search_teacher_policy_target[batch_index, step_index, : len(step.search_teacher_policy_target)] = torch.as_tensor(
                    step.search_teacher_policy_target,
                    dtype=torch.float32,
                )
            if step.search_teacher_high_policy_target:
                search_teacher_high_policy_target[batch_index, step_index, : len(step.search_teacher_high_policy_target)] = torch.as_tensor(
                    step.search_teacher_high_policy_target,
                    dtype=torch.float32,
                )
            search_teacher_value_target[batch_index, step_index] = float(step.search_teacher_value_target)
            search_teacher_weight[batch_index, step_index] = float(step.search_teacher_weight)
            search_acted[batch_index, step_index] = float(step.search_acted)
            if step.high_level_mask:
                high_level_mask[batch_index, step_index, : len(step.high_level_mask)] = torch.as_tensor(
                    step.high_level_mask,
                    dtype=torch.bool,
                )
            opponent_tag_id[batch_index, step_index] = int(step.opponent_tag_id)
            opponent_entity_id[batch_index, step_index] = int(step.opponent_entity_id)
            opponent_deck_id[batch_index, step_index] = int(step.opponent_deck_id)
            action_index[batch_index, step_index] = int(step.action_index)
            chosen_high_level_code[batch_index, step_index] = int(step.chosen_high_level_code)
            old_logprob[batch_index, step_index] = float(step.old_logprob)
            return_target[batch_index, step_index] = float(step.return_target)
            advantage[batch_index, step_index] = float(step.advantage)
            sample_weight[batch_index, step_index] = float(step.sample_weight)
            step_mask[batch_index, step_index] = True
        loss_start = min(chunk.burn_in, len(chunk.steps))
        loss_end = min(len(chunk.steps), loss_start + chunk.loss_length)
        if loss_end > loss_start:
            loss_mask[batch_index, loss_start:loss_end] = True

    return {
        "token_features": token_features,
        "token_mask": token_mask,
        "opponent_token_mask": opponent_token_mask,
        "option_features": option_features,
        "option_mask": option_mask,
        "privileged_state": privileged_state,
        "belief_histogram_target": belief_histogram_target,
        "belief_hand_size_target": belief_hand_size_target,
        "belief_remaining_deck_histogram_target": belief_remaining_deck_histogram_target,
        "belief_burst_ready_count_target": belief_burst_ready_count_target,
        "belief_group_hand_target": belief_group_hand_target,
        "belief_group_deck_target": belief_group_deck_target,
        "search_self_range_summary": search_self_range_summary,
        "search_opponent_range_summary": search_opponent_range_summary,
        "search_context_summary": search_context_summary,
        "search_teacher_policy_target": search_teacher_policy_target,
        "search_teacher_high_policy_target": search_teacher_high_policy_target,
        "search_teacher_value_target": search_teacher_value_target,
        "search_teacher_weight": search_teacher_weight,
        "search_acted": search_acted,
        "high_level_mask": high_level_mask,
        "opponent_tag_id": opponent_tag_id,
        "opponent_entity_id": opponent_entity_id,
        "opponent_deck_id": opponent_deck_id,
        "action_index": action_index,
        "chosen_high_level_code": chosen_high_level_code,
        "old_logprob": old_logprob,
        "return_target": return_target,
        "advantage": advantage,
        "sample_weight": sample_weight,
        "step_mask": step_mask,
        "loss_mask": loss_mask,
    }


def _dataset_ratios(
    *,
    fresh_chunks: Sequence[PpoTrainChunk],
    replay_chunks: Sequence[PpoTrainChunk],
) -> dict[str, float]:
    all_chunks = list(fresh_chunks) + list(replay_chunks)
    if not all_chunks:
        return {
            "late_round_sample_ratio": 0.0,
            "draw_chunk_ratio": 0.0,
            "loss_chunk_ratio": 0.0,
        }
    total = float(len(all_chunks))
    return {
        "late_round_sample_ratio": sum(1 for chunk in all_chunks if chunk.has_late_round) / total,
        "draw_chunk_ratio": sum(1 for chunk in all_chunks if chunk.has_terminal_draw) / total,
        "loss_chunk_ratio": sum(1 for chunk in all_chunks if chunk.has_terminal_loss) / total,
    }


def _context_from_step(step: TrajectoryStep, *, episode_metadata: dict[str, object] | None = None) -> DecisionContext:
    metadata = dict(episode_metadata or {})
    metadata.update(step.metadata)
    return DecisionContext(
        acting_player=step.acting_player,
        request_type=step.request_type,
        step_index=0,
        full_state=step.pre_state,
        legal_low_level_codes=step.legal_low_level_codes,
        legal_low_level_specs=step.legal_low_level_specs,
        legal_low_level_mask=step.legal_low_level_mask,
        legal_high_level_codes=step.legal_high_level_codes,
        high_to_low_map=step.high_to_low_map,
        player_view=step.player_view,
        full_state_json=step.full_state_json_before,
        terminal=False,
        metadata=metadata,
    )


def _selected_action_index(step: TrajectoryStep) -> int:
    selected_code = selected_low_level_code(step)
    if selected_code is None:
        raise KeyError("selected action code could not be derived from trajectory step")
    legal_codes = (
        tuple(int(code) for code in step.legal_low_level_codes)
        if step.legal_low_level_codes
        else tuple(int(spec.action_code) for spec in legal_low_level_specs(_context_from_step(step)))
    )
    for index, action_code in enumerate(legal_codes):
        if int(action_code) == selected_code:
            return index
    raise KeyError(f"selected action_code {selected_code} not found in legal low-level codes")


def _gae_returns(
    transitions: Sequence[_DecisionTransition],
    *,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    returns = [0.0] * len(transitions)
    advantages = [0.0] * len(transitions)
    gae = 0.0
    for index in reversed(range(len(transitions))):
        transition = transitions[index]
        delta = transition.reward + (transition.discount * transition.next_value) - transition.value
        gae = delta + (transition.discount * gae_lambda * gae)
        advantages[index] = gae
        returns[index] = gae + transition.value
    return returns, advantages


def _accumulate_reward_until_next_trainable(
    steps: Sequence[TrajectoryStep],
    *,
    start_index: int,
    next_trainable_index: int | None,
    perspective_player: int,
    gamma: float,
) -> tuple[float, float]:
    reward = 0.0
    discount = 1.0
    for index in range(start_index, len(steps)):
        step = steps[index]
        reward += discount * _shaped_reward(
            step,
            perspective_player=perspective_player,
            gamma=gamma,
        )
        if step.done:
            return reward, 0.0
        discount *= gamma
        if next_trainable_index is not None and (index + 1) >= next_trainable_index:
            return reward, discount
    return reward, 0.0


def _shaped_reward(
    step: TrajectoryStep,
    *,
    perspective_player: int,
    gamma: float,
) -> float:
    base_reward = _terminal_reward(step, perspective_player=perspective_player)
    phi_before = _potential(step.pre_state, perspective_player=perspective_player)
    phi_after = _potential(step.post_state, perspective_player=perspective_player)
    shaping = (gamma * phi_after) - phi_before
    return (
        base_reward
        + shaping
        + _anti_draw_penalty(step.post_state.round_number)
        + _action_efficiency_reward(step, perspective_player=perspective_player)
    )


def _terminal_reward(step: TrajectoryStep, *, perspective_player: int) -> float:
    if not step.done:
        return 0.0
    winner = step.post_state.winner
    if winner == perspective_player:
        return 1.0
    return -1.0


def _potential(state, *, perspective_player: int) -> float:
    self_player = state.players[perspective_player]
    opponent = state.players[1 - perspective_player]
    self_hp = sum(character.health for character in self_player.characters)
    opp_hp = sum(character.health for character in opponent.characters)
    self_alive = sum(0 if character.defeated else 1 for character in self_player.characters)
    opp_alive = sum(0 if character.defeated else 1 for character in opponent.characters)
    self_energy = sum(character.energy for character in self_player.characters)
    opp_energy = sum(character.energy for character in opponent.characters)
    self_board = len(self_player.combat_statuses) + len(self_player.summons) + len(self_player.supports)
    opp_board = len(opponent.combat_statuses) + len(opponent.summons) + len(opponent.supports)
    return (
        0.6 * ((self_hp - opp_hp) / 30.0)
        + 0.2 * ((self_alive - opp_alive) / 3.0)
        + 0.1 * ((self_energy - opp_energy) / 6.0)
        + 0.1 * ((self_board - opp_board) / 10.0)
    )


def _anti_draw_penalty(round_number: int) -> float:
    if round_number >= 15:
        return -0.10
    if round_number == 14:
        return -0.05
    if round_number == 13:
        return -0.02
    return 0.0


def _action_efficiency_reward(step: TrajectoryStep, *, perspective_player: int) -> float:
    severity = premature_end_severity(step)
    if severity <= 0.0:
        return 0.0
    waste_ratio = declare_end_waste_ratio(step)
    advantage = max(0.0, min(1.0, _potential(step.pre_state, perspective_player=perspective_player)))
    # When the current player is already clearly ahead, ending the turn can be a
    # valid simplification. Keep the prior mild and attenuate it in winning states.
    attenuation = max(0.2, 1.0 - (0.8 * advantage))
    return severity * attenuation * (-0.03 - (0.03 * waste_ratio))


def _sample_weight(
    *,
    episode: EpisodeRecord,
    perspective_player: int,
    round_number: int,
) -> float:
    weight = 1.0
    if round_number >= 13:
        weight += 1.0
    if episode.winner is None or episode.winner != perspective_player:
        weight += 1.0
    return min(3.0, weight)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1e-6)
    return torch.sum(values * weights) / denom
