from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .action_hierarchy import (
    low_level_kind_for_code,
    match_low_level_code_index,
    public_spent_dice_for_code,
    semantic_action_key_for_code,
)
from .decks import deck_by_name
from .env import GitcgDecisionEnv
from .ppo_features import TokenObservationEncoder
from .ppo_model import PpoTransformerPolicy, masked_log_softmax
from .public_state import (
    ExplicitRange,
    ExplicitRangeHypothesis,
    PublicActionRecord,
    PublicBeliefState,
    PublicStateTracker,
    mask_state_for_player,
)
from .schema import (
    ActionChoice,
    DecisionContext,
    DecisionType,
    EnvConfig,
    Matchup,
    OptionKind,
    PlayerSnapshot,
    StateSnapshot,
    TrajectoryStep,
)

_DICE_TYPES = (1, 2, 3, 4, 5, 6, 7, 8)


@dataclass(frozen=True)
class SePotPhaseBudget:
    depth: int
    root_top_k: int
    belief_samples: int
    timeout_ms: int


@dataclass(frozen=True)
class SePotSearchConfig:
    enabled: bool = True
    uncertainty_penalty: float = 0.15
    # Runtime SePoT should stay in the low-hundreds of milliseconds. Larger
    # budgets make fallback latency dominate the whole acting pipeline.
    opening_timeout_ms: int = 120
    midgame_timeout_ms: int = 180
    endgame_timeout_ms: int = 250
    template_seed: int = 11
    max_template_steps: int = 96


@dataclass(frozen=True)
class SearchDecision:
    triggered: bool
    fallback_reason: str | None
    selected_action_index: int
    transformed_policy: tuple[float, ...]
    root_value: float
    root_candidate_count: int
    belief_sample_count: int
    phase_depth: int


@dataclass(frozen=True)
class _RangeUpdateResult:
    updated_public_belief: PublicBeliefState
    sampled_self_hypothesis: ExplicitRangeHypothesis
    sampled_opponent_hypothesis: ExplicitRangeHypothesis


class _SearchTimeout(RuntimeError):
    pass


def phase_budget_for_context(context: DecisionContext, *, config: SePotSearchConfig) -> SePotPhaseBudget:
    round_number = int(context.player_view.round_number) if context.player_view is not None else 0
    if round_number <= 3:
        return SePotPhaseBudget(depth=1, root_top_k=3, belief_samples=2, timeout_ms=config.opening_timeout_ms)
    if round_number <= 8:
        return SePotPhaseBudget(depth=2, root_top_k=5, belief_samples=4, timeout_ms=config.midgame_timeout_ms)
    return SePotPhaseBudget(depth=3, root_top_k=8, belief_samples=6, timeout_ms=config.endgame_timeout_ms)


def should_trigger_search(
    context: DecisionContext,
    *,
    config: SePotSearchConfig,
) -> bool:
    if not config.enabled:
        return False
    if context.request_type != DecisionType.ACTION:
        return False
    if context.player_view is None:
        return False
    legal_count = len(context.legal_low_level_codes)
    round_number = int(context.player_view.round_number)
    has_end = any(
        low_level_kind_for_code(int(action_code)) == OptionKind.ACTION_DECLARE_END
        for action_code in context.legal_low_level_codes
    )
    productive_non_end = any(
        low_level_kind_for_code(int(action_code)) in {
            OptionKind.ACTION_USE_SKILL,
            OptionKind.ACTION_PLAY_CARD,
            OptionKind.ACTION_ELEMENTAL_TUNING,
        }
        for action_code in context.legal_low_level_codes
    )
    if round_number <= 3:
        return legal_count >= 6 or (has_end and productive_non_end)
    return round_number >= 4 and legal_count >= 2


def build_public_belief_state(
    *,
    context: DecisionContext,
    tracker: PublicStateTracker | None,
    card_vocabulary: Sequence[int],
    opponent_deck_name: str,
    belief_histogram: Sequence[float],
    belief_remaining_deck_histogram: Sequence[float],
    belief_samples: int,
    root_player: int | None = None,
) -> PublicBeliefState:
    visible_state = context.player_view
    if visible_state is None:
        raise ValueError("player_view is required to build PublicBeliefState")
    acting_player = int(context.acting_player)
    root_player = acting_player if root_player is None else int(root_player)
    self_range = _build_self_range_from_state(visible_state, player_id=root_player)
    opponent_range = _build_opponent_range(
        state=visible_state,
        root_player=root_player,
        tracker=tracker,
        card_vocabulary=card_vocabulary,
        opponent_deck_name=opponent_deck_name,
        belief_histogram=belief_histogram,
        belief_remaining_deck_histogram=belief_remaining_deck_histogram,
        belief_samples=belief_samples,
    )
    history = tracker.public_history() if tracker is not None else ()
    opponent_public_dice_count = (
        tracker.opponent_public_dice_count(context)
        if tracker is not None
        else float(len(visible_state.players[1 - root_player].dice))
    )
    opponent_public_hand_count = (
        tracker.opponent_public_hand_count(context)
        if tracker is not None
        else float(len(visible_state.players[1 - root_player].hand_cards))
    )
    revealed = tracker.revealed_opponent_card_definition_ids() if tracker is not None else ()
    return PublicBeliefState(
        root_player=root_player,
        acting_player=acting_player,
        public_state=visible_state,
        round_number=int(visible_state.round_number),
        phase=str(visible_state.phase),
        opponent_public_dice_count=float(opponent_public_dice_count),
        opponent_public_hand_count=float(opponent_public_hand_count),
        self_range=self_range,
        opponent_range=opponent_range,
        public_history=history,
        revealed_opponent_card_definition_ids=tuple(int(value) for value in revealed),
    )


def range_summary(explicit_range: ExplicitRange, *, card_vocabulary: Sequence[int]) -> tuple[float, ...]:
    card_index = {int(value): index for index, value in enumerate(card_vocabulary)}
    hand_hist = [0.0] * len(card_index)
    deck_hist = [0.0] * len(card_index)
    dice_hist = [0.0] * len(_DICE_TYPES)
    total_weight = max(1e-6, sum(float(hyp.weight) for hyp in explicit_range.hypotheses))
    average_hand_count = 0.0
    average_deck_count = 0.0
    for hypothesis in explicit_range.hypotheses:
        weight = float(hypothesis.weight) / total_weight
        average_hand_count += weight * float(len(hypothesis.hand_definition_ids))
        average_deck_count += weight * float(len(hypothesis.deck_definition_ids))
        for definition_id in hypothesis.hand_definition_ids:
            index = card_index.get(int(definition_id))
            if index is not None:
                hand_hist[index] += weight
        for definition_id in hypothesis.deck_definition_ids:
            index = card_index.get(int(definition_id))
            if index is not None:
                deck_hist[index] += weight
        for die in hypothesis.hidden_dice:
            if int(die) in _DICE_TYPES:
                dice_hist[_DICE_TYPES.index(int(die))] += weight / max(1, len(hypothesis.hidden_dice))
    return tuple(
        hand_hist
        + deck_hist
        + dice_hist
        + [
            min(average_hand_count, 10.0) / 10.0,
            min(average_deck_count, 30.0) / 30.0,
            min(float(explicit_range.count), 8.0) / 8.0,
            1.0,
        ]
    )


def step_search_context_summary(context: DecisionContext) -> tuple[float, ...]:
    visible_state = context.player_view
    round_number = float(getattr(visible_state, "round_number", 0))
    phase = str(getattr(visible_state, "phase", ""))
    legal_count = float(len(context.legal_low_level_codes))
    has_end = any(
        low_level_kind_for_code(int(action_code)) == OptionKind.ACTION_DECLARE_END
        for action_code in context.legal_low_level_codes
    )
    productive = sum(
        1
        for action_code in context.legal_low_level_codes
        if low_level_kind_for_code(int(action_code)) in {
            OptionKind.ACTION_USE_SKILL,
            OptionKind.ACTION_PLAY_CARD,
            OptionKind.ACTION_ELEMENTAL_TUNING,
        }
    )
    return (
        min(round_number, 20.0) / 20.0,
        min(legal_count, 16.0) / 16.0,
        1.0 if phase == "action" else 0.0,
        1.0 if has_end else 0.0,
        min(float(productive), 8.0) / 8.0,
        1.0 if round_number >= 9.0 else 0.0,
        1.0 if round_number >= 4.0 else 0.0,
        1.0 if int(context.acting_player) == 0 else 0.0,
    )


def state_search_context_summary(
    *,
    context: DecisionContext,
    budget: SePotPhaseBudget,
) -> tuple[float, ...]:
    return step_search_context_summary(context) + (
        min(float(budget.depth), 4.0) / 4.0,
        min(float(budget.root_top_k), 8.0) / 8.0,
        min(float(budget.belief_samples), 8.0) / 8.0,
        min(float(budget.timeout_ms), 500.0) / 500.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )


class _RawStoreBuilder:
    def __init__(self, store: list[Any]) -> None:
        self.store = store
        self._definition_cache: dict[tuple[str, int], int] = {}
        self._next_synthetic_id = -900000

    def ref(self, node: Any) -> dict[str, int]:
        self.store.append(node)
        return {"$": len(self.store) - 1}

    def definition_ref(self, *, kind: str, definition_id: int) -> dict[str, int]:
        key = (str(kind), int(definition_id))
        index = self._definition_cache.get(key)
        if index is None:
            self.store.append({"$$": str(kind), "id": int(definition_id)})
            index = len(self.store) - 1
            self._definition_cache[key] = index
        return {"$": index}

    def synthetic_id(self) -> int:
        value = self._next_synthetic_id
        self._next_synthetic_id -= 1
        return value


class PublicStateReconstructor:
    def __init__(self, *, env_config: EnvConfig, matchup: Matchup, config: SePotSearchConfig) -> None:
        self.env_config = env_config
        self.matchup = matchup
        self.config = config
        self._template_cache: dict[int, str] = {}

    def reconstruct(
        self,
        *,
        public_belief: PublicBeliefState,
        self_hypothesis: ExplicitRangeHypothesis,
        opponent_hypothesis: ExplicitRangeHypothesis,
    ) -> tuple[GitcgDecisionEnv | None, DecisionContext | None]:
        template_json = self._action_template_json(acting_player=public_belief.acting_player)
        if template_json is None:
            return None, None
        try:
            payload = self._patch_template_payload(
                template_json=template_json,
                public_belief=public_belief,
                self_hypothesis=self_hypothesis,
                opponent_hypothesis=opponent_hypothesis,
            )
            state_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            env = GitcgDecisionEnv(
                self.env_config,
                self.matchup,
                enable_result_based_action_relabel=False,
            )
            context = env.reset(seed=self.config.template_seed, state_json=state_json)
            return env, context
        except Exception:
            return None, None

    def ensure_template(self, *, acting_player: int) -> str | None:
        return self._action_template_json(acting_player=acting_player)

    def _action_template_json(self, *, acting_player: int) -> str | None:
        cache_key = int(acting_player)
        if cache_key in self._template_cache:
            return self._template_cache[cache_key]
        env = GitcgDecisionEnv(
            self.env_config,
            self.matchup,
            enable_result_based_action_relabel=False,
        )
        try:
            context = env.reset(seed=self.config.template_seed)
            steps = 0
            while steps < self.config.max_template_steps:
                if context.request_type == DecisionType.ACTION and int(context.acting_player) == int(acting_player):
                    if context.full_state_json:
                        # This template comes from a synthetic seed-run of the matchup and is
                        # used only as a structural scaffold for public-state reconstruction.
                        # It is not the live runtime root state's hidden truth.
                        self._template_cache[cache_key] = context.full_state_json
                        return context.full_state_json
                    return None
                if not context.legal_low_level_codes:
                    return None
                next_context, _, done = env.step(ActionChoice(action_code=context.legal_low_level_codes[0]))
                if done or next_context is None:
                    return None
                context = next_context
                steps += 1
            return None
        finally:
            env.close()

    def _patch_template_payload(
        self,
        *,
        template_json: str,
        public_belief: PublicBeliefState,
        self_hypothesis: ExplicitRangeHypothesis,
        opponent_hypothesis: ExplicitRangeHypothesis,
    ) -> dict[str, Any]:
        payload = json.loads(template_json)
        store = payload["store"]
        root = dict(store[-1])
        builder = _RawStoreBuilder(store)
        root_player = int(public_belief.root_player)
        opponent_player = 1 - root_player
        public_state = public_belief.public_state
        root_players_raw = store[root["players"]["$"]]
        self_template_player = store[root_players_raw[root_player]["$"]]
        opponent_template_player = store[root_players_raw[opponent_player]["$"]]
        player_nodes = [None, None]
        player_nodes[root_player] = self._build_player_node(
            builder=builder,
            player=public_state.players[root_player],
            template_player=self_template_player,
            hand_definition_ids=self_hypothesis.hand_definition_ids,
            deck_definition_ids=self_hypothesis.deck_definition_ids,
            hidden_dice=self_hypothesis.hidden_dice,
            is_public_player=True,
        )
        player_nodes[opponent_player] = self._build_player_node(
            builder=builder,
            player=public_state.players[opponent_player],
            template_player=opponent_template_player,
            hand_definition_ids=opponent_hypothesis.hand_definition_ids,
            deck_definition_ids=opponent_hypothesis.deck_definition_ids,
            hidden_dice=opponent_hypothesis.hidden_dice,
            is_public_player=False,
        )
        root["players"] = builder.ref([builder.ref(player_nodes[0]), builder.ref(player_nodes[1])])
        root["currentTurn"] = int(public_state.current_turn)
        root["roundNumber"] = int(public_state.round_number)
        root["phase"] = str(public_state.phase).replace("_", "")
        root["winner"] = None if public_state.winner is None else int(public_state.winner)
        store[-1] = root
        return payload

    def _build_player_node(
        self,
        *,
        builder: _RawStoreBuilder,
        player: PlayerSnapshot,
        template_player: dict[str, Any],
        hand_definition_ids: Sequence[int],
        deck_definition_ids: Sequence[int],
        hidden_dice: Sequence[int],
        is_public_player: bool,
    ) -> dict[str, Any]:
        character_refs = [
            builder.ref(self._build_character_node(builder=builder, character=character))
            for character in player.characters
        ]
        hand_refs = self._build_card_refs(
            builder=builder,
            public_cards=player.hand_cards if is_public_player else (),
            definition_ids=hand_definition_ids,
        )
        pile_refs = self._build_card_refs(
            builder=builder,
            public_cards=player.pile_cards if is_public_player else (),
            definition_ids=deck_definition_ids,
        )
        combat_status_refs = [
            builder.ref(self._build_entity_node(builder=builder, entity=entity))
            for entity in player.combat_statuses
        ]
        summon_refs = [
            builder.ref(self._build_entity_node(builder=builder, entity=entity))
            for entity in player.summons
        ]
        support_refs = [
            builder.ref(self._build_entity_node(builder=builder, entity=entity))
            for entity in player.supports
        ]
        return {
            "who": int(template_player.get("who", 0)),
            "activeCharacterId": int(player.active_character_id or 0),
            "characters": builder.ref(character_refs),
            "initialPile": template_player.get("initialPile", builder.ref([])),
            "pile": builder.ref(pile_refs),
            "hands": builder.ref(hand_refs),
            "dice": builder.ref([int(die) for die in hidden_dice]),
            "combatStatuses": combat_status_refs,
            "summons": summon_refs,
            "supports": support_refs,
            "declaredEnd": bool(player.declared_end),
            "canCharged": bool(player.can_charged),
            "canPlunging": bool(player.can_plunging),
            "hasDefeated": bool(player.has_defeated),
            "legendUsed": bool(player.legend_used),
            "skipNextTurn": bool(template_player.get("skipNextTurn", False)),
            "roundSkillLog": dict(template_player.get("roundSkillLog", {"__type": "map", "entries": []})),
            "removedEntities": list(template_player.get("removedEntities", [])),
        }

    def _build_card_refs(
        self,
        *,
        builder: _RawStoreBuilder,
        public_cards: Sequence[Any],
        definition_ids: Sequence[int],
    ) -> list[dict[str, int]]:
        refs: list[dict[str, int]] = []
        for index, definition_id in enumerate(definition_ids):
            card_id = int(public_cards[index].id) if index < len(public_cards) else builder.synthetic_id()
            refs.append(
                builder.ref(
                    {
                        "id": int(card_id),
                        "definition": builder.definition_ref(kind="entities", definition_id=int(definition_id)),
                        "variables": {},
                        "attachments": [],
                    }
                )
            )
        return refs

    def _build_character_node(self, *, builder: _RawStoreBuilder, character) -> dict[str, Any]:
        return {
            "id": int(character.id),
            "definition": builder.definition_ref(kind="characters", definition_id=int(character.definition_id)),
            "variables": {
                "health": int(character.health),
                "energy": int(character.energy),
                "alive": 0 if character.defeated else 1,
                "aura": int(character.aura),
                "maxHealth": int(character.max_health),
                "maxEnergy": int(character.max_energy),
            },
            "entities": [
                builder.ref(self._build_entity_node(builder=builder, entity=entity))
                for entity in character.entities
            ],
        }

    def _build_entity_node(self, *, builder: _RawStoreBuilder, entity) -> dict[str, Any]:
        return {
            "id": int(entity.id),
            "definition": builder.definition_ref(kind="entities", definition_id=int(entity.definition_id)),
            "variables": {str(key): int(value) for key, value in entity.variables.items()},
        }


class PublicBeliefUpdater:
    def __init__(
        self,
        *,
        reconstructor: PublicStateReconstructor,
        card_vocabulary: Sequence[int],
    ) -> None:
        self.reconstructor = reconstructor
        self.card_vocabulary = tuple(int(value) for value in card_vocabulary)
        self._advance_cache: dict[tuple[Any, ...], ExplicitRange] = {}

    def update_after_step(
        self,
        *,
        pre_public_belief: PublicBeliefState,
        selected_action_code: int,
        post_state: StateSnapshot,
        next_context: DecisionContext | None,
        tracker: PublicStateTracker,
        sampled_self_hypothesis: ExplicitRangeHypothesis,
        sampled_opponent_hypothesis: ExplicitRangeHypothesis,
        deadline: float,
    ) -> _RangeUpdateResult:
        if time.perf_counter() > deadline:
            raise _SearchTimeout()
        cache_key = (
            _public_belief_cache_key(pre_public_belief),
            semantic_action_key_for_code(selected_action_code),
            _hypothesis_cache_key(sampled_self_hypothesis),
        )
        updated_opponent_range = self._advance_cache.get(cache_key)
        if updated_opponent_range is None:
            updated_opponent_range = self._advance_opponent_range(
                pre_public_belief=pre_public_belief,
                selected_action_code=selected_action_code,
                sampled_self_hypothesis=sampled_self_hypothesis,
                sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                deadline=deadline,
            )
            if updated_opponent_range.count > 0:
                self._advance_cache[cache_key] = updated_opponent_range
                if len(self._advance_cache) > 256:
                    self._advance_cache.clear()
        public_state = mask_state_for_player(post_state, perspective_player=pre_public_belief.root_player)
        updated = PublicBeliefState(
            root_player=pre_public_belief.root_player,
            acting_player=(
                int(next_context.acting_player)
                if next_context is not None
                else int(public_state.current_turn)
            ),
            public_state=public_state,
            round_number=int(public_state.round_number),
            phase=str(public_state.phase),
            opponent_public_dice_count=tracker.opponent_public_dice_count_from_state(public_state),
            opponent_public_hand_count=tracker.opponent_public_hand_count_from_state(public_state),
            self_range=ExplicitRange((sampled_self_hypothesis,)),
            opponent_range=updated_opponent_range,
            public_history=tracker.public_history(),
            revealed_opponent_card_definition_ids=tracker.revealed_opponent_card_definition_ids(),
        )
        return _RangeUpdateResult(
            updated_public_belief=updated,
            sampled_self_hypothesis=sampled_self_hypothesis,
            sampled_opponent_hypothesis=sampled_opponent_hypothesis,
        )

    def _advance_opponent_range(
        self,
        *,
        pre_public_belief: PublicBeliefState,
        selected_action_code: int,
        sampled_self_hypothesis: ExplicitRangeHypothesis,
        sampled_opponent_hypothesis: ExplicitRangeHypothesis,
        deadline: float,
    ) -> ExplicitRange:
        updated: list[ExplicitRangeHypothesis] = []
        total_weight = 0.0
        for hypothesis in pre_public_belief.opponent_range.hypotheses:
            if time.perf_counter() > deadline:
                raise _SearchTimeout()
            env, context = self.reconstructor.reconstruct(
                public_belief=pre_public_belief,
                self_hypothesis=sampled_self_hypothesis,
                opponent_hypothesis=hypothesis,
            )
            if env is None or context is None:
                continue
            try:
                matched = match_low_level_code_index(context, selected_action_code)
                if matched is None:
                    continue
                _, step, _ = env.step(ActionChoice(action_code=context.legal_low_level_codes[matched]))
                next_hypothesis = _extract_player_hypothesis(
                    step.post_state,
                    player_id=1 - pre_public_belief.root_player,
                    weight=float(hypothesis.weight),
                )
                updated.append(next_hypothesis)
                total_weight += float(hypothesis.weight)
            finally:
                env.close()
        if not updated or total_weight <= 0.0:
            return ExplicitRange((replace_weight(sampled_opponent_hypothesis, 1.0),))
        normalized = tuple(
            replace_weight(hypothesis, float(hypothesis.weight) / total_weight)
            for hypothesis in updated
        )
        return ExplicitRange(normalized)


class SePotSearchController:
    def __init__(self, *, card_vocabulary: Sequence[int], config: SePotSearchConfig | None = None) -> None:
        self.card_vocabulary = tuple(int(value) for value in card_vocabulary)
        self.config = config or SePotSearchConfig()
        self._reconstructor_cache: dict[tuple[str, str | None, int | None], PublicStateReconstructor] = {}
        self._updater_cache: dict[tuple[str, str | None, int | None], PublicBeliefUpdater] = {}

    def choose_action(
        self,
        *,
        context: DecisionContext,
        encoded: Any,
        history: Sequence[TrajectoryStep],
        tracker: PublicStateTracker | None,
        opponent_deck_name: str,
        policy_log_probs: Sequence[float],
        belief_histogram: Sequence[float],
        belief_remaining_deck_histogram: Sequence[float],
        model: PpoTransformerPolicy,
        encoder: TokenObservationEncoder,
        env_config: EnvConfig | None,
        search_option_logits_fn,
        search_state_value_fn,
        sample: bool,
        temperature: float,
        device: torch.device,
    ) -> SearchDecision | None:
        del encoded, search_option_logits_fn
        if not should_trigger_search(context, config=self.config):
            return None
        budget = phase_budget_for_context(context, config=self.config)
        matchup = _matchup_from_key(str(context.metadata.get("matchup", "")))
        if matchup is None:
            return _fallback_decision(
                reason="missing_matchup",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=0,
            )
        if env_config is None:
            env_config = _env_config_from_matchup(matchup)
        if env_config is None:
            return _fallback_decision(
                reason="missing_env_config",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=0,
            )
        root_player = int(context.acting_player)
        root_tracker = _build_tracker(
            tracker=tracker,
            history=history,
            player_id=root_player,
        )
        public_belief = build_public_belief_state(
            context=context,
            tracker=root_tracker,
            card_vocabulary=self.card_vocabulary,
            opponent_deck_name=opponent_deck_name,
            belief_histogram=belief_histogram,
            belief_remaining_deck_histogram=belief_remaining_deck_histogram,
            belief_samples=budget.belief_samples,
            root_player=root_player,
        )
        if not context.legal_low_level_codes:
            return _fallback_decision(
                reason="no_legal_actions",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
        cache_key = (matchup.key, env_config.version, env_config.max_rounds)
        reconstructor = self._reconstructor_cache.get(cache_key)
        if reconstructor is None:
            reconstructor = PublicStateReconstructor(env_config=env_config, matchup=matchup, config=self.config)
            self._reconstructor_cache[cache_key] = reconstructor
        updater = self._updater_cache.get(cache_key)
        if updater is None or updater.reconstructor is not reconstructor:
            updater = PublicBeliefUpdater(reconstructor=reconstructor, card_vocabulary=self.card_vocabulary)
            self._updater_cache[cache_key] = updater
        policy_cache: dict[tuple[Any, ...], tuple[float, ...]] = {}
        leaf_value_cache: dict[tuple[Any, ...], float] = {}
        if reconstructor.ensure_template(acting_player=root_player) is None:
            return _fallback_decision(
                reason="template_unavailable",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
        deadline = time.perf_counter() + (float(budget.timeout_ms) / 1000.0)
        ranked = sorted(range(len(policy_log_probs)), key=lambda index: float(policy_log_probs[index]), reverse=True)
        root_indices = tuple(ranked[: max(1, min(len(ranked), budget.root_top_k))])
        if not root_indices:
            return _fallback_decision(
                reason="no_root_candidates",
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
        root_scores: dict[int, tuple[float, float]] = {}
        timed_out = False
        for root_index in root_indices:
            if time.perf_counter() > deadline:
                timed_out = True
                break
            candidate_action_code = int(context.legal_low_level_codes[root_index])
            sample_values: list[float] = []
            sample_weights: list[float] = []
            candidate_timed_out = False
            for hypothesis in public_belief.opponent_range.hypotheses[: budget.belief_samples]:
                if time.perf_counter() > deadline:
                    timed_out = True
                    candidate_timed_out = True
                    break
                try:
                    branch_value = self._evaluate_root_candidate(
                        root_context=context,
                        root_history=history,
                        root_tracker=root_tracker,
                        root_public_belief=public_belief,
                        root_action_code=candidate_action_code,
                        sampled_opponent_hypothesis=hypothesis,
                        model=model,
                        encoder=encoder,
                        device=device,
                        budget=budget,
                        deadline=deadline,
                        reconstructor=reconstructor,
                        updater=updater,
                        opponent_deck_name=opponent_deck_name,
                        search_state_value_fn=search_state_value_fn,
                        policy_cache=policy_cache,
                        leaf_value_cache=leaf_value_cache,
                    )
                except _SearchTimeout:
                    timed_out = True
                    candidate_timed_out = True
                    break
                if branch_value is None:
                    continue
                sample_values.append(float(branch_value))
                sample_weights.append(max(1.0e-6, float(hypothesis.weight)))
            if not sample_values:
                if candidate_timed_out:
                    break
                continue
            mean_value, std_value = _weighted_mean_std(sample_values, sample_weights)
            root_scores[root_index] = (mean_value, std_value)
            if candidate_timed_out:
                break
        if not root_scores:
            return _fallback_decision(
                reason=("timeout" if timed_out else "reconstruction_failed"),
                policy_log_probs=policy_log_probs,
                phase_depth=budget.depth,
                belief_sample_count=public_belief.opponent_range.count,
            )
        combined_scores = torch.full((len(context.legal_low_level_codes),), _masked_fill_value(), dtype=torch.float32, device=device)
        for root_index, (mean_value, std_value) in root_scores.items():
            combined_scores[root_index] = float(policy_log_probs[root_index]) + mean_value - (
                self.config.uncertainty_penalty * std_value
            )
        transformed = torch.softmax(combined_scores / max(1.0e-6, temperature), dim=-1)
        if sample:
            selected_action_index = int(torch.multinomial(transformed, num_samples=1).item())
        else:
            selected_action_index = int(torch.argmax(transformed).item())
        root_value = sum(
            float(transformed[index].item()) * float(root_scores[index][0])
            for index in root_scores
        )
        return SearchDecision(
            triggered=True,
            fallback_reason=None,
            selected_action_index=selected_action_index,
            transformed_policy=tuple(float(value) for value in transformed.detach().cpu().tolist()),
            root_value=float(root_value),
            root_candidate_count=len(root_scores),
            belief_sample_count=public_belief.opponent_range.count,
            phase_depth=budget.depth,
        )

    def _evaluate_root_candidate(
        self,
        *,
        root_context: DecisionContext,
        root_history: Sequence[TrajectoryStep],
        root_tracker: PublicStateTracker,
        root_public_belief: PublicBeliefState,
        root_action_code: int,
        sampled_opponent_hypothesis: ExplicitRangeHypothesis,
        model: PpoTransformerPolicy,
        encoder: TokenObservationEncoder,
        device: torch.device,
        budget: SePotPhaseBudget,
        deadline: float,
        reconstructor: PublicStateReconstructor,
        updater: PublicBeliefUpdater,
        opponent_deck_name: str,
        search_state_value_fn,
        policy_cache: dict[tuple[Any, ...], tuple[float, ...]],
        leaf_value_cache: dict[tuple[Any, ...], float],
    ) -> float | None:
        try:
            sampled_self_hypothesis = root_public_belief.self_range.hypotheses[0]
            env, current_context = reconstructor.reconstruct(
                public_belief=root_public_belief,
                self_hypothesis=sampled_self_hypothesis,
                opponent_hypothesis=sampled_opponent_hypothesis,
            )
            if env is None or current_context is None:
                return None
            branch_history = list(root_history)
            branch_tracker = root_tracker.clone(player_id=root_public_belief.root_player)
            try:
                matched_index = match_low_level_code_index(current_context, root_action_code)
                if matched_index is None:
                    return None
                selected_action_code = int(current_context.legal_low_level_codes[matched_index])
                next_context, step, _ = env.step(ActionChoice(action_code=selected_action_code))
                branch_history.append(step)
                branch_tracker.observe_step(step)
                sampled_self_hypothesis = _extract_player_hypothesis(step.post_state, player_id=root_public_belief.root_player)
                sampled_opponent_hypothesis = _extract_player_hypothesis(
                    step.post_state,
                    player_id=1 - root_public_belief.root_player,
                )
                update_result = updater.update_after_step(
                    pre_public_belief=root_public_belief,
                    selected_action_code=selected_action_code,
                    post_state=step.post_state,
                    next_context=next_context,
                    tracker=branch_tracker,
                    sampled_self_hypothesis=sampled_self_hypothesis,
                    sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                    deadline=deadline,
                )
                return self._rollout_value(
                    env=env,
                    current_context=next_context,
                    current_sampled_state_json=next_context.full_state_json,
                    current_public_belief=update_result.updated_public_belief,
                    current_history=branch_history,
                    root_player=root_public_belief.root_player,
                    budget=budget,
                    depth_remaining=max(0, budget.depth - 1),
                    model=model,
                    encoder=encoder,
                    device=device,
                    opponent_deck_name=opponent_deck_name,
                    search_state_value_fn=search_state_value_fn,
                    deadline=deadline,
                    updater=updater,
                    sampled_self_hypothesis=update_result.sampled_self_hypothesis,
                    sampled_opponent_hypothesis=update_result.sampled_opponent_hypothesis,
                    current_tracker=branch_tracker,
                    policy_cache=policy_cache,
                    leaf_value_cache=leaf_value_cache,
                )
            finally:
                env.close()
        except _SearchTimeout:
            raise

    def _rollout_value(
        self,
        *,
        env: GitcgDecisionEnv,
        current_context: DecisionContext | None,
        # Explicitly carry the sampled branch state instead of re-reading a live
        # runtime context's full-state payload. This keeps runtime SePoT on the
        # reconstructed sampled world it started from.
        current_sampled_state_json: str | None,
        current_public_belief: PublicBeliefState,
        current_history: list[TrajectoryStep],
        root_player: int,
        budget: SePotPhaseBudget,
        depth_remaining: int,
        model: PpoTransformerPolicy,
        encoder: TokenObservationEncoder,
        device: torch.device,
        opponent_deck_name: str,
        search_state_value_fn,
        deadline: float,
        updater: PublicBeliefUpdater,
        sampled_self_hypothesis: ExplicitRangeHypothesis,
        sampled_opponent_hypothesis: ExplicitRangeHypothesis,
        current_tracker: PublicStateTracker,
        policy_cache: dict[tuple[Any, ...], tuple[float, ...]],
        leaf_value_cache: dict[tuple[Any, ...], float],
    ) -> float:
        if time.perf_counter() > deadline:
            return _leaf_value(
                public_belief=current_public_belief,
                current_context=current_context,
                budget=budget,
                card_vocabulary=self.card_vocabulary,
                device=device,
                search_state_value_fn=search_state_value_fn,
                cache=leaf_value_cache,
            )
        if current_context is None or current_context.terminal:
            return _leaf_value(
                public_belief=current_public_belief,
                current_context=None,
                budget=budget,
                card_vocabulary=self.card_vocabulary,
                device=device,
                search_state_value_fn=search_state_value_fn,
                cache=leaf_value_cache,
            )
        (
            current_context,
            current_sampled_state_json,
            current_public_belief,
            current_history,
            sampled_self_hypothesis,
            sampled_opponent_hypothesis,
            current_tracker,
        ) = self._advance_until_action(
            env=env,
            current_context=current_context,
            current_sampled_state_json=current_sampled_state_json,
            current_public_belief=current_public_belief,
            current_history=current_history,
            root_player=root_player,
            model=model,
            encoder=encoder,
            device=device,
            opponent_deck_name=opponent_deck_name,
            deadline=deadline,
            updater=updater,
            sampled_self_hypothesis=sampled_self_hypothesis,
            sampled_opponent_hypothesis=sampled_opponent_hypothesis,
            current_tracker=current_tracker,
        )
        if current_context is None or current_context.terminal or depth_remaining <= 0:
            return _leaf_value(
                public_belief=current_public_belief,
                current_context=current_context,
                budget=budget,
                card_vocabulary=self.card_vocabulary,
                device=device,
                search_state_value_fn=search_state_value_fn,
                cache=leaf_value_cache,
            )
        if int(current_context.acting_player) != int(root_player):
            greedy_index = _greedy_action_index(
                context=current_context,
                history=current_history,
                public_history=current_public_belief.public_history,
                opponent_deck_name=opponent_deck_name,
                model=model,
                encoder=encoder,
                device=device,
                cache=policy_cache,
            )
            if greedy_index is None:
                return _leaf_value(
                    public_belief=current_public_belief,
                    current_context=current_context,
                    budget=budget,
                    card_vocabulary=self.card_vocabulary,
                    device=device,
                    search_state_value_fn=search_state_value_fn,
                )
            selected_action_code = int(current_context.legal_low_level_codes[greedy_index])
            next_context, step, _ = env.step(ActionChoice(action_code=selected_action_code))
            current_history = list(current_history)
            current_history.append(step)
            tracker = current_tracker.clone(player_id=root_player)
            tracker.observe_step(step)
            sampled_self_hypothesis = _extract_player_hypothesis(step.post_state, player_id=root_player)
            sampled_opponent_hypothesis = _extract_player_hypothesis(step.post_state, player_id=1 - root_player)
            update_result = updater.update_after_step(
                pre_public_belief=current_public_belief,
                selected_action_code=selected_action_code,
                post_state=step.post_state,
                next_context=next_context,
                tracker=tracker,
                sampled_self_hypothesis=sampled_self_hypothesis,
                sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                deadline=deadline,
            )
            return self._rollout_value(
                env=env,
                current_context=next_context,
                current_sampled_state_json=next_context.full_state_json,
                current_public_belief=update_result.updated_public_belief,
                current_history=current_history,
                root_player=root_player,
                budget=budget,
                depth_remaining=depth_remaining - 1,
                model=model,
                encoder=encoder,
                device=device,
                opponent_deck_name=opponent_deck_name,
                search_state_value_fn=search_state_value_fn,
                deadline=deadline,
                updater=updater,
                sampled_self_hypothesis=update_result.sampled_self_hypothesis,
                sampled_opponent_hypothesis=update_result.sampled_opponent_hypothesis,
                current_tracker=tracker,
                policy_cache=policy_cache,
                leaf_value_cache=leaf_value_cache,
            )
        log_probs = _policy_log_probs(
            context=current_context,
            history=current_history,
            public_history=current_public_belief.public_history,
            opponent_deck_name=opponent_deck_name,
            model=model,
            encoder=encoder,
            device=device,
            cache=policy_cache,
        )
        ranked = sorted(range(len(log_probs)), key=lambda index: float(log_probs[index]), reverse=True)
        candidate_indices = tuple(ranked[: max(1, min(len(ranked), budget.root_top_k))])
        branch_scores: list[float] = []
        for candidate_index in candidate_indices:
            if time.perf_counter() > deadline:
                break
            try:
                branch_value = self._evaluate_internal_self_candidate(
                    current_context=current_context,
                    current_sampled_state_json=current_sampled_state_json,
                    current_public_belief=current_public_belief,
                    current_history=current_history,
                    root_player=root_player,
                    budget=budget,
                    depth_remaining=depth_remaining,
                    model=model,
                    encoder=encoder,
                    device=device,
                    opponent_deck_name=opponent_deck_name,
                    search_state_value_fn=search_state_value_fn,
                    deadline=deadline,
                    updater=updater,
                    selected_action_code=int(current_context.legal_low_level_codes[candidate_index]),
                    env_config=getattr(env, "_config", None),
                    matchup=getattr(env, "_matchup", None),
                    current_tracker=current_tracker,
                    policy_cache=policy_cache,
                    leaf_value_cache=leaf_value_cache,
                )
            except _SearchTimeout:
                break
            if branch_value is not None:
                branch_scores.append(float(branch_value))
        if not branch_scores:
            return _leaf_value(
                public_belief=current_public_belief,
                current_context=current_context,
                budget=budget,
                card_vocabulary=self.card_vocabulary,
                device=device,
                search_state_value_fn=search_state_value_fn,
                cache=leaf_value_cache,
            )
        return max(branch_scores)

    def _evaluate_internal_self_candidate(
        self,
        *,
        current_context: DecisionContext,
        # This is the full-state JSON of the already reconstructed/sampled
        # branch, not the live root context's hidden truth.
        current_sampled_state_json: str | None,
        current_public_belief: PublicBeliefState,
        current_history: list[TrajectoryStep],
        root_player: int,
        budget: SePotPhaseBudget,
        depth_remaining: int,
        model: PpoTransformerPolicy,
        encoder: TokenObservationEncoder,
        device: torch.device,
        opponent_deck_name: str,
        search_state_value_fn,
        deadline: float,
        updater: PublicBeliefUpdater,
        selected_action_code: int,
        env_config: EnvConfig | None,
        matchup: Matchup | None,
        current_tracker: PublicStateTracker,
        policy_cache: dict[tuple[Any, ...], tuple[float, ...]],
        leaf_value_cache: dict[tuple[Any, ...], float],
    ) -> float | None:
        if current_sampled_state_json is None or env_config is None or matchup is None:
            return None
        child_env = GitcgDecisionEnv(
            env_config,
            matchup,
            enable_result_based_action_relabel=False,
        )
        try:
            child_context = child_env.reset(state_json=current_sampled_state_json)
            matched = match_low_level_code_index(child_context, selected_action_code)
            if matched is None:
                return None
            next_context, step, _ = child_env.step(ActionChoice(action_code=child_context.legal_low_level_codes[matched]))
            child_history = list(current_history)
            child_history.append(step)
            tracker = current_tracker.clone(player_id=root_player)
            tracker.observe_step(step)
            sampled_self_hypothesis = _extract_player_hypothesis(step.post_state, player_id=root_player)
            sampled_opponent_hypothesis = _extract_player_hypothesis(step.post_state, player_id=1 - root_player)
            update_result = updater.update_after_step(
                pre_public_belief=current_public_belief,
                selected_action_code=int(child_context.legal_low_level_codes[matched]),
                post_state=step.post_state,
                next_context=next_context,
                tracker=tracker,
                sampled_self_hypothesis=sampled_self_hypothesis,
                sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                deadline=deadline,
            )
            return self._rollout_value(
                env=child_env,
                current_context=next_context,
                current_sampled_state_json=next_context.full_state_json,
                current_public_belief=update_result.updated_public_belief,
                current_history=child_history,
                root_player=root_player,
                budget=budget,
                depth_remaining=depth_remaining - 1,
                model=model,
                encoder=encoder,
                device=device,
                opponent_deck_name=opponent_deck_name,
                search_state_value_fn=search_state_value_fn,
                deadline=deadline,
                updater=updater,
                sampled_self_hypothesis=update_result.sampled_self_hypothesis,
                sampled_opponent_hypothesis=update_result.sampled_opponent_hypothesis,
                current_tracker=tracker,
                policy_cache=policy_cache,
                leaf_value_cache=leaf_value_cache,
            )
        finally:
            child_env.close()

    def _advance_until_action(
        self,
        *,
        env: GitcgDecisionEnv,
        current_context: DecisionContext,
        current_sampled_state_json: str | None,
        current_public_belief: PublicBeliefState,
        current_history: list[TrajectoryStep],
        root_player: int,
        model: PpoTransformerPolicy,
        encoder: TokenObservationEncoder,
        device: torch.device,
        opponent_deck_name: str,
        deadline: float,
        updater: PublicBeliefUpdater,
        sampled_self_hypothesis: ExplicitRangeHypothesis,
        sampled_opponent_hypothesis: ExplicitRangeHypothesis,
        current_tracker: PublicStateTracker,
    ) -> tuple[
        DecisionContext | None,
        str | None,
        PublicBeliefState,
        list[TrajectoryStep],
        ExplicitRangeHypothesis,
        ExplicitRangeHypothesis,
        PublicStateTracker,
    ]:
        while current_context is not None and not current_context.terminal and current_context.request_type != DecisionType.ACTION:
            if time.perf_counter() > deadline:
                raise _SearchTimeout()
            action_index = _greedy_action_index(
                context=current_context,
                history=current_history,
                public_history=current_public_belief.public_history,
                opponent_deck_name=opponent_deck_name,
                model=model,
                encoder=encoder,
                device=device,
            )
            if action_index is None:
                break
            selected_action_code = int(current_context.legal_low_level_codes[action_index])
            next_context, step, _ = env.step(ActionChoice(action_code=selected_action_code))
            current_history.append(step)
            tracker = current_tracker.clone(player_id=root_player)
            tracker.observe_step(step)
            sampled_self_hypothesis = _extract_player_hypothesis(step.post_state, player_id=root_player)
            sampled_opponent_hypothesis = _extract_player_hypothesis(step.post_state, player_id=1 - root_player)
            update_result = updater.update_after_step(
                pre_public_belief=current_public_belief,
                selected_action_code=selected_action_code,
                post_state=step.post_state,
                next_context=next_context,
                tracker=tracker,
                sampled_self_hypothesis=sampled_self_hypothesis,
                sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                deadline=deadline,
            )
            current_public_belief = update_result.updated_public_belief
            sampled_self_hypothesis = update_result.sampled_self_hypothesis
            sampled_opponent_hypothesis = update_result.sampled_opponent_hypothesis
            current_context = next_context
            current_sampled_state_json = next_context.full_state_json if next_context is not None else None
            current_tracker = tracker
        return (
            current_context,
            current_sampled_state_json,
            current_public_belief,
            current_history,
            sampled_self_hypothesis,
            sampled_opponent_hypothesis,
            current_tracker,
        )


def _build_self_range_from_state(state: StateSnapshot, *, player_id: int) -> ExplicitRange:
    player = state.players[player_id]
    return ExplicitRange(
        hypotheses=(
            ExplicitRangeHypothesis(
                hand_definition_ids=tuple(int(card.definition_id) for card in player.hand_cards),
                deck_definition_ids=tuple(int(card.definition_id) for card in player.pile_cards),
                hidden_dice=tuple(int(die) for die in player.dice),
                weight=1.0,
            ),
        )
    )


def _build_opponent_range(
    *,
    state: StateSnapshot,
    root_player: int,
    tracker: PublicStateTracker | None,
    card_vocabulary: Sequence[int],
    opponent_deck_name: str,
    belief_histogram: Sequence[float],
    belief_remaining_deck_histogram: Sequence[float],
    belief_samples: int,
) -> ExplicitRange:
    opponent_player = 1 - int(root_player)
    opponent = state.players[opponent_player]
    opponent_hand_count = (
        int(tracker.opponent_public_hand_count_from_state(state))
        if tracker is not None
        else int(len(opponent.hand_cards))
    )
    opponent_dice_count = (
        int(round(tracker.opponent_public_dice_count_from_state(state)))
        if tracker is not None
        else int(len(opponent.dice))
    )
    revealed_counts = Counter(tracker.revealed_opponent_card_definition_ids() if tracker is not None else ())
    try:
        deck_cards = tuple(int(card_id) for card_id in deck_by_name(opponent_deck_name).cards)
    except Exception:
        deck_cards = tuple(int(card_id) for card_id in card_vocabulary)
    remaining_cards = Counter(deck_cards)
    for definition_id, count in revealed_counts.items():
        if definition_id in remaining_cards:
            remaining_cards[definition_id] = max(0, remaining_cards[definition_id] - int(count))
    ordered_cards = [card_id for card_id, count in remaining_cards.items() for _ in range(max(0, count))]
    if not ordered_cards:
        ordered_cards = list(deck_cards)
    vocabulary = list(int(value) for value in card_vocabulary)
    belief_scores = {definition_id: 0.0 for definition_id in vocabulary}
    for definition_id, score in zip(vocabulary, belief_histogram):
        belief_scores[int(definition_id)] = float(score)
    deck_scores = {definition_id: 0.0 for definition_id in vocabulary}
    for definition_id, score in zip(vocabulary, belief_remaining_deck_histogram):
        deck_scores[int(definition_id)] = float(score)
    sorted_hand_cards = sorted(
        ordered_cards,
        key=lambda definition_id: (
            belief_scores.get(int(definition_id), 0.0),
            deck_scores.get(int(definition_id), 0.0),
        ),
        reverse=True,
    )
    sample_count = max(1, int(belief_samples))
    hypotheses: list[ExplicitRangeHypothesis] = []
    for sample_index in range(sample_count):
        hand_cards = (
            tuple(
                sorted_hand_cards[(sample_index + offset) % len(sorted_hand_cards)]
                for offset in range(opponent_hand_count)
            )
            if sorted_hand_cards
            else ()
        )
        hand_counter = Counter(hand_cards)
        deck_after_hand = Counter(remaining_cards)
        for definition_id, count in hand_counter.items():
            if definition_id in deck_after_hand:
                deck_after_hand[definition_id] = max(0, deck_after_hand[definition_id] - count)
        deck_cards_after = tuple(
            card_id
            for card_id, count in sorted(deck_after_hand.items())
            for _ in range(max(0, count))
        )
        hidden_dice = tuple(_sample_hidden_dice(sample_index=sample_index, count=opponent_dice_count))
        hypotheses.append(
            ExplicitRangeHypothesis(
                hand_definition_ids=hand_cards,
                deck_definition_ids=deck_cards_after,
                hidden_dice=hidden_dice,
                weight=1.0 / float(sample_count),
            )
        )
    return ExplicitRange(hypotheses=tuple(hypotheses))


def _policy_log_probs(
    *,
    context: DecisionContext,
    history: Sequence[TrajectoryStep],
    public_history: Sequence[PublicActionRecord],
    opponent_deck_name: str,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    cache: dict[tuple[Any, ...], tuple[float, ...]] | None = None,
) -> tuple[float, ...]:
    cache_key = _context_policy_cache_key(
        context=context,
        public_history=public_history,
        opponent_deck_name=opponent_deck_name,
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    tracker = PublicStateTracker.from_public_history(
        player_id=int(context.acting_player),
        history=tuple(public_history),
    )
    encoded = encoder.encode_context(
        context,
        history=history,
        opponent_tag="unknown",
        opponent_entity_key="unknown",
        opponent_deck_name=opponent_deck_name,
        opponent_public_dice_count=tracker.opponent_public_dice_count(context),
        include_training_targets=False,
    )
    token_features = torch.as_tensor(encoded.token_features, dtype=torch.float32, device=device).unsqueeze(0)
    token_mask = torch.as_tensor(encoded.token_mask, dtype=torch.bool, device=device).unsqueeze(0)
    opponent_token_mask = torch.as_tensor(encoded.opponent_token_mask, dtype=torch.bool, device=device).unsqueeze(0)
    option_features = torch.as_tensor(encoded.option_features, dtype=torch.float32, device=device).unsqueeze(0)
    option_mask = torch.as_tensor(encoded.option_mask, dtype=torch.bool, device=device).unsqueeze(0)
    opponent_tag_ids = torch.as_tensor([encoded.opponent_tag_id], dtype=torch.long, device=device)
    opponent_entity_ids = torch.as_tensor([encoded.opponent_entity_id], dtype=torch.long, device=device)
    opponent_deck_ids = torch.as_tensor([encoded.opponent_deck_id], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            token_features=token_features,
            token_mask=token_mask,
            option_features=option_features,
            option_mask=option_mask,
            opponent_token_mask=opponent_token_mask,
            opponent_tag_ids=opponent_tag_ids,
            opponent_entity_ids=opponent_entity_ids,
            opponent_deck_ids=opponent_deck_ids,
            recurrent_state=None,
        )
        log_probs = masked_log_softmax(output.policy_logits[0], option_mask[0])
    result = tuple(float(value) for value in log_probs.detach().cpu().tolist())
    if cache is not None:
        cache[cache_key] = result
    return result


def _context_policy_cache_key(
    *,
    context: DecisionContext,
    public_history: Sequence[PublicActionRecord],
    opponent_deck_name: str,
) -> tuple[Any, ...]:
    player_view = context.player_view
    round_number = int(getattr(player_view, "round_number", 0)) if player_view is not None else 0
    phase = str(getattr(player_view, "phase", "")) if player_view is not None else ""
    semantic_options = tuple(
        semantic_action_key_for_code(int(action_code))
        for action_code in context.legal_low_level_codes
    )
    return (
        int(context.acting_player),
        str(context.request_type.value),
        round_number,
        phase,
        len(public_history),
        str(opponent_deck_name),
        semantic_options,
    )


def _hypothesis_cache_key(hypothesis: ExplicitRangeHypothesis) -> tuple[Any, ...]:
    return (
        hypothesis.hand_definition_ids,
        hypothesis.deck_definition_ids,
        hypothesis.hidden_dice,
    )


def _public_belief_cache_key(public_belief: PublicBeliefState) -> tuple[Any, ...]:
    return (
        int(public_belief.root_player),
        int(public_belief.acting_player),
        int(public_belief.round_number),
        str(public_belief.phase),
        tuple(record.semantic_key for record in public_belief.public_history),
        tuple(
            _hypothesis_cache_key(hypothesis) + (float(hypothesis.weight),)
            for hypothesis in public_belief.self_range.hypotheses
        ),
        tuple(
            _hypothesis_cache_key(hypothesis) + (float(hypothesis.weight),)
            for hypothesis in public_belief.opponent_range.hypotheses
        ),
    )


def _greedy_action_index(
    *,
    context: DecisionContext,
    history: Sequence[TrajectoryStep],
    public_history: Sequence[PublicActionRecord],
    opponent_deck_name: str,
    model: PpoTransformerPolicy,
    encoder: TokenObservationEncoder,
    device: torch.device,
    cache: dict[tuple[Any, ...], tuple[float, ...]] | None = None,
) -> int | None:
    if not context.legal_low_level_codes or context.player_view is None:
        return None
    log_probs = _policy_log_probs(
        context=context,
        history=history,
        public_history=public_history,
        opponent_deck_name=opponent_deck_name,
        model=model,
        encoder=encoder,
        device=device,
        cache=cache,
    )
    return max(range(len(log_probs)), key=lambda index: log_probs[index])


def _extract_player_hypothesis(
    state: StateSnapshot,
    *,
    player_id: int,
    weight: float = 1.0,
) -> ExplicitRangeHypothesis:
    player = state.players[player_id]
    return ExplicitRangeHypothesis(
        hand_definition_ids=tuple(int(card.definition_id) for card in player.hand_cards),
        deck_definition_ids=tuple(int(card.definition_id) for card in player.pile_cards),
        hidden_dice=tuple(int(die) for die in player.dice),
        weight=float(weight),
    )


def _leaf_value(
    *,
    public_belief: PublicBeliefState,
    current_context: DecisionContext | None,
    budget: SePotPhaseBudget,
    card_vocabulary: Sequence[int],
    device: torch.device,
    search_state_value_fn,
    cache: dict[tuple[Any, ...], float] | None = None,
) -> float:
    cache_key = (
        tuple(range_summary(public_belief.self_range, card_vocabulary=card_vocabulary)),
        tuple(range_summary(public_belief.opponent_range, card_vocabulary=card_vocabulary)),
        tuple(_belief_context_summary(public_belief=public_belief, current_context=current_context, budget=budget)),
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    self_range = cache_key[0]
    opponent_range = cache_key[1]
    context_summary_tuple = cache_key[2]
    self_summary = torch.as_tensor(
        self_range,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    opponent_summary = torch.as_tensor(
        opponent_range,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    context_summary = torch.as_tensor(
        context_summary_tuple,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        value = search_state_value_fn(
            self_range_summary=self_summary,
            opponent_range_summary=opponent_summary,
            search_context_summary=context_summary,
        )
    result = float(value[0].item())
    if cache is not None:
        cache[cache_key] = result
    return result


def _belief_context_summary(
    *,
    public_belief: PublicBeliefState,
    current_context: DecisionContext | None,
    budget: SePotPhaseBudget,
) -> tuple[float, ...]:
    legal_count = float(len(current_context.legal_low_level_codes)) if current_context is not None else 0.0
    has_end = bool(
        current_context is not None
        and any(
            low_level_kind_for_code(int(action_code)) == OptionKind.ACTION_DECLARE_END
            for action_code in current_context.legal_low_level_codes
        )
    )
    productive = float(
        sum(
            1
            for action_code in (current_context.legal_low_level_codes if current_context is not None else ())
            if low_level_kind_for_code(int(action_code)) in {
                OptionKind.ACTION_USE_SKILL,
                OptionKind.ACTION_PLAY_CARD,
                OptionKind.ACTION_ELEMENTAL_TUNING,
            }
        )
    )
    return (
        min(float(public_belief.round_number), 20.0) / 20.0,
        min(legal_count, 16.0) / 16.0,
        1.0 if str(public_belief.phase) == "action" else 0.0,
        1.0 if has_end else 0.0,
        min(productive, 8.0) / 8.0,
        1.0 if float(public_belief.round_number) >= 9.0 else 0.0,
        1.0 if float(public_belief.round_number) >= 4.0 else 0.0,
        1.0 if int(public_belief.acting_player) == 0 else 0.0,
        min(float(budget.depth), 4.0) / 4.0,
        min(float(budget.root_top_k), 8.0) / 8.0,
        min(float(budget.belief_samples), 8.0) / 8.0,
        min(float(budget.timeout_ms), 500.0) / 500.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )


def _build_tracker(
    *,
    tracker: PublicStateTracker | None,
    history: Sequence[TrajectoryStep],
    player_id: int,
) -> PublicStateTracker:
    if tracker is not None:
        return tracker.clone(player_id=player_id)
    rebuilt = PublicStateTracker(player_id=player_id)
    for step in history:
        rebuilt.observe_step(step)
    return rebuilt


def _matchup_from_key(value: str) -> Matchup | None:
    if "__vs__" not in value:
        return None
    deck0, deck1 = value.split("__vs__", maxsplit=1)
    if not deck0 or not deck1:
        return None
    return Matchup(deck0, deck1)


def _env_config_from_matchup(matchup: Matchup) -> EnvConfig | None:
    try:
        deck0 = deck_by_name(matchup.deck0)
        deck1 = deck_by_name(matchup.deck1)
    except KeyError:
        return None
    deck_pool = (deck0,) if deck0.name == deck1.name else (deck0, deck1)
    return EnvConfig(deck_pool=deck_pool)


def _sample_hidden_dice(*, sample_index: int, count: int) -> list[int]:
    if count <= 0:
        return []
    return [_DICE_TYPES[(sample_index + offset) % len(_DICE_TYPES)] for offset in range(count)]


def _fallback_decision(
    *,
    reason: str,
    policy_log_probs: Sequence[float],
    phase_depth: int,
    belief_sample_count: int,
    root_candidate_count: int | None = None,
) -> SearchDecision:
    transformed_policy = _softmax_tuple(policy_log_probs)
    if transformed_policy:
        selected_action_index = int(max(range(len(transformed_policy)), key=lambda index: transformed_policy[index]))
    else:
        selected_action_index = 0
    effective_root_candidate_count = (
        int(root_candidate_count)
        if root_candidate_count is not None
        else len(transformed_policy)
    )
    return SearchDecision(
        triggered=True,
        fallback_reason=reason,
        selected_action_index=selected_action_index,
        transformed_policy=tuple(transformed_policy),
        root_value=0.0,
        root_candidate_count=max(0, effective_root_candidate_count),
        belief_sample_count=int(belief_sample_count),
        phase_depth=int(phase_depth),
    )


def _softmax_tuple(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        return ()
    max_logit = max(float(value) for value in logits)
    exp_values = [math.exp(float(value) - max_logit) for value in logits]
    total = sum(exp_values)
    if total <= 0.0:
        return tuple(0.0 for _ in logits)
    return tuple(float(value / total) for value in exp_values)


def _weighted_mean_std(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    total = max(1.0e-6, sum(float(weight) for weight in weights))
    mean = sum(float(value) * float(weight) for value, weight in zip(values, weights)) / total
    variance = sum(float(weight) * ((float(value) - mean) ** 2) for value, weight in zip(values, weights)) / total
    return float(mean), float(math.sqrt(max(0.0, variance)))


def replace_weight(hypothesis: ExplicitRangeHypothesis, weight: float) -> ExplicitRangeHypothesis:
    return ExplicitRangeHypothesis(
        hand_definition_ids=hypothesis.hand_definition_ids,
        deck_definition_ids=hypothesis.deck_definition_ids,
        hidden_dice=hypothesis.hidden_dice,
        weight=float(weight),
    )


def _masked_fill_value() -> float:
    return float(torch.finfo(torch.float32).min)
