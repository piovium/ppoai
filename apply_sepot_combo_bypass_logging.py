
from __future__ import annotations

import re
import sys
from pathlib import Path
import textwrap


TARGET = Path("research/world_model/src/gitcg_world_model/sepot_search.py")


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(f"missing snippet for {label}")
    return new_text


def indent_method(body: str) -> str:
    return textwrap.indent(textwrap.dedent(body).strip() + "\n", "    ")


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    if not target.exists():
        raise FileNotFoundError(target)
    text = target.read_text(encoding="utf-8")
    backup = target.with_suffix(target.suffix + ".bak_phase34_combo_bypass_logging")
    backup.write_text(text, encoding="utf-8")

    if "import os\n" not in text:
        text = text.replace(
            "import json\nimport math\nimport time\n",
            "import json\nimport math\nimport os\nimport time\n",
            1,
        )

    text = replace_once(
        text,
        r"@dataclass\(frozen=True\)\nclass _RangeUpdateResult:.*?class _SearchTimeout\(RuntimeError\):\n    pass\n",
        textwrap.dedent(
            """
            @dataclass(frozen=True)
            class _RangeUpdateResult:
                updated_public_belief: PublicBeliefState
                sampled_self_hypothesis: ExplicitRangeHypothesis
                sampled_opponent_hypothesis: ExplicitRangeHypothesis


            @dataclass(frozen=True)
            class _ObservationDelta:
                hand_delta: int
                dice_delta: int
                revealed_delta_count: int
                has_public_delta: bool
                requires_exact: bool


            class _SearchTimeout(RuntimeError):
                def __init__(self, stage: str = "timeout") -> None:
                    self.stage = str(stage)
                    super().__init__(self.stage)
            """
        ).strip() + "\n",
        "timeout class and observation delta",
    )

    text = replace_once(
        text,
        r"class PublicBeliefUpdater:.*?class SePotSearchController:\n",
        textwrap.dedent(
            """
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
                    self._partial_update_cache: dict[tuple[Any, ...], ExplicitRange] = {}

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
                    runtime_stats: dict[str, float] | None = None,
                ) -> _RangeUpdateResult:
                    if time.perf_counter() > deadline:
                        raise _SearchTimeout("posterior_update")
                    public_state = mask_state_for_player(post_state, perspective_player=pre_public_belief.root_player)
                    next_public_hand_count = tracker.opponent_public_hand_count_from_state(public_state)
                    next_public_dice_count = tracker.opponent_public_dice_count_from_state(public_state)
                    next_revealed = tuple(int(value) for value in tracker.revealed_opponent_card_definition_ids())
                    observation_delta = _observation_delta(
                        pre_public_belief=pre_public_belief,
                        next_public_hand_count=next_public_hand_count,
                        next_public_dice_count=next_public_dice_count,
                        next_revealed=next_revealed,
                    )
                    cache_key = (
                        _public_belief_cache_key(pre_public_belief),
                        semantic_action_key_for_code(selected_action_code),
                        _hypothesis_cache_key(sampled_self_hypothesis),
                        _hypothesis_cache_key(sampled_opponent_hypothesis),
                        int(observation_delta.hand_delta),
                        int(observation_delta.dice_delta),
                        int(observation_delta.revealed_delta_count),
                        int(observation_delta.requires_exact),
                    )
                    updated_opponent_range = self._advance_cache.get(cache_key)
                    if updated_opponent_range is None:
                        if not observation_delta.has_public_delta:
                            if runtime_stats is not None:
                                runtime_stats["posterior_keep"] = float(runtime_stats.get("posterior_keep", 0.0) + 1.0)
                            updated_opponent_range = ExplicitRange((replace_weight(sampled_opponent_hypothesis, 1.0),))
                        elif not observation_delta.requires_exact:
                            if runtime_stats is not None:
                                runtime_stats["posterior_partial"] = float(runtime_stats.get("posterior_partial", 0.0) + 1.0)
                            partial_key = cache_key
                            updated_opponent_range = self._partial_update_cache.get(partial_key)
                            if updated_opponent_range is None:
                                approx_hypothesis = _approximate_opponent_hypothesis_after_public_delta(
                                    sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                                    next_public_hand_count=next_public_hand_count,
                                    next_public_dice_count=next_public_dice_count,
                                    next_revealed=next_revealed,
                                )
                                updated_opponent_range = ExplicitRange((replace_weight(approx_hypothesis, 1.0),))
                                self._partial_update_cache[partial_key] = updated_opponent_range
                                if len(self._partial_update_cache) > 512:
                                    self._partial_update_cache.clear()
                        else:
                            if runtime_stats is not None:
                                runtime_stats["posterior_exact"] = float(runtime_stats.get("posterior_exact", 0.0) + 1.0)
                            updated_opponent_range = self._advance_opponent_range(
                                pre_public_belief=pre_public_belief,
                                selected_action_code=selected_action_code,
                                sampled_self_hypothesis=sampled_self_hypothesis,
                                sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                                deadline=deadline,
                                runtime_stats=runtime_stats,
                            )
                        if updated_opponent_range.count > 0:
                            self._advance_cache[cache_key] = updated_opponent_range
                            if len(self._advance_cache) > 512:
                                self._advance_cache.clear()
                    sampled_next_opponent = (
                        updated_opponent_range.hypotheses[0]
                        if updated_opponent_range.hypotheses
                        else replace_weight(sampled_opponent_hypothesis, 1.0)
                    )
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
                        opponent_public_dice_count=float(next_public_dice_count),
                        opponent_public_hand_count=float(next_public_hand_count),
                        self_range=ExplicitRange((replace_weight(sampled_self_hypothesis, 1.0),)),
                        opponent_range=updated_opponent_range,
                        public_history=tracker.public_history(),
                        revealed_opponent_card_definition_ids=next_revealed,
                    )
                    return _RangeUpdateResult(
                        updated_public_belief=updated,
                        sampled_self_hypothesis=replace_weight(sampled_self_hypothesis, 1.0),
                        sampled_opponent_hypothesis=replace_weight(sampled_next_opponent, 1.0),
                    )

                def _can_keep_sampled_opponent(
                    self,
                    *,
                    pre_public_belief: PublicBeliefState,
                    post_public_state: StateSnapshot,
                    tracker: PublicStateTracker,
                ) -> bool:
                    next_public_hand = tracker.opponent_public_hand_count_from_state(post_public_state)
                    next_public_dice = tracker.opponent_public_dice_count_from_state(post_public_state)
                    next_revealed = tracker.revealed_opponent_card_definition_ids()
                    return (
                        abs(float(next_public_hand) - float(pre_public_belief.opponent_public_hand_count)) <= 1.0e-6
                        and abs(float(next_public_dice) - float(pre_public_belief.opponent_public_dice_count)) <= 1.0e-6
                        and tuple(int(value) for value in next_revealed) == tuple(int(value) for value in pre_public_belief.revealed_opponent_card_definition_ids)
                    )

                def _advance_opponent_range(
                    self,
                    *,
                    pre_public_belief: PublicBeliefState,
                    selected_action_code: int,
                    sampled_self_hypothesis: ExplicitRangeHypothesis,
                    sampled_opponent_hypothesis: ExplicitRangeHypothesis,
                    deadline: float,
                    runtime_stats: dict[str, float] | None = None,
                ) -> ExplicitRange:
                    exact_start = time.perf_counter()
                    updated: list[ExplicitRangeHypothesis] = []
                    total_weight = 0.0
                    hypotheses = tuple(pre_public_belief.opponent_range.hypotheses)
                    for hypothesis in hypotheses[: max(1, min(2, len(hypotheses)))]:
                        now = time.perf_counter()
                        if now > deadline:
                            if updated and total_weight > 0.0:
                                break
                            raise _SearchTimeout("posterior_exact")
                        env, context = self.reconstructor.reconstruct(
                            public_belief=pre_public_belief,
                            self_hypothesis=sampled_self_hypothesis,
                            opponent_hypothesis=hypothesis,
                        )
                        if env is None or context is None:
                            continue
                        try:
                            if time.perf_counter() > deadline:
                                if updated and total_weight > 0.0:
                                    break
                                raise _SearchTimeout("posterior_exact")
                            matched = match_low_level_code_index(context, selected_action_code)
                            if matched is None:
                                continue
                            if time.perf_counter() > deadline:
                                if updated and total_weight > 0.0:
                                    break
                                raise _SearchTimeout("posterior_exact")
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
                    if runtime_stats is not None:
                        runtime_stats["posterior_exact_ms"] = float(runtime_stats.get("posterior_exact_ms", 0.0) + ((time.perf_counter() - exact_start) * 1000.0))
                    if not updated or total_weight <= 0.0:
                        return ExplicitRange((replace_weight(sampled_opponent_hypothesis, 1.0),))
                    normalized = tuple(
                        replace_weight(hypothesis, float(hypothesis.weight) / total_weight)
                        for hypothesis in updated
                    )
                    return ExplicitRange(normalized)


            class SePotSearchController:
            """
        ).strip() + "\n",
        "PublicBeliefUpdater class",
    )

    text = replace_once(
        text,
        r"    def choose_action\(\n.*?\n    def _evaluate_root_candidate\(",
        indent_method(
            """
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
                search_start = time.perf_counter()
                runtime_stats: dict[str, float] = {
                    "posterior_keep": 0.0,
                    "posterior_partial": 0.0,
                    "posterior_exact": 0.0,
                    "posterior_exact_ms": 0.0,
                    "combo_evals": 0.0,
                }
                matchup = _matchup_from_key(str(context.metadata.get("matchup", "")))
                if matchup is None:
                    decision = _fallback_decision(
                        reason="missing_matchup",
                        policy_log_probs=policy_log_probs,
                        phase_depth=budget.depth,
                        belief_sample_count=0,
                    )
                    _runtime_log_search_outcome(
                        outcome="fallback",
                        decision=decision,
                        runtime_ms=(time.perf_counter() - search_start) * 1000.0,
                        runtime_stats=runtime_stats,
                    )
                    return decision
                if env_config is None:
                    env_config = _env_config_from_matchup(matchup)
                if env_config is None:
                    decision = _fallback_decision(
                        reason="missing_env_config",
                        policy_log_probs=policy_log_probs,
                        phase_depth=budget.depth,
                        belief_sample_count=0,
                    )
                    _runtime_log_search_outcome(
                        outcome="fallback",
                        decision=decision,
                        runtime_ms=(time.perf_counter() - search_start) * 1000.0,
                        runtime_stats=runtime_stats,
                    )
                    return decision
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
                    decision = _fallback_decision(
                        reason="no_legal_actions",
                        policy_log_probs=policy_log_probs,
                        phase_depth=budget.depth,
                        belief_sample_count=public_belief.opponent_range.count,
                    )
                    _runtime_log_search_outcome(
                        outcome="fallback",
                        decision=decision,
                        runtime_ms=(time.perf_counter() - search_start) * 1000.0,
                        runtime_stats=runtime_stats,
                    )
                    return decision
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
                deadline = time.perf_counter() + (float(budget.timeout_ms) / 1000.0)
                root_indices = _prioritized_root_indices(
                    context=context,
                    policy_log_probs=policy_log_probs,
                    budget=budget,
                )
                if not root_indices:
                    decision = _fallback_decision(
                        reason="no_root_candidates",
                        policy_log_probs=policy_log_probs,
                        phase_depth=budget.depth,
                        belief_sample_count=public_belief.opponent_range.count,
                        root_candidate_count=0,
                    )
                    _runtime_log_search_outcome(
                        outcome="fallback",
                        decision=decision,
                        runtime_ms=(time.perf_counter() - search_start) * 1000.0,
                        runtime_stats=runtime_stats,
                    )
                    return decision
                root_scores: dict[int, tuple[float, float]] = {}
                timed_out = False
                timed_out_stage = "root_eval"
                for root_index in root_indices:
                    if time.perf_counter() > deadline:
                        timed_out = True
                        timed_out_stage = "root_budget"
                        break
                    candidate_action_code = int(context.legal_low_level_codes[root_index])
                    sample_values: list[float] = []
                    sample_weights: list[float] = []
                    candidate_timed_out = False
                    for hypothesis in public_belief.opponent_range.hypotheses[: budget.belief_samples]:
                        if time.perf_counter() > deadline:
                            timed_out = True
                            timed_out_stage = "root_budget"
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
                                env_config=env_config,
                                matchup=matchup,
                                runtime_stats=runtime_stats,
                            )
                        except _SearchTimeout as exc:
                            timed_out = True
                            timed_out_stage = getattr(exc, "stage", "root_eval")
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
                    decision = _fallback_decision(
                        reason=(f"timeout_{timed_out_stage}" if timed_out else "reconstruction_failed"),
                        policy_log_probs=policy_log_probs,
                        phase_depth=budget.depth,
                        belief_sample_count=public_belief.opponent_range.count,
                        root_candidate_count=len(root_indices),
                    )
                    _runtime_log_search_outcome(
                        outcome="fallback",
                        decision=decision,
                        runtime_ms=(time.perf_counter() - search_start) * 1000.0,
                        runtime_stats=runtime_stats,
                    )
                    return decision
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
                decision = SearchDecision(
                    triggered=True,
                    fallback_reason=None,
                    selected_action_index=selected_action_index,
                    transformed_policy=tuple(float(value) for value in transformed.detach().cpu().tolist()),
                    root_value=float(root_value),
                    root_candidate_count=len(root_scores),
                    belief_sample_count=public_belief.opponent_range.count,
                    phase_depth=budget.depth,
                )
                _runtime_log_search_outcome(
                    outcome="success",
                    decision=decision,
                    runtime_ms=(time.perf_counter() - search_start) * 1000.0,
                    runtime_stats=runtime_stats,
                )
                return decision

            def _evaluate_root_candidate(
            """
        ),
        "choose_action method",
    )

    text = replace_once(
        text,
        r"    def _evaluate_root_candidate\(\n.*?\n    def _rollout_value\(",
        indent_method(
            """
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
                env_config: EnvConfig,
                matchup: Matchup,
                runtime_stats: dict[str, float] | None = None,
            ) -> float | None:
                try:
                    sampled_self_hypothesis = root_public_belief.self_range.hypotheses[0]
                    env, current_context = _try_reset_branch_env(
                        env_config=env_config,
                        matchup=matchup,
                        state_json=root_context.full_state_json,
                    )
                    if env is None or current_context is None:
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
                        next_depth = max(0, budget.depth - 1)
                        if next_depth <= 0:
                            cheap_public_belief = _singleton_public_belief_after_step(
                                pre_public_belief=root_public_belief,
                                post_state=step.post_state,
                                next_context=next_context,
                                tracker=branch_tracker,
                                sampled_self_hypothesis=sampled_self_hypothesis,
                                sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                            )
                            return _leaf_value(
                                public_belief=cheap_public_belief,
                                current_context=next_context,
                                budget=budget,
                                card_vocabulary=self.card_vocabulary,
                                device=device,
                                search_state_value_fn=search_state_value_fn,
                                cache=leaf_value_cache,
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
                            runtime_stats=runtime_stats,
                        )
                        return self._rollout_value(
                            env=env,
                            current_context=next_context,
                            current_sampled_state_json=next_context.full_state_json,
                            current_public_belief=update_result.updated_public_belief,
                            current_history=branch_history,
                            root_player=root_public_belief.root_player,
                            budget=budget,
                            depth_remaining=next_depth,
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
                            env_config=env_config,
                            matchup=matchup,
                            runtime_stats=runtime_stats,
                        )
                    finally:
                        env.close()
                except _SearchTimeout:
                    raise

            def _rollout_value(
            """
        ),
        "_evaluate_root_candidate method",
    )

    text = replace_once(
        text,
        r"    def _rollout_value\(\n.*?\n    def _evaluate_internal_self_candidate\(",
        indent_method(
            """
            def _rollout_value(
                self,
                *,
                env: GitcgDecisionEnv,
                current_context: DecisionContext | None,
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
                env_config: EnvConfig,
                matchup: Matchup,
                runtime_stats: dict[str, float] | None = None,
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
                    runtime_stats=runtime_stats,
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
                    next_depth = depth_remaining - 1
                    if next_depth <= 0:
                        cheap_public_belief = _singleton_public_belief_after_step(
                            pre_public_belief=current_public_belief,
                            post_state=step.post_state,
                            next_context=next_context,
                            tracker=tracker,
                            sampled_self_hypothesis=sampled_self_hypothesis,
                            sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                        )
                        return _leaf_value(
                            public_belief=cheap_public_belief,
                            current_context=next_context,
                            budget=budget,
                            card_vocabulary=self.card_vocabulary,
                            device=device,
                            search_state_value_fn=search_state_value_fn,
                            cache=leaf_value_cache,
                        )
                    update_result = updater.update_after_step(
                        pre_public_belief=current_public_belief,
                        selected_action_code=selected_action_code,
                        post_state=step.post_state,
                        next_context=next_context,
                        tracker=tracker,
                        sampled_self_hypothesis=sampled_self_hypothesis,
                        sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                        deadline=deadline,
                        runtime_stats=runtime_stats,
                    )
                    return self._rollout_value(
                        env=env,
                        current_context=next_context,
                        current_sampled_state_json=next_context.full_state_json,
                        current_public_belief=update_result.updated_public_belief,
                        current_history=current_history,
                        root_player=root_player,
                        budget=budget,
                        depth_remaining=next_depth,
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
                        env_config=env_config,
                        matchup=matchup,
                        runtime_stats=runtime_stats,
                    )
                combo_value = self._evaluate_self_followup_value(
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
                    current_tracker=current_tracker,
                    policy_cache=policy_cache,
                    leaf_value_cache=leaf_value_cache,
                    env_config=env_config,
                    matchup=matchup,
                    runtime_stats=runtime_stats,
                )
                if combo_value is not None:
                    return float(combo_value)
                return _leaf_value(
                    public_belief=current_public_belief,
                    current_context=current_context,
                    budget=budget,
                    card_vocabulary=self.card_vocabulary,
                    device=device,
                    search_state_value_fn=search_state_value_fn,
                    cache=leaf_value_cache,
                )

            def _evaluate_self_followup_value(
                self,
                *,
                current_context: DecisionContext,
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
                current_tracker: PublicStateTracker,
                policy_cache: dict[tuple[Any, ...], tuple[float, ...]],
                leaf_value_cache: dict[tuple[Any, ...], float],
                env_config: EnvConfig,
                matchup: Matchup,
                runtime_stats: dict[str, float] | None = None,
            ) -> float | None:
                if current_sampled_state_json is None:
                    return None
                if depth_remaining <= 0:
                    return None
                if current_context.request_type != DecisionType.ACTION:
                    return None
                if not _should_expand_self_followup_action(current_context):
                    return None
                followup_policy = _policy_log_probs(
                    context=current_context,
                    history=current_history,
                    public_history=current_public_belief.public_history,
                    opponent_deck_name=opponent_deck_name,
                    model=model,
                    encoder=encoder,
                    device=device,
                    cache=policy_cache,
                )
                followup_budget = SePotPhaseBudget(
                    depth=1,
                    root_top_k=max(1, min(2, int(budget.root_top_k))),
                    belief_samples=1,
                    timeout_ms=budget.timeout_ms,
                )
                followup_indices = _prioritized_root_indices(
                    context=current_context,
                    policy_log_probs=followup_policy,
                    budget=followup_budget,
                )
                best_value: float | None = None
                best_score: float | None = None
                for candidate_index in followup_indices[: max(1, min(2, len(followup_indices)))]:
                    if time.perf_counter() > deadline:
                        raise _SearchTimeout("combo_followup")
                    if runtime_stats is not None:
                        runtime_stats["combo_evals"] = float(runtime_stats.get("combo_evals", 0.0) + 1.0)
                    selected_action_code = int(current_context.legal_low_level_codes[candidate_index])
                    child_value = self._evaluate_internal_self_candidate(
                        current_context=current_context,
                        current_sampled_state_json=current_sampled_state_json,
                        current_public_belief=current_public_belief,
                        current_history=current_history,
                        root_player=root_player,
                        budget=followup_budget,
                        depth_remaining=1,
                        model=model,
                        encoder=encoder,
                        device=device,
                        opponent_deck_name=opponent_deck_name,
                        search_state_value_fn=search_state_value_fn,
                        deadline=deadline,
                        updater=updater,
                        selected_action_code=selected_action_code,
                        env_config=env_config,
                        matchup=matchup,
                        current_tracker=current_tracker,
                        policy_cache=policy_cache,
                        leaf_value_cache=leaf_value_cache,
                        runtime_stats=runtime_stats,
                    )
                    if child_value is None:
                        continue
                    score = float(followup_policy[candidate_index]) + float(child_value)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_value = float(child_value)
                return best_value

            def _evaluate_internal_self_candidate(
            """
        ),
        "_rollout_value and self followup methods",
    )

    text = replace_once(
        text,
        r"    def _evaluate_internal_self_candidate\(\n.*?\n    def _advance_until_action\(",
        indent_method(
            """
            def _evaluate_internal_self_candidate(
                self,
                *,
                current_context: DecisionContext,
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
                runtime_stats: dict[str, float] | None = None,
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
                    next_depth = max(0, depth_remaining - 1)
                    if next_depth <= 0:
                        cheap_public_belief = _singleton_public_belief_after_step(
                            pre_public_belief=current_public_belief,
                            post_state=step.post_state,
                            next_context=next_context,
                            tracker=tracker,
                            sampled_self_hypothesis=sampled_self_hypothesis,
                            sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                        )
                        return _leaf_value(
                            public_belief=cheap_public_belief,
                            current_context=next_context,
                            budget=budget,
                            card_vocabulary=self.card_vocabulary,
                            device=device,
                            search_state_value_fn=search_state_value_fn,
                            cache=leaf_value_cache,
                        )
                    update_result = updater.update_after_step(
                        pre_public_belief=current_public_belief,
                        selected_action_code=int(child_context.legal_low_level_codes[matched]),
                        post_state=step.post_state,
                        next_context=next_context,
                        tracker=tracker,
                        sampled_self_hypothesis=sampled_self_hypothesis,
                        sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                        deadline=deadline,
                        runtime_stats=runtime_stats,
                    )
                    return self._rollout_value(
                        env=child_env,
                        current_context=next_context,
                        current_sampled_state_json=next_context.full_state_json,
                        current_public_belief=update_result.updated_public_belief,
                        current_history=child_history,
                        root_player=root_player,
                        budget=budget,
                        depth_remaining=next_depth,
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
                        env_config=env_config,
                        matchup=matchup,
                        runtime_stats=runtime_stats,
                    )
                finally:
                    child_env.close()

            def _advance_until_action(
            """
        ),
        "_evaluate_internal_self_candidate method",
    )

    text = replace_once(
        text,
        r"    def _advance_until_action\(\n.*?\n\ndef _semantic_key_dict",
        indent_method(
            """
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
                runtime_stats: dict[str, float] | None = None,
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
                        raise _SearchTimeout("non_action_advance")
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
                        runtime_stats=runtime_stats,
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
            """
        ) + "\n\ndef _semantic_key_dict",
        "_advance_until_action method",
    )

    helpers = textwrap.dedent(
        """
        def _runtime_log_enabled() -> bool:
            value = os.environ.get("SEPOT_RUNTIME_LOG", "")
            return str(value).strip().lower() not in {"", "0", "false", "off", "no"}


        def _runtime_log_search_outcome(
            *,
            outcome: str,
            decision: SearchDecision,
            runtime_ms: float,
            runtime_stats: dict[str, float],
        ) -> None:
            if not _runtime_log_enabled():
                return
            stats_blob = " ".join(
                f"{key}={value:.2f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in sorted(runtime_stats.items())
            )
            print(
                "[sepot-runtime]",
                f"outcome={outcome}",
                f"reason={decision.fallback_reason or 'ok'}",
                f"root_k={int(decision.root_candidate_count)}",
                f"belief={int(decision.belief_sample_count)}",
                f"phase_depth={int(decision.phase_depth)}",
                f"latency_ms={float(runtime_ms):.2f}",
                stats_blob,
            )


        def _try_reset_branch_env(
            *,
            env_config: EnvConfig,
            matchup: Matchup,
            state_json: str | None,
        ) -> tuple[GitcgDecisionEnv | None, DecisionContext | None]:
            if not state_json:
                return None, None
            env = GitcgDecisionEnv(
                env_config,
                matchup,
                enable_result_based_action_relabel=False,
            )
            try:
                context = env.reset(state_json=state_json)
                return env, context
            except Exception:
                env.close()
                return None, None


        def _observation_delta(
            *,
            pre_public_belief: PublicBeliefState,
            next_public_hand_count: float,
            next_public_dice_count: float,
            next_revealed: Sequence[int],
        ) -> _ObservationDelta:
            hand_delta = int(round(float(next_public_hand_count) - float(pre_public_belief.opponent_public_hand_count)))
            dice_delta = int(round(float(next_public_dice_count) - float(pre_public_belief.opponent_public_dice_count)))
            pre_revealed = {int(value) for value in pre_public_belief.revealed_opponent_card_definition_ids}
            next_revealed_set = {int(value) for value in next_revealed}
            revealed_delta_count = len(next_revealed_set.difference(pre_revealed))
            has_public_delta = bool(hand_delta or dice_delta or revealed_delta_count)
            requires_exact = bool(
                revealed_delta_count >= 2
                or abs(hand_delta) > 1
                or abs(dice_delta) > 2
            )
            return _ObservationDelta(
                hand_delta=hand_delta,
                dice_delta=dice_delta,
                revealed_delta_count=revealed_delta_count,
                has_public_delta=has_public_delta,
                requires_exact=requires_exact,
            )


        def _approximate_opponent_hypothesis_after_public_delta(
            *,
            sampled_opponent_hypothesis: ExplicitRangeHypothesis,
            next_public_hand_count: float,
            next_public_dice_count: float,
            next_revealed: Sequence[int],
        ) -> ExplicitRangeHypothesis:
            revealed = {int(value) for value in next_revealed}
            hand_cards = [int(value) for value in sampled_opponent_hypothesis.hand_definition_ids if int(value) not in revealed]
            deck_cards = [int(value) for value in sampled_opponent_hypothesis.deck_definition_ids if int(value) not in revealed]
            target_hand_count = max(0, int(round(float(next_public_hand_count))))
            if len(hand_cards) > target_hand_count:
                deck_cards = hand_cards[target_hand_count:] + deck_cards
                hand_cards = hand_cards[:target_hand_count]
            elif len(hand_cards) < target_hand_count:
                needed = target_hand_count - len(hand_cards)
                hand_cards.extend(deck_cards[:needed])
                deck_cards = deck_cards[needed:]
            hidden_dice = [int(value) for value in sampled_opponent_hypothesis.hidden_dice]
            target_dice_count = max(0, int(round(float(next_public_dice_count))))
            if len(hidden_dice) > target_dice_count:
                hidden_dice = hidden_dice[:target_dice_count]
            elif len(hidden_dice) < target_dice_count:
                filler = tuple(int(value) for value in sampled_opponent_hypothesis.hidden_dice) or (1,)
                while len(hidden_dice) < target_dice_count:
                    hidden_dice.append(int(filler[len(hidden_dice) % len(filler)]))
            return ExplicitRangeHypothesis(
                hand_definition_ids=tuple(int(value) for value in hand_cards),
                deck_definition_ids=tuple(int(value) for value in deck_cards),
                hidden_dice=tuple(int(value) for value in hidden_dice),
                weight=1.0,
            )


        def _should_expand_self_followup_action(context: DecisionContext) -> bool:
            if context.request_type != DecisionType.ACTION:
                return False
            legal_kinds = _legal_option_kinds(context)
            productive_count = sum(1 for kind in legal_kinds if kind in _PRODUCTIVE_ACTION_KINDS)
            has_end = OptionKind.ACTION_DECLARE_END in legal_kinds
            if _tactical_trigger_reason(context) is not None:
                return True
            if has_end and productive_count >= 1:
                return True
            return productive_count >= 2
        """
    ).strip()

    text = text.replace("\n\ndef _semantic_key_dict", "\n\n" + helpers + "\n\ndef _semantic_key_dict", 1)

    target.write_text(text, encoding="utf-8")
    print(f"patched {target}")
    print(f"backup at {backup}")


if __name__ == "__main__":
    main()
