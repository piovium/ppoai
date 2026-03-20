from pathlib import Path

TARGET = Path("research/world_model/src/gitcg_world_model/sepot_search.py")


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"start marker not found: {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"end marker not found: {end_marker!r}")
    return text[:start] + replacement + "\n\n" + text[end:]


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"missing target file: {TARGET}")

    text = TARGET.read_text(encoding="utf-8")
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_effect_first")
    backup.write_text(text, encoding="utf-8")

    new_phase_budget = '''def phase_budget_for_context(context: DecisionContext, *, config: SePotSearchConfig) -> SePotPhaseBudget:
    round_number = int(context.player_view.round_number) if context.player_view is not None else 0
    if round_number <= 3:
        return SePotPhaseBudget(depth=1, root_top_k=2, belief_samples=1, timeout_ms=config.opening_timeout_ms)
    if round_number <= 8:
        return SePotPhaseBudget(depth=2, root_top_k=3, belief_samples=2, timeout_ms=config.midgame_timeout_ms)
    return SePotPhaseBudget(depth=2, root_top_k=4, belief_samples=3, timeout_ms=config.endgame_timeout_ms)'''

    new_should_trigger = '''def should_trigger_search(
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
    if legal_count <= 1:
        return False
    if round_number <= 3:
        return legal_count >= 5 and has_end and productive_non_end
    if round_number <= 8:
        return legal_count >= 4 or (has_end and productive_non_end)
    return legal_count >= 3'''

    helper_block = '''def _top_two_logprob_margin(policy_log_probs: Sequence[float]) -> float:
    if len(policy_log_probs) < 2:
        return 999.0
    ordered = sorted((float(value) for value in policy_log_probs), reverse=True)
    return float(ordered[0] - ordered[1])


def _should_search_for_policy(
    *,
    context: DecisionContext,
    policy_log_probs: Sequence[float],
) -> bool:
    if len(policy_log_probs) < 2:
        return False
    round_number = int(context.player_view.round_number) if context.player_view is not None else 0
    legal_count = len(context.legal_low_level_codes)
    margin = _top_two_logprob_margin(policy_log_probs)
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
        if has_end and productive_non_end:
            return margin < 1.35 or legal_count >= 8
        return legal_count >= 7 and margin < 0.85

    if round_number <= 8:
        if has_end and productive_non_end:
            return margin < 1.50 or legal_count >= 7
        return legal_count >= 5 and margin < 0.95

    if has_end and productive_non_end:
        return margin < 1.75 or legal_count >= 6
    return legal_count >= 4 and margin < 1.15


def _select_root_candidate_indices(
    *,
    context: DecisionContext,
    policy_log_probs: Sequence[float],
    budget: SePotPhaseBudget,
) -> tuple[int, ...]:
    ranked = sorted(range(len(policy_log_probs)), key=lambda index: float(policy_log_probs[index]), reverse=True)
    if not ranked:
        return ()

    margin = _top_two_logprob_margin(policy_log_probs)
    if margin >= 1.50:
        limit = 1
    elif margin >= 0.80:
        limit = min(2, budget.root_top_k)
    elif margin >= 0.35:
        limit = min(3, budget.root_top_k)
    else:
        limit = min(4, budget.root_top_k)

    end_indices = [
        index
        for index in ranked
        if low_level_kind_for_code(int(context.legal_low_level_codes[index])) == OptionKind.ACTION_DECLARE_END
    ]
    non_end_indices = [
        index
        for index in ranked
        if low_level_kind_for_code(int(context.legal_low_level_codes[index])) != OptionKind.ACTION_DECLARE_END
    ]

    chosen: list[int] = []

    if non_end_indices:
        chosen.append(non_end_indices[0])
    else:
        chosen.append(ranked[0])

    if end_indices:
        best_non_end_logp = float(policy_log_probs[chosen[0]])
        best_end_index = end_indices[0]
        best_end_logp = float(policy_log_probs[best_end_index])
        include_end = (
            not non_end_indices
            or abs(best_non_end_logp - best_end_logp) <= 0.60
            or (len(context.legal_low_level_codes) >= 8 and best_end_logp >= best_non_end_logp - 0.90)
        )
        if include_end and best_end_index not in chosen and len(chosen) < limit:
            chosen.append(best_end_index)

    for index in ranked:
        if index in chosen:
            continue
        kind = low_level_kind_for_code(int(context.legal_low_level_codes[index]))
        if kind == OptionKind.ACTION_DECLARE_END:
            continue
        chosen.append(index)
        if len(chosen) >= limit:
            break

    return tuple(chosen[: max(1, limit)])'''

    choose_action_start = "    def choose_action(\n"
    choose_action_end = "\n    def _evaluate_root_candidate(\n"
    new_choose_action = '''    def choose_action(
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
        if not _should_search_for_policy(context=context, policy_log_probs=policy_log_probs):
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
        root_indices = _select_root_candidate_indices(
            context=context,
            policy_log_probs=policy_log_probs,
            budget=budget,
        )
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
        )'''

    text = replace_block(text, "def phase_budget_for_context(", "\n\ndef should_trigger_search(", new_phase_budget)
    text = replace_block(text, "def should_trigger_search(", "\n\ndef build_public_belief_state(", new_should_trigger + "\n\n" + helper_block)
    text = replace_block(text, choose_action_start, choose_action_end, new_choose_action)

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET}")
    print(f"backup  {backup}")


if __name__ == "__main__":
    main()
