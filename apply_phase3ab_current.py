
from __future__ import annotations

from pathlib import Path

TARGET = Path("research/world_model/src/gitcg_world_model/sepot_search.py")
BACKUP = TARGET.with_suffix(TARGET.suffix + ".bak_phase3ab")

NEW_BLOCK = """
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
    ) -> _RangeUpdateResult:
        if time.perf_counter() > deadline:
            raise _SearchTimeout()
        public_state = mask_state_for_player(post_state, perspective_player=pre_public_belief.root_player)
        next_opponent_public_dice_count = float(tracker.opponent_public_dice_count_from_state(public_state))
        next_opponent_public_hand_count = float(tracker.opponent_public_hand_count_from_state(public_state))
        next_revealed = tuple(int(value) for value in tracker.revealed_opponent_card_definition_ids())
        observation_delta = self._observation_delta(
            pre_public_belief=pre_public_belief,
            next_public_hand_count=next_opponent_public_hand_count,
            next_public_dice_count=next_opponent_public_dice_count,
            next_revealed=next_revealed,
        )
        cache_key = (
            _public_belief_cache_key(pre_public_belief),
            semantic_action_key_for_code(selected_action_code),
            _hypothesis_cache_key(sampled_self_hypothesis),
            _hypothesis_cache_key(sampled_opponent_hypothesis),
            observation_delta,
        )
        updated_opponent_range = self._advance_cache.get(cache_key)
        if updated_opponent_range is None:
            if self._can_keep_sampled_opponent(
                pre_public_belief=pre_public_belief,
                next_public_hand_count=next_opponent_public_hand_count,
                next_public_dice_count=next_opponent_public_dice_count,
                next_revealed=next_revealed,
            ):
                updated_opponent_range = ExplicitRange((replace_weight(sampled_opponent_hypothesis, 1.0),))
            elif self._can_partial_update_opponent(
                pre_public_belief=pre_public_belief,
                next_public_hand_count=next_opponent_public_hand_count,
                next_public_dice_count=next_opponent_public_dice_count,
                next_revealed=next_revealed,
            ):
                partial_key = (
                    _public_belief_cache_key(pre_public_belief),
                    observation_delta,
                    tuple(
                        _hypothesis_cache_key(hypothesis) + (float(hypothesis.weight),)
                        for hypothesis in pre_public_belief.opponent_range.hypotheses
                    ),
                )
                updated_opponent_range = self._partial_update_cache.get(partial_key)
                if updated_opponent_range is None:
                    updated_opponent_range = self._partial_update_opponent_range(
                        pre_public_belief=pre_public_belief,
                        next_public_hand_count=next_opponent_public_hand_count,
                        next_public_dice_count=next_opponent_public_dice_count,
                        next_revealed=next_revealed,
                    )
                    self._partial_update_cache[partial_key] = updated_opponent_range
                    if len(self._partial_update_cache) > 512:
                        self._partial_update_cache.clear()
            else:
                updated_opponent_range = self._advance_opponent_range(
                    pre_public_belief=pre_public_belief,
                    selected_action_code=selected_action_code,
                    sampled_self_hypothesis=sampled_self_hypothesis,
                    sampled_opponent_hypothesis=sampled_opponent_hypothesis,
                    deadline=deadline,
                )
            if updated_opponent_range.count > 0:
                self._advance_cache[cache_key] = updated_opponent_range
                if len(self._advance_cache) > 512:
                    self._advance_cache.clear()
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
            opponent_public_dice_count=float(next_opponent_public_dice_count),
            opponent_public_hand_count=float(next_opponent_public_hand_count),
            self_range=ExplicitRange((sampled_self_hypothesis,)),
            opponent_range=updated_opponent_range,
            public_history=tracker.public_history(),
            revealed_opponent_card_definition_ids=next_revealed,
        )
        return _RangeUpdateResult(
            updated_public_belief=updated,
            sampled_self_hypothesis=sampled_self_hypothesis,
            sampled_opponent_hypothesis=sampled_opponent_hypothesis,
        )

    def _observation_delta(
        self,
        *,
        pre_public_belief: PublicBeliefState,
        next_public_hand_count: float,
        next_public_dice_count: float,
        next_revealed: Sequence[int],
    ) -> tuple[int, int, tuple[int, ...]]:
        hand_delta = int(round(float(next_public_hand_count) - float(pre_public_belief.opponent_public_hand_count)))
        dice_delta = int(round(float(next_public_dice_count) - float(pre_public_belief.opponent_public_dice_count)))
        previous_revealed = Counter(int(value) for value in pre_public_belief.revealed_opponent_card_definition_ids)
        newly_revealed: list[int] = []
        for value in (int(item) for item in next_revealed):
            if previous_revealed[value] > 0:
                previous_revealed[value] -= 1
            else:
                newly_revealed.append(value)
        return hand_delta, dice_delta, tuple(sorted(newly_revealed))

    def _can_keep_sampled_opponent(
        self,
        *,
        pre_public_belief: PublicBeliefState,
        next_public_hand_count: float,
        next_public_dice_count: float,
        next_revealed: Sequence[int],
    ) -> bool:
        return (
            abs(float(next_public_hand_count) - float(pre_public_belief.opponent_public_hand_count)) <= 1.0e-6
            and abs(float(next_public_dice_count) - float(pre_public_belief.opponent_public_dice_count)) <= 1.0e-6
            and tuple(int(value) for value in next_revealed)
            == tuple(int(value) for value in pre_public_belief.revealed_opponent_card_definition_ids)
        )

    def _can_partial_update_opponent(
        self,
        *,
        pre_public_belief: PublicBeliefState,
        next_public_hand_count: float,
        next_public_dice_count: float,
        next_revealed: Sequence[int],
    ) -> bool:
        previous_revealed = tuple(int(value) for value in pre_public_belief.revealed_opponent_card_definition_ids)
        current_revealed = tuple(int(value) for value in next_revealed)
        if current_revealed != previous_revealed:
            return False
        hand_delta = abs(int(round(float(next_public_hand_count) - float(pre_public_belief.opponent_public_hand_count))))
        dice_delta = abs(int(round(float(next_public_dice_count) - float(pre_public_belief.opponent_public_dice_count))))
        if hand_delta == 0 and dice_delta == 0:
            return False
        return hand_delta <= 2 and dice_delta <= 8

    def _partial_update_opponent_range(
        self,
        *,
        pre_public_belief: PublicBeliefState,
        next_public_hand_count: float,
        next_public_dice_count: float,
        next_revealed: Sequence[int],
    ) -> ExplicitRange:
        target_hand_count = max(0, int(round(next_public_hand_count)))
        target_dice_count = max(0, int(round(next_public_dice_count)))
        adjusted: list[ExplicitRangeHypothesis] = []
        total_weight = 0.0
        for sample_index, hypothesis in enumerate(pre_public_belief.opponent_range.hypotheses):
            adjusted_hypothesis = self._adjust_hypothesis_to_public_counts(
                hypothesis=hypothesis,
                target_hand_count=target_hand_count,
                target_dice_count=target_dice_count,
                next_revealed=next_revealed,
                sample_index=sample_index,
            )
            adjusted.append(adjusted_hypothesis)
            total_weight += float(adjusted_hypothesis.weight)
        if not adjusted or total_weight <= 0.0:
            return ExplicitRange(())
        normalized = tuple(
            replace_weight(hypothesis, float(hypothesis.weight) / total_weight)
            for hypothesis in adjusted
        )
        return ExplicitRange(normalized)

    def _adjust_hypothesis_to_public_counts(
        self,
        *,
        hypothesis: ExplicitRangeHypothesis,
        target_hand_count: int,
        target_dice_count: int,
        next_revealed: Sequence[int],
        sample_index: int,
    ) -> ExplicitRangeHypothesis:
        revealed_counter = Counter(int(value) for value in next_revealed)
        hidden_hand = self._remove_revealed_cards(
            cards=hypothesis.hand_definition_ids,
            revealed_counter=revealed_counter,
        )
        hidden_deck = self._remove_revealed_cards(
            cards=hypothesis.deck_definition_ids,
            revealed_counter=revealed_counter,
        )
        if len(hidden_hand) > target_hand_count:
            overflow = hidden_hand[target_hand_count:]
            hidden_hand = hidden_hand[:target_hand_count]
            hidden_deck = list(overflow) + hidden_deck
        elif len(hidden_hand) < target_hand_count:
            draw_count = min(len(hidden_deck), target_hand_count - len(hidden_hand))
            hidden_hand = hidden_hand + hidden_deck[:draw_count]
            hidden_deck = hidden_deck[draw_count:]
        hidden_dice = list(int(value) for value in hypothesis.hidden_dice[:target_dice_count])
        if len(hidden_dice) < target_dice_count:
            hidden_dice.extend(
                _sample_hidden_dice(
                    sample_index=sample_index,
                    count=target_dice_count - len(hidden_dice),
                )
            )
        return ExplicitRangeHypothesis(
            hand_definition_ids=tuple(int(value) for value in hidden_hand),
            deck_definition_ids=tuple(int(value) for value in hidden_deck),
            hidden_dice=tuple(int(value) for value in hidden_dice[:target_dice_count]),
            weight=float(hypothesis.weight),
        )

    def _remove_revealed_cards(
        self,
        *,
        cards: Sequence[int],
        revealed_counter: Counter[int],
    ) -> list[int]:
        remaining = Counter(revealed_counter)
        kept: list[int] = []
        for value in (int(item) for item in cards):
            if remaining[value] > 0:
                remaining[value] -= 1
            else:
                kept.append(value)
        return kept

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
            now = time.perf_counter()
            if now > deadline:
                if updated and total_weight > 0.0:
                    break
                raise _SearchTimeout()
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
                    raise _SearchTimeout()
                matched = match_low_level_code_index(context, selected_action_code)
                if matched is None:
                    continue
                if time.perf_counter() > deadline:
                    if updated and total_weight > 0.0:
                        break
                    raise _SearchTimeout()
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
""".strip("\n")


def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"not found: {TARGET}")
    text = TARGET.read_text(encoding="utf-8", newline="")
    line_ending = "\r\n" if "\r\n" in text else "\n"
    start_marker = "class PublicBeliefUpdater:"
    end_marker = "class SePotSearchController:"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("could not locate PublicBeliefUpdater block")
    replacement = NEW_BLOCK.replace("\n", line_ending) + line_ending * 2
    new_text = text[:start] + replacement + text[end:]
    if BACKUP.exists():
        BACKUP.unlink()
    BACKUP.write_text(text, encoding="utf-8", newline="")
    TARGET.write_text(new_text, encoding="utf-8", newline="")
    print(f"patched: {TARGET}")
    print(f"backup:  {BACKUP}")


if __name__ == "__main__":
    main()
