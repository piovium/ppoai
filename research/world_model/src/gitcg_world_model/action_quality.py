from __future__ import annotations

from .action_hierarchy import (
    legal_low_level_specs,
    selected_low_level_spec,
)
from .semantic_priors import action_quality_config
from .schema import DecisionContext, LowLevelActionSpec, OptionKind, TrajectoryStep

_ACTION_QUALITY = action_quality_config()
_PRODUCTIVE_NON_END_KINDS = frozenset(OptionKind[name] for name in _ACTION_QUALITY["productive_non_end_kinds"])
_MAX_DICE_PER_TURN = float(_ACTION_QUALITY["max_dice_per_turn"])
_OMNI_DICE = int(_ACTION_QUALITY["omni_dice"])
_DIRECT_PRODUCTIVE_SEVERITY = float(_ACTION_QUALITY["direct_productive_severity"])
_TUNING_PRODUCTIVE_SEVERITY = float(_ACTION_QUALITY["tuning_productive_severity"])
_TUNING_SKILL_FOLLOWUP_MIN_REMAINING_DICE = int(_ACTION_QUALITY["tuning_skill_followup_min_remaining_dice"])
_TUNING_CARD_FOLLOWUP_MIN_REMAINING_DICE = int(_ACTION_QUALITY["tuning_card_followup_min_remaining_dice"])


def selected_option(step: TrajectoryStep) -> LowLevelActionSpec | None:
    return selected_low_level_spec(step)


def is_declare_end_option(option: LowLevelActionSpec | None) -> bool:
    return option is not None and option.kind == OptionKind.ACTION_DECLARE_END


def has_productive_non_end_option(options: tuple[LowLevelActionSpec, ...] | list[LowLevelActionSpec]) -> bool:
    return any(option.kind in _PRODUCTIVE_NON_END_KINDS for option in options)


def has_tuning_followup_potential(step: TrajectoryStep) -> bool:
    player = step.pre_state.players[step.acting_player]
    context = DecisionContext(
        acting_player=step.acting_player,
        request_type=step.request_type,
        step_index=0,
        full_state=step.pre_state,
        legal_low_level_codes=step.legal_low_level_codes,
        legal_low_level_mask=step.legal_low_level_mask,
        legal_high_level_codes=step.legal_high_level_codes,
        high_to_low_map=step.high_to_low_map,
    )
    tuning_options = [
        option for option in legal_low_level_specs(context) if option.kind == OptionKind.ACTION_ELEMENTAL_TUNING
    ]
    if not tuning_options:
        return False
    remaining_after_tuning = max(0, len(player.dice) - 1)
    has_active_character = any(character.is_active and not character.defeated for character in player.characters)
    # This is a weak prior, not a rules-perfect check. It only says that tuning
    # may be worthwhile if it can improve alignment and leaves enough resources
    # to plausibly cast a skill or a card afterwards.
    has_skill_followup = (
        has_active_character and remaining_after_tuning >= _TUNING_SKILL_FOLLOWUP_MIN_REMAINING_DICE
    )
    has_card_followup = bool(player.hand_cards) and remaining_after_tuning >= _TUNING_CARD_FOLLOWUP_MIN_REMAINING_DICE
    if not (has_skill_followup or has_card_followup):
        return False
    return any(_tuning_improves_alignment(player.dice, option) for option in tuning_options)


def premature_end_severity(step: TrajectoryStep) -> float:
    option = selected_option(step)
    if not is_declare_end_option(option):
        return 0.0
    context = DecisionContext(
        acting_player=step.acting_player,
        request_type=step.request_type,
        step_index=0,
        full_state=step.pre_state,
        legal_low_level_codes=step.legal_low_level_codes,
        legal_low_level_mask=step.legal_low_level_mask,
        legal_high_level_codes=step.legal_high_level_codes,
        high_to_low_map=step.high_to_low_map,
    )
    if has_productive_non_end_option(legal_low_level_specs(context)):
        return _DIRECT_PRODUCTIVE_SEVERITY
    if has_tuning_followup_potential(step):
        return _TUNING_PRODUCTIVE_SEVERITY
    return 0.0


def declare_end_with_productive_options(step: TrajectoryStep) -> bool:
    return premature_end_severity(step) > 0.0


def declare_end_waste_ratio(step: TrajectoryStep) -> float:
    if premature_end_severity(step) <= 0.0:
        return 0.0
    remaining_dice = len(step.pre_state.players[step.acting_player].dice)
    return max(0.0, min(1.0, float(remaining_dice) / _MAX_DICE_PER_TURN))


def _tuning_improves_alignment(dice: tuple[int, ...], option: LowLevelActionSpec) -> bool:
    target_dice = int(option.target_dice)
    if target_dice <= 0:
        return False
    return any(die not in (target_dice, _OMNI_DICE) for die in dice)
