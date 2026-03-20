from __future__ import annotations

from dataclasses import dataclass

from .action_hierarchy import high_level_spec_for_code, legal_low_level_specs
from .semantic_priors import option_kind_priority_map, string_sequence
from .schema import ActionChoice, DecisionContext, DecisionType, OptionKind


@dataclass
class ScriptedBootstrapAgent:
    def choose_action(self, context: DecisionContext) -> ActionChoice:
        available_specs = legal_low_level_specs(context)
        available_codes = (
            tuple(int(code) for code in context.legal_low_level_codes)
            if context.legal_low_level_codes
            else tuple(int(spec.action_code) for spec in available_specs)
        )
        if not available_codes:
            raise RuntimeError("no legal options available")
        if context.request_type != DecisionType.ACTION:
            return self._choose_non_action(context, available_codes=available_codes, available_specs=available_specs)
        available = tuple(zip(available_codes, available_specs, strict=False))
        ranked = sorted(available, key=lambda pair: self._action_rank(pair[1], tuple(spec for _, spec in available)))
        return ActionChoice(action_code=int(ranked[0][0]))

    def _choose_non_action(
        self,
        context: DecisionContext,
        *,
        available_codes: tuple[int, ...],
        available_specs,
    ) -> ActionChoice:
        available = tuple(zip(available_codes, available_specs, strict=False))
        if context.request_type == DecisionType.REROLL_DICE:
            ranked = sorted(
                available,
                key=lambda pair: self._reroll_rank(
                    pair[0],
                    pair[1],
                    context=context,
                ),
            )
            return ActionChoice(action_code=int(ranked[0][0]))
        if context.request_type == DecisionType.SWITCH_HANDS:
            for action_code, option in available:
                if option.switch_hand_slot_mask == 0:
                    return ActionChoice(action_code=int(action_code))
        return ActionChoice(action_code=int(available_codes[0]))

    def _action_rank(self, option, available) -> tuple[int, int, int, str]:
        has_non_end = any(candidate.kind != OptionKind.ACTION_DECLARE_END for candidate in available)
        if option.kind == OptionKind.ACTION_DECLARE_END and has_non_end:
            return (99, 99, 99, option.label)
        priority = option_kind_priority_map(("agents", "bootstrap", "action_priority"))[option.kind]
        is_fast = 0 if option.metadata.get("is_fast") else 1
        used_dice = len(option.used_dice)
        return (priority, is_fast, used_dice, option.label)

    def _reroll_rank(
        self,
        action_code: int,
        option,
        *,
        context: DecisionContext,
    ) -> tuple[int, int, str]:
        category = self._high_level_key_for_code(context, action_code)
        order = string_sequence(("agents", "bootstrap", "reroll_high_priority"))
        try:
            category_rank = int(order.index(category))
        except ValueError:
            category_rank = len(order)
        rerolled_count = int(option.reroll_dice_mask).bit_count()
        return (
            category_rank,
            -rerolled_count,
            str(option.label),
        )

    def _high_level_key_for_code(self, context: DecisionContext, action_code: int) -> str:
        for high_code, low_codes in context.high_to_low_map:
            if int(action_code) in (int(code) for code in low_codes):
                return str(high_level_spec_for_code(int(high_code)).key)
        return ""
