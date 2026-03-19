from __future__ import annotations

from dataclasses import dataclass

from .action_hierarchy import legal_low_level_specs
from .semantic_priors import option_kind_priority_map
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
            for action_code, option in available:
                if option.reroll_dice_mask == 0:
                    return ActionChoice(action_code=int(action_code))
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
