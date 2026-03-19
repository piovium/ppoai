from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from .action_hierarchy import legal_low_level_specs
from .semantic_priors import option_kind_priority_map
from .schema import ActionChoice, DecisionContext, OptionKind


class PolicyAgent(Protocol):
    def choose_action(self, context: DecisionContext) -> ActionChoice:
        ...


@dataclass
class LegalRandomAgent:
    seed: int | None = None

    def __post_init__(self):
        self._random = random.Random(self.seed)

    def choose_action(self, context: DecisionContext) -> ActionChoice:
        available_specs = legal_low_level_specs(context)
        available_codes = (
            tuple(int(code) for code in context.legal_low_level_codes)
            if context.legal_low_level_codes
            else tuple(int(spec.action_code) for spec in available_specs)
        )
        if not available_codes:
            raise RuntimeError("no legal options available")
        action_code = int(self._random.choice(list(available_codes)))
        return ActionChoice(action_code=action_code)


@dataclass
class HeuristicAgent:
    def choose_action(self, context: DecisionContext) -> ActionChoice:
        available_specs = legal_low_level_specs(context)
        available_codes = (
            tuple(int(code) for code in context.legal_low_level_codes)
            if context.legal_low_level_codes
            else tuple(int(spec.action_code) for spec in available_specs)
        )
        if not available_codes:
            raise RuntimeError("no legal options available")
        ranked = sorted(
            zip(available_codes, available_specs, strict=False),
            key=lambda pair: self._rank(pair[1]),
        )
        return ActionChoice(action_code=int(ranked[0][0]))

    def _rank(self, option):
        priority = option_kind_priority_map(("agents", "heuristic", "priority"))[option.kind]
        return (
            priority,
            len(option.used_dice),
            option.metadata.get("action_index", -1),
            option.label,
        )


def build_baseline_agent(name: str, seed: int | None = None) -> PolicyAgent:
    normalized = name.strip().lower()
    if normalized == "heuristic":
        return HeuristicAgent()
    if normalized == "random":
        return LegalRandomAgent(seed=seed)
    raise KeyError(f"unknown baseline agent: {name}")
