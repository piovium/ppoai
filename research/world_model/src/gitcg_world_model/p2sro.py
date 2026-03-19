from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


class _SupportsAgentLifecycle(Protocol):
    def choose_action(self, context): ...


@dataclass(frozen=True)
class P2SROConfig:
    active_slots: int = 2
    meta_iterations: int = 256
    meta_tolerance: float = 1e-7
    promotion_margin_main: float = 0.05
    promotion_margin_br: float = 0.05


@dataclass(frozen=True)
class PolicyEntry:
    policy_id: str
    checkpoint_path: str
    lineage: str
    status: str
    born_round: int
    parent_policy_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "checkpoint_path": self.checkpoint_path,
            "lineage": self.lineage,
            "status": self.status,
            "born_round": self.born_round,
            "parent_policy_id": self.parent_policy_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyEntry":
        return cls(
            policy_id=str(payload["policy_id"]),
            checkpoint_path=str(payload["checkpoint_path"]),
            lineage=str(payload["lineage"]),
            status=str(payload["status"]),
            born_round=int(payload["born_round"]),
            parent_policy_id=(
                str(payload["parent_policy_id"])
                if payload.get("parent_policy_id") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ActiveSlotState:
    slot_name: str
    lineage: str
    seed_policy_id: str | None
    seed_checkpoint_path: str
    parent_policy_id: str | None = None
    selected_epoch: int | None = None
    selected_checkpoint_path: str | None = None
    expected_payoff_vs_meta: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_name": self.slot_name,
            "lineage": self.lineage,
            "seed_policy_id": self.seed_policy_id,
            "seed_checkpoint_path": self.seed_checkpoint_path,
            "parent_policy_id": self.parent_policy_id,
            "selected_epoch": self.selected_epoch,
            "selected_checkpoint_path": self.selected_checkpoint_path,
            "expected_payoff_vs_meta": self.expected_payoff_vs_meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActiveSlotState":
        return cls(
            slot_name=str(payload["slot_name"]),
            lineage=str(payload["lineage"]),
            seed_policy_id=(
                str(payload["seed_policy_id"])
                if payload.get("seed_policy_id") is not None
                else None
            ),
            seed_checkpoint_path=str(payload["seed_checkpoint_path"]),
            parent_policy_id=(
                str(payload["parent_policy_id"])
                if payload.get("parent_policy_id") is not None
                else None
            ),
            selected_epoch=(
                int(payload["selected_epoch"])
                if payload.get("selected_epoch") is not None
                else None
            ),
            selected_checkpoint_path=(
                str(payload["selected_checkpoint_path"])
                if payload.get("selected_checkpoint_path") is not None
                else None
            ),
            expected_payoff_vs_meta=float(payload.get("expected_payoff_vs_meta", 0.0)),
        )


@dataclass(frozen=True)
class PayoffResult:
    episodes: int
    wins: int
    losses: int
    draws: int
    payoff: float
    matchup_results: tuple[dict[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "payoff": self.payoff,
            "matchup_results": [dict(item) for item in self.matchup_results],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PayoffResult":
        return cls(
            episodes=int(payload.get("episodes", 0)),
            wins=int(payload.get("wins", 0)),
            losses=int(payload.get("losses", 0)),
            draws=int(payload.get("draws", 0)),
            payoff=float(payload.get("payoff", 0.0)),
            matchup_results=tuple(dict(item) for item in payload.get("matchup_results", [])),
        )


@dataclass
class PayoffMatrix:
    entries: dict[str, dict[str, PayoffResult]] = field(default_factory=dict)

    def policy_ids(self) -> tuple[str, ...]:
        identifiers: set[str] = set(self.entries.keys())
        for opponents in self.entries.values():
            identifiers.update(opponents.keys())
        return tuple(sorted(identifiers))

    def get(self, row_policy_id: str, col_policy_id: str) -> PayoffResult | None:
        return self.entries.get(row_policy_id, {}).get(col_policy_id)

    def set_pair(
        self,
        *,
        row_policy_id: str,
        col_policy_id: str,
        row_result: PayoffResult,
        col_result: PayoffResult,
    ) -> None:
        self.entries.setdefault(row_policy_id, {})[col_policy_id] = row_result
        self.entries.setdefault(col_policy_id, {})[row_policy_id] = col_result

    def expected_payoff(
        self,
        policy_id: str,
        meta_probabilities: dict[str, float],
    ) -> float:
        total = 0.0
        for opponent_id, probability in meta_probabilities.items():
            result = self.get(policy_id, opponent_id)
            if result is None:
                continue
            total += float(probability) * float(result.payoff)
        return total

    def to_dict(self) -> dict[str, object]:
        return {
            "entries": {
                row_policy_id: {
                    col_policy_id: result.to_dict()
                    for col_policy_id, result in sorted(opponents.items())
                }
                for row_policy_id, opponents in sorted(self.entries.items())
            }
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PayoffMatrix":
        entries: dict[str, dict[str, PayoffResult]] = {}
        for row_policy_id, opponents_payload in dict(payload.get("entries", {})).items():
            entries[str(row_policy_id)] = {
                str(col_policy_id): PayoffResult.from_dict(dict(result_payload))
                for col_policy_id, result_payload in dict(opponents_payload).items()
            }
        return cls(entries=entries)


@dataclass(frozen=True)
class MetaStrategySnapshot:
    probabilities: dict[str, float]
    expected_utilities: dict[str, float]
    support_set: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "probabilities": dict(self.probabilities),
            "expected_utilities": dict(self.expected_utilities),
            "support_set": list(self.support_set),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaStrategySnapshot":
        return cls(
            probabilities={
                str(key): float(value)
                for key, value in dict(payload.get("probabilities", {})).items()
            },
            expected_utilities={
                str(key): float(value)
                for key, value in dict(payload.get("expected_utilities", {})).items()
            },
            support_set=tuple(str(item) for item in payload.get("support_set", [])),
        )


@dataclass
class P2SROManager:
    state_path: Path
    config: P2SROConfig = field(default_factory=P2SROConfig)
    frozen_population: dict[str, PolicyEntry] = field(default_factory=dict)
    active_main_slot: ActiveSlotState | None = None
    active_br_slot: ActiveSlotState | None = None
    payoff_matrix: PayoffMatrix = field(default_factory=PayoffMatrix)
    meta_strategy: MetaStrategySnapshot = field(
        default_factory=lambda: MetaStrategySnapshot(probabilities={}, expected_utilities={}, support_set=())
    )
    next_policy_index: int = 1

    @classmethod
    def load(cls, state_path: str | Path, *, config: P2SROConfig | None = None) -> "P2SROManager":
        target = Path(state_path)
        if not target.exists():
            return cls(state_path=target, config=config or P2SROConfig())
        payload = json.loads(target.read_text(encoding="utf-8"))
        manager = cls(
            state_path=target,
            config=config or P2SROConfig(),
            frozen_population={
                policy_id: PolicyEntry.from_dict(dict(entry_payload))
                for policy_id, entry_payload in dict(payload.get("frozen_population", {})).items()
            },
            active_main_slot=(
                ActiveSlotState.from_dict(dict(payload["active_main_slot"]))
                if payload.get("active_main_slot") is not None
                else None
            ),
            active_br_slot=(
                ActiveSlotState.from_dict(dict(payload["active_br_slot"]))
                if payload.get("active_br_slot") is not None
                else None
            ),
            payoff_matrix=PayoffMatrix.from_dict(dict(payload.get("payoff_matrix", {}))),
            meta_strategy=MetaStrategySnapshot.from_dict(dict(payload.get("meta_strategy", {}))),
            next_policy_index=int(payload.get("next_policy_index", 1)),
        )
        return manager

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "frozen_population": {
                policy_id: entry.to_dict()
                for policy_id, entry in sorted(self.frozen_population.items())
            },
            "active_main_slot": self.active_main_slot.to_dict() if self.active_main_slot is not None else None,
            "active_br_slot": self.active_br_slot.to_dict() if self.active_br_slot is not None else None,
            "payoff_matrix": self.payoff_matrix.to_dict(),
            "meta_strategy": self.meta_strategy.to_dict(),
            "next_policy_index": self.next_policy_index,
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def initialize_seed_population(self, *, seed_checkpoint_path: str, round_index: int) -> PolicyEntry:
        if self.frozen_population:
            return self.latest_frozen_policy("main") or next(iter(self.frozen_population.values()))
        seed_entry = PolicyEntry(
            policy_id=self._next_policy_id("main"),
            checkpoint_path=str(seed_checkpoint_path),
            lineage="main",
            status="frozen",
            born_round=max(0, int(round_index) - 1),
            parent_policy_id=None,
        )
        self.frozen_population[seed_entry.policy_id] = seed_entry
        self.active_main_slot = ActiveSlotState(
            slot_name="active_main_slot",
            lineage="main",
            seed_policy_id=seed_entry.policy_id,
            seed_checkpoint_path=seed_entry.checkpoint_path,
            parent_policy_id=seed_entry.policy_id,
        )
        self.active_br_slot = ActiveSlotState(
            slot_name="active_br_slot",
            lineage="best_response",
            seed_policy_id=seed_entry.policy_id,
            seed_checkpoint_path=seed_entry.checkpoint_path,
            parent_policy_id=seed_entry.policy_id,
        )
        self.recompute_meta_strategy()
        self.save()
        return seed_entry

    def set_active_slot_seed(
        self,
        *,
        slot_name: str,
        lineage: str,
        checkpoint_path: str,
        seed_policy_id: str | None,
        parent_policy_id: str | None,
    ) -> ActiveSlotState:
        slot = ActiveSlotState(
            slot_name=slot_name,
            lineage=lineage,
            seed_policy_id=seed_policy_id,
            seed_checkpoint_path=str(checkpoint_path),
            parent_policy_id=parent_policy_id,
        )
        if slot_name == "active_main_slot":
            self.active_main_slot = slot
        elif slot_name == "active_br_slot":
            self.active_br_slot = slot
        else:  # pragma: no cover - defensive
            raise KeyError(f"unknown slot name: {slot_name}")
        return slot

    def update_active_slot_result(
        self,
        *,
        slot_name: str,
        selected_epoch: int,
        selected_checkpoint_path: str,
        expected_payoff_vs_meta: float,
    ) -> ActiveSlotState:
        slot = self.active_main_slot if slot_name == "active_main_slot" else self.active_br_slot
        if slot is None:
            raise RuntimeError(f"slot not initialized: {slot_name}")
        updated = replace(
            slot,
            selected_epoch=int(selected_epoch),
            selected_checkpoint_path=str(selected_checkpoint_path),
            expected_payoff_vs_meta=float(expected_payoff_vs_meta),
        )
        if slot_name == "active_main_slot":
            self.active_main_slot = updated
        else:
            self.active_br_slot = updated
        return updated

    def freeze_active_slot(
        self,
        *,
        slot_name: str,
        round_index: int,
    ) -> PolicyEntry:
        slot = self.active_main_slot if slot_name == "active_main_slot" else self.active_br_slot
        if slot is None or slot.selected_checkpoint_path is None:
            raise RuntimeError(f"slot has no selected checkpoint: {slot_name}")
        entry = PolicyEntry(
            policy_id=self._next_policy_id("main" if slot.lineage == "main" else "br"),
            checkpoint_path=slot.selected_checkpoint_path,
            lineage=slot.lineage,
            status="frozen",
            born_round=int(round_index),
            parent_policy_id=slot.parent_policy_id,
        )
        self.frozen_population[entry.policy_id] = entry
        return entry

    def latest_frozen_policy(self, lineage: str) -> PolicyEntry | None:
        matches = [entry for entry in self.frozen_population.values() if entry.lineage == lineage]
        if not matches:
            return None
        matches.sort(key=lambda entry: (entry.born_round, entry.policy_id))
        return matches[-1]

    def frozen_entries(self) -> tuple[PolicyEntry, ...]:
        return tuple(sorted(self.frozen_population.values(), key=lambda entry: entry.policy_id))

    def active_reference(self, lineage: str) -> PolicyEntry | None:
        reference = self.latest_frozen_policy(lineage)
        if reference is not None:
            return reference
        frozen = self.frozen_entries()
        return frozen[-1] if frozen else None

    def should_promote(
        self,
        *,
        lineage: str,
        candidate_payoff: float,
    ) -> bool:
        reference = self.active_reference(lineage)
        if reference is None:
            return True
        reference_payoff = self.payoff_matrix.expected_payoff(
            reference.policy_id,
            self.meta_strategy.probabilities,
        )
        margin = (
            self.config.promotion_margin_main
            if lineage == "main"
            else self.config.promotion_margin_br
        )
        return float(candidate_payoff) >= float(reference_payoff + margin)

    def recompute_meta_strategy(self) -> MetaStrategySnapshot:
        policies = [entry.policy_id for entry in self.frozen_entries()]
        if not policies:
            snapshot = MetaStrategySnapshot(probabilities={}, expected_utilities={}, support_set=())
            self.meta_strategy = snapshot
            return snapshot
        matrix = [[0.0 for _ in policies] for _ in policies]
        for row_index, row_policy_id in enumerate(policies):
            for col_index, col_policy_id in enumerate(policies):
                if row_policy_id == col_policy_id:
                    continue
                result = self.payoff_matrix.get(row_policy_id, col_policy_id)
                matrix[row_index][col_index] = float(result.payoff) if result is not None else 0.0
        probabilities, expected_utilities = solve_replicator_dynamics(
            matrix,
            iterations=self.config.meta_iterations,
            tolerance=self.config.meta_tolerance,
        )
        probability_map = {
            policy_id: float(probabilities[index])
            for index, policy_id in enumerate(policies)
        }
        utility_map = {
            policy_id: float(expected_utilities[index])
            for index, policy_id in enumerate(policies)
        }
        support = tuple(
            policy_id
            for policy_id, probability in probability_map.items()
            if probability > 1e-6
        )
        snapshot = MetaStrategySnapshot(
            probabilities=probability_map,
            expected_utilities=utility_map,
            support_set=support,
        )
        self.meta_strategy = snapshot
        return snapshot

    def sample_policy(self, *, seed: int) -> PolicyEntry:
        frozen = self.frozen_entries()
        if not frozen:
            raise RuntimeError("cannot sample from empty frozen population")
        probabilities = self.meta_strategy.probabilities
        weights = [max(0.0, float(probabilities.get(entry.policy_id, 0.0))) for entry in frozen]
        if sum(weights) <= 0.0:
            weights = [1.0 for _ in frozen]
        rng = random.Random(seed)
        return rng.choices(frozen, weights=weights, k=1)[0]

    def _next_policy_id(self, prefix: str) -> str:
        identifier = f"{prefix}_{self.next_policy_index:04d}"
        self.next_policy_index += 1
        return identifier


@dataclass
class MetaMixtureAgent:
    seed: int
    weighted_builders: tuple[tuple[float, str, Callable[[int], _SupportsAgentLifecycle]], ...]

    def __post_init__(self) -> None:
        if not self.weighted_builders:
            raise ValueError("weighted_builders must not be empty")
        rng = random.Random(self.seed)
        weights = [max(0.0, float(weight)) for weight, _policy_id, _builder in self.weighted_builders]
        if sum(weights) <= 0.0:
            weights = [1.0 for _ in self.weighted_builders]
        selected_index = rng.choices(range(len(self.weighted_builders)), weights=weights, k=1)[0]
        _weight, self.selected_policy_id, builder = self.weighted_builders[selected_index]
        self.inner = builder(self.seed)

    def choose_action(self, context):
        return self.inner.choose_action(context)

    def observe_step(self, step) -> None:
        if hasattr(self.inner, "observe_step"):
            getattr(self.inner, "observe_step")(step)

    def reset_history(self) -> None:
        if hasattr(self.inner, "reset_history"):
            getattr(self.inner, "reset_history")()

    def pop_last_decision_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "meta_selected_policy_id": self.selected_policy_id,
        }
        if hasattr(self.inner, "pop_last_decision_metadata"):
            payload.update(dict(getattr(self.inner, "pop_last_decision_metadata")() or {}))
        return payload


def solve_replicator_dynamics(
    payoff_matrix: Sequence[Sequence[float]],
    *,
    iterations: int,
    tolerance: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    size = len(payoff_matrix)
    if size == 0:
        return (), ()
    if size == 1:
        return (1.0,), (float(payoff_matrix[0][0]) if payoff_matrix and payoff_matrix[0] else 0.0,)
    strategy = [1.0 / size for _ in range(size)]
    for _ in range(max(1, iterations)):
        utilities = [
            sum(float(payoff_matrix[row_index][col_index]) * strategy[col_index] for col_index in range(size))
            for row_index in range(size)
        ]
        minimum_utility = min(utilities)
        shifted = [utility - minimum_utility + 1e-6 for utility in utilities]
        updated = [max(1e-12, strategy[index] * shifted[index]) for index in range(size)]
        total = sum(updated)
        if total <= 0.0 or not math.isfinite(total):
            updated = [1.0 / size for _ in range(size)]
        else:
            updated = [value / total for value in updated]
        delta = max(abs(updated[index] - strategy[index]) for index in range(size))
        strategy = updated
        if delta <= tolerance:
            break
    utilities = [
        sum(float(payoff_matrix[row_index][col_index]) * strategy[col_index] for col_index in range(size))
        for row_index in range(size)
    ]
    return tuple(float(value) for value in strategy), tuple(float(value) for value in utilities)
