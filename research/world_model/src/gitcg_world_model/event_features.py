from __future__ import annotations

from dataclasses import dataclass

from .features import (
    FeatureEncoder,
    _COST_BUCKETS,
    _PHASES,
    _TARGET_AGGREGATE_FEATURE_DIM,
    _TARGET_SLOT_FEATURE_DIM,
    _TARGET_SLOTS,
)
from .schema import (
    DecisionContext,
    DecisionType,
    LowLevelActionSpec,
    OptionKind,
    StateSnapshot,
)


@dataclass(frozen=True)
class StructuredActionTarget:
    full_features: tuple[float, ...]
    kind_index: int
    metadata_summary: tuple[float, ...]
    cost_features: tuple[float, ...]
    target_features: tuple[float, ...]


class EventFeatureEncoder:
    def __init__(
        self,
        card_vocabulary: tuple[int, ...] | None = None,
        character_vocabulary: tuple[int, ...] | None = None,
    ) -> None:
        self.base = FeatureEncoder(
            card_vocabulary=card_vocabulary,
            character_vocabulary=character_vocabulary,
        )
        self.request_types = tuple(DecisionType)
        self.option_kinds = tuple(OptionKind)
        self.kind_dim = len(self.option_kinds)
        self.cost_dim = len(_COST_BUCKETS)
        self.target_dim = _TARGET_SLOTS * _TARGET_SLOT_FEATURE_DIM + _TARGET_AGGREGATE_FEATURE_DIM
        self.metadata_dim = self.base.option_dim - self.kind_dim - self.cost_dim - self.target_dim
        self.public_state_dim = len(_PHASES) + 1 + 2 + 3 + 2 * self.base._player_dim
        self.observation_dim = self.public_state_dim + len(self.request_types) + 2
        self.public_observation_dim = self.observation_dim
        self.privileged_observation_dim = self.public_state_dim
        self.option_dim = self.base.option_dim
        self._kind_slice = slice(0, self.kind_dim)
        self._metadata_slice = slice(self.kind_dim, self.kind_dim + self.metadata_dim)
        self._cost_slice = slice(self._metadata_slice.stop, self._metadata_slice.stop + self.cost_dim)
        self._target_slice = slice(self._cost_slice.stop, self._cost_slice.stop + self.target_dim)
        self.zero_action = tuple(0.0 for _ in range(self.option_dim))
        self.zero_observation = tuple(0.0 for _ in range(self.observation_dim))
        self.zero_privileged_observation = tuple(0.0 for _ in range(self.privileged_observation_dim))
        self.zero_structured_action = StructuredActionTarget(
            full_features=self.zero_action,
            kind_index=0,
            metadata_summary=tuple(0.0 for _ in range(self.metadata_dim)),
            cost_features=tuple(0.0 for _ in range(self.cost_dim)),
            target_features=tuple(0.0 for _ in range(self.target_dim)),
        )

    @property
    def card_vocabulary(self) -> tuple[int, ...]:
        return self.base.card_vocabulary

    @property
    def character_vocabulary(self) -> tuple[int, ...]:
        return self.base.character_vocabulary

    def request_index(self, request_type: DecisionType) -> int:
        return self.request_types.index(request_type)

    def encode_public_observation(self, context: DecisionContext) -> tuple[float, ...]:
        if context.player_view is None:
            raise ValueError("player_view is required for public observation encoding")
        features = self.encode_public_state(context.player_view)
        features.extend(
            1.0 if request_type == context.request_type else 0.0
            for request_type in self.request_types
        )
        features.extend(
            [
                1.0 if context.acting_player == 0 else 0.0,
                1.0 if context.acting_player == 1 else 0.0,
            ]
        )
        return tuple(features)

    def encode_public_state(self, state: StateSnapshot) -> list[float]:
        return list(self.base.encode_state(state))

    def encode_privileged_state(self, state: StateSnapshot) -> tuple[float, ...]:
        return tuple(self.base.encode_state(state))

    def encode_public_low_level_spec(
        self,
        spec: LowLevelActionSpec,
        context: DecisionContext,
    ) -> tuple[float, ...]:
        if context.player_view is None:
            raise ValueError("player_view is required for public low-level encoding")
        return tuple(
            self.base.encode_low_level_spec(
                spec,
                state=context.player_view,
                acting_player=context.acting_player,
            )
        )

    def encode_action_descriptor(self, spec: LowLevelActionSpec) -> tuple[float, ...]:
        return tuple(self.base.encode_low_level_spec(spec, state=None, acting_player=None))

    def encode_structured_action_target(
        self,
        spec: LowLevelActionSpec,
        context: DecisionContext,
    ) -> StructuredActionTarget:
        features = self.encode_public_low_level_spec(spec, context)
        kind_block = features[self._kind_slice]
        kind_index = max(range(len(kind_block)), key=lambda index: kind_block[index])
        return StructuredActionTarget(
            full_features=features,
            kind_index=kind_index,
            metadata_summary=features[self._metadata_slice],
            cost_features=features[self._cost_slice],
            target_features=features[self._target_slice],
        )
