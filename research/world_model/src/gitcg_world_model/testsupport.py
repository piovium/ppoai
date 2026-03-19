from __future__ import annotations

from dataclasses import replace

from .action_hierarchy import build_action_legality_engine
from .schema import (
    ActionChoice,
    CharacterSnapshot,
    DecisionType,
    EpisodeRecord,
    LowLevelActionSpec,
    PublicSlotTarget,
    PlayerSnapshot,
    StateSnapshot,
    TrajectoryStep,
    OptionKind,
)


def synthetic_episode_record(
    *,
    matchup: str = "sample_a__vs__sample_b",
    winner: int | None = 0,
) -> EpisodeRecord:
    legal_specs = synthetic_legal_specs()
    pre_state = StateSnapshot(
        phase="action",
        round_number=1,
        current_turn=0,
        winner=None,
        players=(
            _player_snapshot(1, 1411),
            _player_snapshot(11, 1609),
        ),
    )
    post_state = StateSnapshot(
        phase="game_end",
        round_number=1,
        current_turn=0,
        winner=winner,
        players=pre_state.players,
    )
    step = TrajectoryStep(
        acting_player=0,
        request_type=DecisionType.ACTION,
        choice=ActionChoice(action_code=-1),
        pre_state=pre_state,
        post_state=post_state,
        reward=1.0 if winner == 0 else -1.0 if winner == 1 else 0.0,
        done=True,
        player_view=pre_state,
    )
    step = encode_step_actions(step, legal_specs=legal_specs)
    return EpisodeRecord(
        matchup=matchup,
        seed=7,
        winner=winner,
        steps=(step,),
        final_state=post_state,
    )


def synthetic_legal_specs() -> tuple[LowLevelActionSpec, ...]:
    return (
        LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.ACTION,
            kind=OptionKind.ACTION_USE_SKILL,
            label="use_skill",
            used_dice=(3, 3),
            subject_definition_id=1101,
            target_slots=(PublicSlotTarget(owner="opponent", zone="character", index=0),),
            metadata={
                "action_index": 0,
                "required_cost": ((3, 2),),
            },
        ),
        LowLevelActionSpec(
            action_code=-1,
            request_type=DecisionType.ACTION,
            kind=OptionKind.ACTION_DECLARE_END,
            label="declare_end",
            metadata={"action_index": 1},
        ),
    )


def encode_step_actions(
    step: TrajectoryStep,
    *,
    legal_specs: tuple[LowLevelActionSpec, ...] | None = None,
) -> TrajectoryStep:
    legal_specs = synthetic_legal_specs() if legal_specs is None else legal_specs
    visible_state = step.player_view or step.pre_state
    encoded, materialized_specs, low_to_high = build_action_legality_engine().encode_legal_actions(
        acting_player=step.acting_player,
        visible_state=visible_state,
        legal_specs=legal_specs,
    )
    action_code = int(step.choice.action_code)
    if action_code < 0 and encoded.legal_low_level_codes:
        action_code = int(encoded.legal_low_level_codes[0])
    chosen_high = low_to_high.get(action_code) if action_code >= 0 else None
    return replace(
        step,
        choice=ActionChoice(action_code=action_code),
        legal_low_level_codes=encoded.legal_low_level_codes,
        legal_low_level_mask=encoded.legal_low_level_mask,
        legal_high_level_codes=encoded.legal_high_level_codes,
        high_to_low_map=encoded.high_to_low_map,
        chosen_high_level_code=int(chosen_high) if chosen_high is not None else None,
    )


def materialize_legal_specs(
    *,
    acting_player: int,
    visible_state: StateSnapshot,
    legal_specs: tuple[LowLevelActionSpec, ...] | None = None,
):
    legal_specs = synthetic_legal_specs() if legal_specs is None else legal_specs
    return build_action_legality_engine().encode_legal_actions(
        acting_player=acting_player,
        visible_state=visible_state,
        legal_specs=legal_specs,
    )


def _player_snapshot(active_id: int, definition_id: int) -> PlayerSnapshot:
    return PlayerSnapshot(
        active_character_id=active_id,
        characters=(
            CharacterSnapshot(
                id=active_id,
                definition_id=definition_id,
                health=10,
                max_health=10,
                energy=0,
                max_energy=2,
                defeated=False,
                is_active=True,
                aura=0,
                entities=(),
            ),
        ),
        combat_statuses=(),
        summons=(),
        supports=(),
        hand_cards=(),
        pile_cards=(),
        dice=(3, 3, 8),
        declared_end=False,
        legend_used=False,
    )
