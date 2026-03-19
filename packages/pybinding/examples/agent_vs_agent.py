from __future__ import annotations

import json

from gitcg import (
    ActionRequest,
    ActionResponse,
    ActionValidity,
    ChooseActiveRequest,
    ChooseActiveResponse,
    CreateParam,
    Deck,
    DiceRequirementType,
    DiceType,
    Game,
    Player,
    RerollDiceRequest,
    RerollDiceResponse,
    SelectCardRequest,
    SelectCardResponse,
    State,
    SwitchHandsRequest,
    SwitchHandsResponse,
    low_level,
)

DECK0 = Deck(
    characters=[1411, 1510, 2103],
    cards=[
        214111,
        214111,
        215101,
        311503,
        312004,
        312004,
        312025,
        312025,
        312029,
        312029,
        321002,
        321011,
        321016,
        321016,
        322002,
        322009,
        322009,
        330008,
        332002,
        332002,
        332004,
        332004,
        332005,
        332005,
        332006,
        332006,
        332018,
        332025,
        333004,
        333004,
    ],
)

DECK1 = Deck(
    characters=[1609, 2203, 1608],
    cards=[
        216091,
        216091,
        222031,
        312004,
        312004,
        312021,
        312021,
        312025,
        312025,
        321002,
        321002,
        321011,
        322025,
        323004,
        323004,
        330005,
        331601,
        331601,
        332002,
        332003,
        332003,
        332004,
        332004,
        332005,
        332005,
        332006,
        332025,
        332025,
        333003,
        333003,
    ],
)


class DeterministicAgent(Player):
    def __init__(self, who: int):
        self.who = who
        self.notification_count = 0
        self.omni_dice_count = 0
        self.last_visible_dice: list[int] = []
        self.last_error: str | None = None

    def on_notify(self, notification):
        self.notification_count += 1
        dice = list(notification.state.player[self.who].dice)
        self.last_visible_dice = dice
        self.omni_dice_count = sum(
            1 for die in dice if die == DiceType.DICE_TYPE_OMNI
        )

    def on_io_error(self, error_msg: str):
        self.last_error = error_msg

    def on_choose_active(self, request: ChooseActiveRequest) -> ChooseActiveResponse:
        return ChooseActiveResponse(active_character_id=request.candidate_ids[0])

    def on_reroll_dice(self, request: RerollDiceRequest) -> RerollDiceResponse:
        return RerollDiceResponse(dice_to_reroll=[])

    def on_select_card(self, request: SelectCardRequest) -> SelectCardResponse:
        return SelectCardResponse(
            selected_definition_id=request.candidate_definition_ids[0]
        )

    def on_switch_hands(self, request: SwitchHandsRequest) -> SwitchHandsResponse:
        return SwitchHandsResponse(removed_hand_ids=[])

    def _action_priority(self, action) -> tuple[int, int]:
        if action.HasField("use_skill"):
            return (0, 0)
        if action.HasField("play_card"):
            return (1, 0)
        if action.HasField("switch_active"):
            return (2, 0)
        if action.HasField("declare_end"):
            return (3, 0)
        if action.HasField("elemental_tuning"):
            return (4, 0)
        return (5, 0)

    def _used_dice_if_affordable(self, action) -> list[DiceType] | None:
        required_count = 0
        for req in action.required_cost:
            if req.type in (
                DiceRequirementType.DICE_REQUIREMENT_TYPE_ENERGY,
                DiceRequirementType.DICE_REQUIREMENT_TYPE_LEGEND,
            ):
                return None
            required_count += req.count
        if required_count > self.omni_dice_count:
            return None
        return [DiceType.DICE_TYPE_OMNI] * required_count

    def on_action(self, request: ActionRequest) -> ActionResponse:
        declare_end_index: int | None = None
        candidates = sorted(
            enumerate(request.action),
            key=lambda item: (self._action_priority(item[1]), item[0]),
        )
        for index, action in candidates:
            if action.validity != ActionValidity.ACTION_VALIDITY_VALID:
                continue
            if action.HasField("elemental_tuning"):
                continue
            if action.HasField("declare_end"):
                declare_end_index = index
                continue
            used_dice = self._used_dice_if_affordable(action)
            if used_dice is None:
                continue
            return ActionResponse(
                chosen_action_index=index,
                used_dice=used_dice,
            )
        if declare_end_index is not None:
            return ActionResponse(chosen_action_index=declare_end_index)
        return ActionResponse(chosen_action_index=0)


def _build_create_param(seed: int) -> CreateParam:
    create_param = CreateParam(deck0=DECK0, deck1=DECK1)
    create_param.set_attr(low_level.ATTR_CREATEPARAM_NO_SHUFFLE_0, 1)
    create_param.set_attr(low_level.ATTR_CREATEPARAM_NO_SHUFFLE_1, 1)
    create_param.set_attr(low_level.ATTR_STATE_CONFIG_RANDOM_SEED, seed)
    return create_param


def _summarize_characters(state: State, who: int) -> list[dict[str, int]]:
    return [
        {
            "id": entity.id(),
            "definition_id": entity.definition_id(),
            "health": entity.variable("health"),
            "max_health": entity.variable("maxHealth"),
            "energy": entity.variable("energy"),
            "max_energy": entity.variable("maxEnergy"),
        }
        for entity in state.query(who, "my characters include defeated")
    ]


def run_demo_game(seed: int = 7) -> dict[str, object]:
    create_param = _build_create_param(seed)
    initial_state = State(create_param=create_param)

    # Binding-level card inspection returns ids. Rich metadata belongs to packages/data.
    opening_pile_definition_ids = [
        entity.definition_id()
        for entity in initial_state.query(0, "my pile cards limit 5")
    ]

    game = Game(state=initial_state)
    player0 = DeterministicAgent(0)
    player1 = DeterministicAgent(1)
    game.set_attr(low_level.ATTR_PLAYER_ALWAYS_OMNI_0, 1)
    game.set_attr(low_level.ATTR_PLAYER_ALWAYS_OMNI_1, 1)
    game.set_player(0, player0)
    game.set_player(1, player1)

    game.start()
    while game.is_running():
        game.step()

    final_state = game.state()
    return {
        "seed": seed,
        "status": game.status().name,
        "winner": game.winner(),
        "round_number": game.round_number(),
        "current_turn": game.current_turn(),
        "opening_pile_definition_ids": opening_pile_definition_ids,
        "player0_notifications": player0.notification_count,
        "player1_notifications": player1.notification_count,
        "player0_last_visible_dice": player0.last_visible_dice,
        "player1_last_visible_dice": player1.last_visible_dice,
        "player0_last_error": player0.last_error,
        "player1_last_error": player1.last_error,
        "player0_characters": _summarize_characters(final_state, 0),
        "player1_characters": _summarize_characters(final_state, 1),
        "final_state_json": final_state.json(),
    }


def main():
    print(json.dumps(run_demo_game(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
