from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

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
    SwitchHandsRequest,
    SwitchHandsResponse,
)


DECK0 = Deck(
    [1411, 1510, 2103],
    [
        214111, 214111, 215101, 311503, 312004, 312004, 312025, 312025, 312029, 312029,
        321002, 321011, 321016, 321016, 322002, 322009, 322009, 330008, 332002, 332002,
        332004, 332004, 332005, 332005, 332006, 332006, 332018, 332025, 333004, 333004,
    ],
)

DECK1 = Deck(
    [1609, 2203, 1608],
    [
        312025, 321002, 321002, 321011, 322025, 323004, 323004, 330005, 331601, 331601,
        332002, 332003, 332003, 332004, 332004, 332005, 332005, 332006, 332025, 332025,
        333003, 333003,
    ],
)


class OmniPlayer(Player):
    def __init__(self, who: int):
        self.who = who
        self.omni_dice_count = 0

    def on_notify(self, notification):
        self.omni_dice_count = sum(
            1
            for die in notification.state.player[self.who].dice
            if die == DiceType.DICE_TYPE_OMNI
        )

    def on_io_error(self, error_msg: str):
        pass

    def on_choose_active(self, request: ChooseActiveRequest) -> ChooseActiveResponse:
        return ChooseActiveResponse(active_character_id=request.candidate_ids[0])

    def on_reroll_dice(self, request: RerollDiceRequest) -> RerollDiceResponse:
        return RerollDiceResponse(dice_to_reroll=[])

    def on_select_card(self, request: SelectCardRequest) -> SelectCardResponse:
        return SelectCardResponse(selected_definition_id=request.candidate_definition_ids[0])

    def on_switch_hands(self, request: SwitchHandsRequest) -> SwitchHandsResponse:
        return SwitchHandsResponse(removed_hand_ids=[])

    def on_action(self, request: ActionRequest) -> ActionResponse:
        for index, action in enumerate(request.action):
            if action.validity != ActionValidity.ACTION_VALIDITY_VALID:
                continue
            required_count = 0
            blocked = False
            for req in action.required_cost:
                if req.type in (
                    DiceRequirementType.DICE_REQUIREMENT_TYPE_ENERGY,
                    DiceRequirementType.DICE_REQUIREMENT_TYPE_LEGEND,
                ):
                    blocked = True
                    break
                required_count += req.count
            if blocked or required_count > self.omni_dice_count:
                continue
            return ActionResponse(
                chosen_action_index=index,
                used_dice=[DiceType.DICE_TYPE_OMNI] * required_count,
            )
        return ActionResponse(chosen_action_index=0)


def _run_threaded_game(seed: int) -> int | None:
    with CreateParam(deck0=DECK0, deck1=DECK1) as create_param:
        with Game(create_param=create_param) as game:
            game.set_player(0, OmniPlayer(0))
            game.set_player(1, OmniPlayer(1))
            game.start()
            while game.is_running():
                game.step()
            return game.winner()


class ThreadingTests(unittest.TestCase):
    def test_game_rejects_cross_thread_access(self):
        with CreateParam(deck0=DECK0, deck1=DECK1) as create_param:
            with Game(create_param=create_param) as game:
                result: list[str] = []

                def worker():
                    try:
                        game.status()
                    except RuntimeError as exc:
                        result.append(str(exc))

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
                self.assertTrue(result)
                self.assertIn("cannot be used across threads", result[0])

    def test_multiple_games_can_run_in_parallel_threads(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            winners = list(executor.map(_run_threaded_game, [1, 2, 3, 4]))
        self.assertEqual(len(winners), 4)
        for winner in winners:
            self.assertIn(winner, (0, 1, None))
