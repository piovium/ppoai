
import gitcg

SEED = 102011
EXPECTED_ERROR = "invalid int32: 116092.02"

DECK_SAMPLE_B = gitcg.Deck(
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

DECK_SAMPLE_A = gitcg.Deck(
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


class ProbePlayer(gitcg.Player):
    def __init__(self, who: int):
        self.who = who
        self.last_notification = None
        self.last_error = None
        self.action_turn = 0

    def on_notify(self, notification):
        self.last_notification = notification

    def on_io_error(self, error_msg: str):
        self.last_error = error_msg
        print(f"[io_error] player={self.who} error={error_msg}")

    def on_switch_hands(self, request):
        return gitcg.SwitchHandsResponse(removed_hand_ids=[])

    def on_choose_active(self, request):
        chosen = request.candidate_ids[0] if self.who == 0 else request.candidate_ids[-1]
        return gitcg.ChooseActiveResponse(active_character_id=chosen)

    def on_reroll_dice(self, request):
        if self.last_notification and self.last_notification.state.round_number == 1:
            dice = [2, 2, 5] if self.who == 0 else [2, 2, 3, 3, 4, 5, 8, 8]
            return gitcg.RerollDiceResponse(dice_to_reroll=dice)
        return gitcg.RerollDiceResponse(dice_to_reroll=[])

    def on_select_card(self, request):
        chosen = 116092 if 116092 in request.candidate_definition_ids else request.candidate_definition_ids[0]
        print(
            f"[select_card] player={self.who} "
            f"candidates={list(request.candidate_definition_ids)} chosen={chosen}"
        )
        return gitcg.SelectCardResponse(selected_definition_id=chosen) #这里的报错

    def _pick_skill(self, request, skill_definition_id: int):
        for index, action in enumerate(request.action):
            if not action.HasField("use_skill"):
                continue
            if action.use_skill.skill_definition_id != skill_definition_id:
                continue
            used_dice = list(action.auto_selected_dice)
            if not used_dice:
                fallback = {
                    16092: [8, 8, 6],
                    21031: [1, 2, 2],
                }
                used_dice = fallback[skill_definition_id]
            print(
                f"[action] player={self.who} "
                f"use_skill={skill_definition_id} index={index} used_dice={used_dice}"
            )
            return gitcg.ActionResponse(chosen_action_index=index, used_dice=used_dice)
        raise RuntimeError(f"player={self.who} missing skill {skill_definition_id}")

    def _pick_declare_end(self, request):
        for index, action in enumerate(request.action):
            if action.HasField("declare_end"):
                print(f"[action] player={self.who} declare_end index={index}")
                return gitcg.ActionResponse(chosen_action_index=index)
        raise RuntimeError(f"player={self.who} missing declare_end")

    def on_action(self, request):
        if self.who == 0:
            response = (
                self._pick_skill(request, 16092)
                if self.action_turn == 0
                else self._pick_declare_end(request)
            )
        else:
            response = (
                self._pick_skill(request, 21031)
                if self.action_turn == 0
                else self._pick_declare_end(request)
            )
        self.action_turn += 1
        return response


def main():
    create_param = gitcg.CreateParam(deck0=DECK_SAMPLE_B, deck1=DECK_SAMPLE_A)
    create_param.set_attr(gitcg.low_level.ATTR_STATE_CONFIG_RANDOM_SEED, SEED)
    state = gitcg.State(create_param=create_param)
    game = gitcg.Game(state=state)
    player0 = ProbePlayer(0)
    player1 = ProbePlayer(1)
    game.set_player(0, player0)
    game.set_player(1, player1)

    try:
        game.start()
        while game.is_running():
            game.step()

        result = {
            "status": game.status().name,
            "winner": game.winner(),
            "round_number": game.round_number(),
            "player0_last_error": player0.last_error,
            "player1_last_error": player1.last_error,
        }
        print(result)

        if player0.last_error != EXPECTED_ERROR and player1.last_error != EXPECTED_ERROR:
            raise SystemExit(f"did not reproduce expected error: {EXPECTED_ERROR!r}")
    finally:
        try:
            game._finalize()
        except Exception:
            pass
        try:
            state._finalize()
        except Exception:
            pass
        try:
            create_param._finalize()
        except Exception:
            pass


if __name__ == "__main__":
    main()
