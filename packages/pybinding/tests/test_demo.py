import unittest

from examples.agent_vs_agent import run_demo_game


class TestAgentVsAgentDemo(unittest.TestCase):
    def test_demo_reaches_terminal_state(self):
        result = run_demo_game()

        self.assertEqual(result["status"], "FINISHED")
        self.assertIn(result["winner"], [0, 1])
        self.assertGreaterEqual(result["round_number"], 1)
        self.assertEqual(len(result["opening_pile_definition_ids"]), 5)
        self.assertEqual(len(result["player0_characters"]), 3)
        self.assertEqual(len(result["player1_characters"]), 3)
        self.assertIsNone(result["player0_last_error"])
        self.assertIsNone(result["player1_last_error"])
        self.assertTrue(str(result["final_state_json"]).startswith("{"))

    def test_demo_is_deterministic(self):
        first = run_demo_game(seed=7)
        second = run_demo_game(seed=7)

        self.assertEqual(first["winner"], second["winner"])
        self.assertEqual(first["round_number"], second["round_number"])
        self.assertEqual(
            first["opening_pile_definition_ids"],
            second["opening_pile_definition_ids"],
        )
        self.assertEqual(first["final_state_json"], second["final_state_json"])
