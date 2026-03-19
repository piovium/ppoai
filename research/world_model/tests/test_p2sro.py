from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gitcg_world_model.p2sro import (
    MetaMixtureAgent,
    P2SROManager,
    PolicyEntry,
    solve_replicator_dynamics,
)


class _StubAgent:
    def __init__(self, label: str) -> None:
        self.label = label

    def choose_action(self, context):
        return context


class P2SROTests(unittest.TestCase):
    def test_replicator_dynamics_returns_stable_distribution(self):
        probabilities, utilities = solve_replicator_dynamics(
            (
                (0.0, 1.0),
                (-1.0, 0.0),
            ),
            iterations=256,
            tolerance=1e-7,
        )
        self.assertEqual(len(probabilities), 2)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertEqual(len(utilities), 2)

    def test_manager_maintains_two_active_slots_and_frozen_population(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "p2sro_state.json"
            checkpoint_path = Path(temp_dir) / "seed.pt"
            checkpoint_path.write_bytes(b"seed")
            manager = P2SROManager.load(state_path)
            seed_entry = manager.initialize_seed_population(
                seed_checkpoint_path=str(checkpoint_path),
                round_index=1,
            )
            manager.set_active_slot_seed(
                slot_name="active_main_slot",
                lineage="main",
                checkpoint_path=str(checkpoint_path),
                seed_policy_id=seed_entry.policy_id,
                parent_policy_id=seed_entry.policy_id,
            )
            manager.set_active_slot_seed(
                slot_name="active_br_slot",
                lineage="best_response",
                checkpoint_path=str(checkpoint_path),
                seed_policy_id=seed_entry.policy_id,
                parent_policy_id=seed_entry.policy_id,
            )
            self.assertTrue(manager.frozen_population)
            self.assertIsNotNone(manager.active_main_slot)
            self.assertIsNotNone(manager.active_br_slot)

    def test_meta_mixture_agent_samples_once_per_episode(self):
        agent = MetaMixtureAgent(
            seed=7,
            weighted_builders=(
                (0.0, "a", lambda seed: _StubAgent("a")),
                (1.0, "b", lambda seed: _StubAgent("b")),
            ),
        )
        self.assertEqual(agent.selected_policy_id, "b")
        self.assertEqual(agent.inner.label, "b")


if __name__ == "__main__":
    unittest.main()
