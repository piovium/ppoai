from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from gitcg_world_model.decks import SMALL_DECK_MATCHUPS
from gitcg_world_model.progress import RunStatus, atomic_write_json, load_run_status, reconstruct_bootstrap_status


def _expected_bootstrap_shard_names() -> tuple[str, ...]:
    baselines = ("baseline_random", "baseline_heuristic", "baseline_scripted")
    shard_names: list[str] = []
    for matchup in SMALL_DECK_MATCHUPS:
        for baseline in baselines:
            shard_names.append(f"{matchup.key}__{baseline}")
    return tuple(shard_names)


class ProgressTests(unittest.TestCase):
    def test_core_modules_import_cleanly(self):
        self.assertIsNotNone(import_module("gitcg_world_model.progress"))
        self.assertIsNotNone(import_module("gitcg_world_model.ppo_pipeline"))
        self.assertIsNotNone(import_module("gitcg_world_model.ppo_training"))

    def test_load_run_status_round_trip(self):
        status = RunStatus(
            phase="bootstrap_rollout",
            phase_percent=50.0,
            overall_percent=10.0,
            completed_units=48,
            total_units=96,
            current_task="halfway",
            worker_count=4,
            started_at="2026-03-09T00:00:00+00:00",
            updated_at="2026-03-09T00:10:00+00:00",
            artifacts={"bootstrap_dir": "tmp/bootstrap"},
            last_error=None,
            overall_completed_units=48,
            overall_total_units=480,
            details={"completed_shards": 6},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run_status.json"
            atomic_write_json(path, status.to_dict())
            loaded = load_run_status(path)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.phase, status.phase)
        self.assertEqual(loaded.completed_units, status.completed_units)
        self.assertEqual(loaded.details["completed_shards"], 6)

    def test_reconstruct_bootstrap_status_from_partial_directory(self):
        shard_names = _expected_bootstrap_shard_names()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            for shard_name in shard_names[:7]:
                (directory / f"{shard_name}.jsonl").write_text("{}", encoding="utf-8")
                (directory / f"{shard_name}.meta.json").write_text("{}", encoding="utf-8")
            status = reconstruct_bootstrap_status(
                bootstrap_dir=directory,
                expected_shard_names=shard_names,
                episodes_per_shard=8,
                worker_count=6,
            )
        self.assertIsNotNone(status)
        assert status is not None
        self.assertEqual(status.completed_units, 56)
        self.assertEqual(status.total_units, 96)
        self.assertAlmostEqual(status.phase_percent, 58.333333333333336)
        self.assertEqual(status.worker_count, 6)


if __name__ == "__main__":
    unittest.main()
