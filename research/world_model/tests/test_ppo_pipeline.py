from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from gitcg_world_model.decks import SMALL_DECK_MATCHUPS, SMALL_DECK_POOL
from gitcg_world_model.ppo_pipeline import (
    _EpisodeJob,
    _LoopState,
    _TaggedAgent,
    _execute_episode_jobs,
    _recommended_rollout_workers,
    _resolve_initial_checkpoint,
    _round_base_checkpoint,
    _safe_candidate_epochs,
    _shortlist_candidate_epochs,
    run_ppo_managed_loop,
)
from gitcg_world_model.schema import EnvConfig


class PpoPipelineTests(unittest.TestCase):
    def test_execute_episode_jobs_uses_parallel_path_when_workers_gt_one(self):
        config = EnvConfig(deck_pool=SMALL_DECK_POOL)
        jobs = tuple(
            _EpisodeJob(
                job_index=index,
                matchup=SMALL_DECK_MATCHUPS[0],
                seed=index,
                agent0_factory=lambda _seed: _TaggedAgent(object(), trainable=True, tag="a"),
                agent1_factory=lambda _seed: _TaggedAgent(object(), trainable=False, tag="b"),
                metadata={},
            )
            for index in range(3)
        )

        with patch(
            "gitcg_world_model.ppo_pipeline._run_episode_job",
            side_effect=lambda **kwargs: type(
                "Result",
                (),
                {"job": kwargs["job"], "episode": {"seed": kwargs["job"].seed}},
            )(),
        ) as runner:
            episodes = _execute_episode_jobs(
                config=config,
                jobs=jobs,
                max_decisions=4,
                workers=3,
            )

        self.assertEqual(runner.call_count, 3)
        self.assertEqual(tuple(item["seed"] for item in episodes), (0, 1, 2))

    def test_safe_candidate_epochs_filter_by_kl(self):
        payload = {
            "candidate_epochs": [1, 2, 3, 4],
            "history": [
                {"total_loss": 0.5, "value_loss": 0.5, "approx_kl": 0.10},
                {"total_loss": 0.4, "value_loss": 0.3, "approx_kl": 0.31},
                {"total_loss": 0.3, "value_loss": 0.2, "approx_kl": 0.30},
                {"total_loss": 0.2, "value_loss": 0.4, "approx_kl": 0.29},
            ],
        }
        self.assertEqual(_safe_candidate_epochs(payload, max_safe_kl=0.30), [1, 3, 4])

    def test_shortlist_candidate_epochs_prefers_latest_total_and_value(self):
        payload = {
            "candidate_epochs": [1, 2, 3, 4],
            "history": [
                {"total_loss": 0.10, "value_loss": 0.50, "approx_kl": 0.10},
                {"total_loss": 0.05, "value_loss": 0.40, "approx_kl": 0.35},
                {"total_loss": 0.20, "value_loss": 0.05, "approx_kl": 0.20},
                {"total_loss": 0.30, "value_loss": 0.10, "approx_kl": 0.25},
            ],
        }

        shortlist, safe_epochs, fallback_reason = _shortlist_candidate_epochs(
            payload,
            max_safe_kl=0.30,
            shortlist_size=3,
        )

        self.assertEqual(safe_epochs, [1, 3, 4])
        self.assertEqual(shortlist, [4, 1, 3])
        self.assertIsNone(fallback_reason)

    def test_shortlist_candidate_epochs_falls_back_to_lowest_kl(self):
        payload = {
            "candidate_epochs": [1, 2, 3],
            "history": [
                {"total_loss": 0.10, "value_loss": 0.50, "approx_kl": 0.80},
                {"total_loss": 0.05, "value_loss": 0.40, "approx_kl": 0.45},
                {"total_loss": 0.20, "value_loss": 0.05, "approx_kl": 0.60},
            ],
        }

        shortlist, safe_epochs, fallback_reason = _shortlist_candidate_epochs(
            payload,
            max_safe_kl=0.30,
            shortlist_size=3,
        )

        self.assertEqual(safe_epochs, [])
        self.assertEqual(shortlist, [2])
        self.assertEqual(fallback_reason, "no_safe_epoch")

    def test_resolve_initial_checkpoint_prefers_deployment_sibling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_checkpoint = Path(temp_dir) / "last.pt"
            deployment_checkpoint = Path(temp_dir) / "deployment.pt"
            train_checkpoint.write_bytes(b"train")
            deployment_checkpoint.write_bytes(b"deploy")
            empty_state = _LoopState(working_checkpoint=None, completed_round=0)
            state_with_training = _LoopState(
                working_checkpoint=train_checkpoint,
                completed_round=1,
            )

            self.assertEqual(
                _resolve_initial_checkpoint(train_checkpoint, empty_state),
                deployment_checkpoint,
            )
            self.assertEqual(
                _resolve_initial_checkpoint(None, state_with_training),
                deployment_checkpoint,
            )

    def test_round_base_checkpoint_prefers_deployment_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            round_dir = Path(temp_dir) / "round_0001"
            train_dir = round_dir / "train"
            train_dir.mkdir(parents=True)
            (train_dir / "last.pt").write_bytes(b"train")
            deployment_checkpoint = train_dir / "deployment.pt"
            deployment_checkpoint.write_bytes(b"deploy")
            (train_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "best_epoch": 1,
                        "best_score": 0.0,
                        "device": "cpu",
                        "sample_count": 1,
                        "objective": "ppo",
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _round_base_checkpoint(round_dir=round_dir, fallback=None),
                deployment_checkpoint,
            )

    def test_recommended_rollout_workers_caps_large_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "deployment.pt"
            torch.save(
                {
                    "model_config": {
                        "d_model": 512,
                        "num_layers": 8,
                        "recurrent_hidden_dim": 512,
                    }
                },
                checkpoint_path,
            )
            self.assertEqual(_recommended_rollout_workers(8, checkpoint_path), 4)
            self.assertEqual(_recommended_rollout_workers(2, checkpoint_path), 2)

    def test_managed_loop_smoke_round_uses_p2sro_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            artifacts = run_ppo_managed_loop(
                workspace=workspace,
                rounds=1,
                bootstrap_episodes_per_matchup=1,
                self_play_episodes_per_matchup=1,
                max_decisions=8,
                batch_size=2,
                epochs=1,
                bootstrap_epochs=1,
                device="cpu",
                workers=1,
                resume=False,
                enable_status_print=False,
                retain_round_directories=None,
                matchups=(SMALL_DECK_MATCHUPS[0],),
            )

            self.assertTrue(Path(artifacts.working_checkpoint).exists())
            self.assertTrue(Path(artifacts.deployment_checkpoint).exists())
            self.assertEqual(
                Path(artifacts.deployment_checkpoint),
                workspace / "meta_deployment.json",
            )
            self.assertTrue((workspace / "p2sro_state.json").exists())
            self.assertTrue((workspace / "event_loop_status.json").exists())
            self.assertTrue((workspace / "event_loop_history.jsonl").exists())
            self.assertTrue((workspace / "round_0001" / "self_play.jsonl").exists())
            self.assertTrue((workspace / "round_0001" / "br_self_play.jsonl").exists())
            self.assertTrue((workspace / "round_0001" / "train" / "selection_summary.json").exists())
            self.assertTrue((workspace / "round_0001" / "br_train" / "selection_summary.json").exists())

            state_payload = json.loads((workspace / "event_loop_state.json").read_text(encoding="utf-8"))
            analysis_payload = json.loads((workspace / "round_0001" / "analysis.json").read_text(encoding="utf-8"))
            main_selection = json.loads(
                (workspace / "round_0001" / "train" / "selection_summary.json").read_text(encoding="utf-8")
            )
            br_selection = json.loads(
                (workspace / "round_0001" / "br_train" / "selection_summary.json").read_text(encoding="utf-8")
            )
            p2sro_payload = json.loads((workspace / "p2sro_state.json").read_text(encoding="utf-8"))

            self.assertEqual(analysis_payload["algorithm"], "p2sro")
            self.assertIn("frozen_population", analysis_payload)
            self.assertIn("meta_probabilities", analysis_payload)
            self.assertIn("active_main_slot", analysis_payload)
            self.assertIn("active_br_slot", analysis_payload)
            self.assertEqual(main_selection["selection_method"], "expected_payoff_vs_meta")
            self.assertEqual(br_selection["selection_method"], "expected_payoff_vs_meta")
            self.assertIn("p2sro_state_path", state_payload)
            self.assertIn("meta_strategy_path", state_payload)
            self.assertTrue(p2sro_payload["frozen_population"])
            self.assertEqual(Path(state_payload["working_checkpoint"]), Path(artifacts.working_checkpoint))

    def test_p2sro_round_does_not_write_legacy_gate_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            artifacts = run_ppo_managed_loop(
                workspace=workspace,
                rounds=2,
                bootstrap_episodes_per_matchup=1,
                self_play_episodes_per_matchup=1,
                max_decisions=8,
                batch_size=2,
                epochs=1,
                bootstrap_epochs=1,
                device="cpu",
                workers=1,
                resume=False,
                enable_status_print=False,
                retain_round_directories=None,
                matchups=(SMALL_DECK_MATCHUPS[0],),
            )

            round_one_gate_artifacts = sorted(path.name for path in (workspace / "round_0001").glob("*gate*.json"))
            round_two_gate_artifacts = sorted(path.name for path in (workspace / "round_0002").glob("*gate*.json"))
            self.assertEqual(round_one_gate_artifacts, [])
            self.assertEqual(round_two_gate_artifacts, [])
            self.assertEqual(len(artifacts.rounds), 2)


if __name__ == "__main__":
    unittest.main()
