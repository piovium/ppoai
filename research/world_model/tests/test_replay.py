from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gitcg_world_model.replay import (
    append_episode_records,
    group_episode_records_by_matchup,
    iter_episode_records,
    iter_episode_records_from_paths,
    load_episode_records,
    load_episode_records_from_paths,
    resolve_replay_paths,
    sample_episode_records,
)
from gitcg_world_model.schema import DecisionType
from gitcg_world_model.testsupport import synthetic_episode_record


class ReplayTests(unittest.TestCase):
    def test_round_trip_episode_record(self):
        episode = synthetic_episode_record()
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_path = Path(temp_dir) / "replay.jsonl"
            append_episode_records(replay_path, [episode])
            loaded = load_episode_records(replay_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].matchup, episode.matchup)
        self.assertEqual(loaded[0].winner, episode.winner)
        self.assertEqual(loaded[0].steps[0].request_type, DecisionType.ACTION)
        self.assertEqual(loaded[0].steps[0].choice.action_code, episode.steps[0].choice.action_code)
        self.assertTrue(loaded[0].steps[0].legal_low_level_codes)

    def test_resolve_and_load_from_directory(self):
        episode = synthetic_episode_record()
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_dir = Path(temp_dir) / "shards"
            replay_dir.mkdir()
            append_episode_records(replay_dir / "a.jsonl", [episode])
            append_episode_records(replay_dir / "b.jsonl", [episode])
            resolved = resolve_replay_paths([replay_dir])
            loaded = load_episode_records_from_paths([replay_dir])
        self.assertEqual(len(resolved), 2)
        self.assertEqual(len(loaded), 2)

    def test_iterators_stream_episode_records(self):
        episode = synthetic_episode_record()
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_dir = Path(temp_dir) / "shards"
            replay_dir.mkdir()
            replay_path = replay_dir / "a.jsonl"
            append_episode_records(replay_path, [episode, episode])
            loaded_path = list(iter_episode_records(replay_path))
            loaded_dir = list(iter_episode_records_from_paths([replay_dir]))
        self.assertEqual(len(loaded_path), 2)
        self.assertEqual(len(loaded_dir), 2)
        self.assertEqual(loaded_path[0].matchup, episode.matchup)

    def test_resolve_replay_paths_can_preserve_duplicates(self):
        episode = synthetic_episode_record()
        with tempfile.TemporaryDirectory() as temp_dir:
            replay_path = Path(temp_dir) / "a.jsonl"
            append_episode_records(replay_path, [episode])
            resolved = resolve_replay_paths([replay_path, replay_path], dedupe=False)
        self.assertEqual(resolved, [replay_path, replay_path])

    def test_group_and_sample_by_matchup(self):
        episodes = [synthetic_episode_record(), synthetic_episode_record(), synthetic_episode_record()]
        grouped = group_episode_records_by_matchup(episodes)
        self.assertEqual(list(grouped), ["sample_a__vs__sample_b"])
        sampled = sample_episode_records(grouped["sample_a__vs__sample_b"], 2, seed=11)
        self.assertEqual(len(sampled), 2)


if __name__ == "__main__":
    unittest.main()
