from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import torch

from gitcg_world_model.ppo_checkpoint import (
    load_ppo_deployment_checkpoint,
    save_ppo_deployment_checkpoint,
)
from gitcg_world_model.ppo_features import TokenObservationEncoder
from gitcg_world_model.ppo_model import PpoModelConfig, PpoTransformerPolicy


class PpoCheckpointTests(unittest.TestCase):
    def test_deployment_checkpoint_excludes_oracle_parameters(self):
        encoder = TokenObservationEncoder()
        model = PpoTransformerPolicy(
            PpoModelConfig(
                token_dim=encoder.token_dim,
                option_dim=encoder.option_dim,
                high_action_dim=encoder.high_action_dim,
                privileged_dim=encoder.privileged_state_dim,
                belief_histogram_dim=encoder.belief_histogram_dim,
                belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
                max_tokens=encoder.max_tokens,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "deployment.pt"
            save_ppo_deployment_checkpoint(path, model, encoder)
            payload = torch.load(path, map_location="cpu")
        actor_state = dict(payload["actor_state_dict"])
        self.assertTrue(actor_state)
        self.assertFalse(any(key.startswith("oracle_") for key in actor_state))

    def test_model_init_does_not_emit_nested_tensor_warning(self):
        encoder = TokenObservationEncoder()
        config = PpoModelConfig(
            token_dim=encoder.token_dim,
            option_dim=encoder.option_dim,
            high_action_dim=encoder.high_action_dim,
            privileged_dim=encoder.privileged_state_dim,
            belief_histogram_dim=encoder.belief_histogram_dim,
            belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
            max_tokens=encoder.max_tokens,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            PpoTransformerPolicy(config)
        nested_tensor_warnings = [
            warning
            for warning in captured
            if "nested_tensor" in str(warning.message)
        ]
        self.assertEqual(nested_tensor_warnings, [])

    def test_load_old_option_dim_checkpoint_and_resave_with_current_dim(self):
        encoder = TokenObservationEncoder()
        old_option_dim = encoder.option_dim - 3
        old_model = PpoTransformerPolicy(
            PpoModelConfig(
                token_dim=encoder.token_dim,
                option_dim=old_option_dim,
                high_action_dim=encoder.high_action_dim,
                privileged_dim=encoder.privileged_state_dim,
                belief_histogram_dim=encoder.belief_histogram_dim,
                belief_deck_histogram_dim=encoder.belief_remaining_deck_histogram_dim,
                max_tokens=encoder.max_tokens,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            old_path = Path(temp_dir) / "old_deployment.pt"
            payload = {
                "kind": "ppo_deployment",
                "model_config": {
                    **old_model.config.__dict__,
                },
                "encoder_config": encoder.to_dict(),
                "actor_state_dict": {
                    key: value
                    for key, value in old_model.state_dict().items()
                    if not key.startswith("oracle_")
                },
                "metadata": {},
            }
            torch.save(payload, old_path)

            loaded_model, loaded_encoder, _ = load_ppo_deployment_checkpoint(old_path)
            self.assertEqual(loaded_model.config.option_dim, encoder.option_dim)
            self.assertEqual(loaded_encoder.option_dim, encoder.option_dim)
            self.assertEqual(
                loaded_model.option_projection[0].weight.shape[1],
                encoder.option_dim,
            )

            new_path = Path(temp_dir) / "new_deployment.pt"
            save_ppo_deployment_checkpoint(new_path, loaded_model, loaded_encoder)
            new_payload = torch.load(new_path, map_location="cpu")
            self.assertEqual(new_payload["model_config"]["option_dim"], encoder.option_dim)


if __name__ == "__main__":
    unittest.main()
