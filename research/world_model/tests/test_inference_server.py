from __future__ import annotations

import unittest

from gitcg_world_model.inference_server import GpuInferenceServer


class _DummyModel:
    def __init__(self) -> None:
        self.devices: list[str] = []
        self.eval_called = 0

    def to(self, device):
        self.devices.append(str(device))
        return self

    def eval(self):
        self.eval_called += 1
        return self


class InferenceServerTests(unittest.TestCase):
    def test_register_model_reuses_existing_instance(self):
        server = GpuInferenceServer(device="cpu")
        try:
            first = _DummyModel()
            second = _DummyModel()

            registered = server.register_model("model-a", first)
            reused = server.register_model("model-a", second)

            self.assertIs(registered, first)
            self.assertIs(reused, first)
            self.assertIs(server.get_registered_model("model-a"), first)
            self.assertEqual(first.eval_called, 1)
            self.assertEqual(second.eval_called, 0)
        finally:
            server.close()

    def test_retain_models_evicts_stale_entries(self):
        server = GpuInferenceServer(device="cpu")
        try:
            keep = _DummyModel()
            drop = _DummyModel()
            server.register_model("keep", keep)
            server.register_model("drop", drop)

            server.retain_models({"keep"})

            self.assertIs(server.get_registered_model("keep"), keep)
            self.assertIsNone(server.get_registered_model("drop"))
            self.assertIn("cpu", drop.devices)
        finally:
            server.close()
