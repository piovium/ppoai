from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from .ppo_model import PpoTransformerPolicy, masked_log_softmax


@dataclass(frozen=True)
class InferenceServerConfig:
    max_batch_size: int = 128
    max_wait_ms: int = 2


@dataclass(frozen=True)
class InferenceResult:
    action_index: int
    selected_logprob: float
    partial_value: float
    policy_entropy: float
    batch_size: int
    recurrent_state: tuple[float, ...] | None = None
    log_probs: tuple[float, ...] = ()
    belief_histogram: tuple[float, ...] = ()
    belief_remaining_deck_histogram: tuple[float, ...] = ()
    high_policy_logits: tuple[float, ...] = ()


@dataclass(frozen=True)
class _InferenceRequest:
    model_key: str
    token_features: tuple[tuple[float, ...], ...]
    token_mask: tuple[bool, ...]
    opponent_token_mask: tuple[bool, ...]
    option_features: tuple[tuple[float, ...], ...]
    option_mask: tuple[bool, ...]
    opponent_tag_id: int
    opponent_entity_id: int
    opponent_deck_id: int
    recurrent_state: tuple[float, ...] | None
    sample: bool
    temperature: float
    response_queue: queue.Queue[InferenceResult | BaseException]


class GpuInferenceServer:
    def __init__(self, *, device: str, config: InferenceServerConfig | None = None) -> None:
        self.device = torch.device(device)
        self.config = config or InferenceServerConfig()
        self._queue: queue.Queue[_InferenceRequest | object] = queue.Queue()
        self._models: dict[str, PpoTransformerPolicy] = {}
        self._lock = threading.Lock()
        self._shutdown = object()
        self._request_count = 0
        self._batch_count = 0
        self._last_batch_size = 0
        self._worker = threading.Thread(
            target=self._serve,
            daemon=True,
            name=f"gitcg-ppo-inference-{self.device.type}",
        )
        self._worker.start()

    def register_model(self, model_key: str, model: PpoTransformerPolicy) -> None:
        with self._lock:
            if model_key in self._models:
                return
            model.to(self.device)
            model.eval()
            self._models[model_key] = model

    def infer(
        self,
        *,
        model_key: str,
        token_features: tuple[tuple[float, ...], ...],
        token_mask: tuple[bool, ...],
        opponent_token_mask: tuple[bool, ...],
        option_features: tuple[tuple[float, ...], ...],
        option_mask: tuple[bool, ...],
        opponent_tag_id: int,
        opponent_entity_id: int,
        opponent_deck_id: int,
        recurrent_state: tuple[float, ...] | None,
        sample: bool,
        temperature: float,
    ) -> InferenceResult:
        response_queue: queue.Queue[InferenceResult | BaseException] = queue.Queue(maxsize=1)
        self._queue.put(
            _InferenceRequest(
                model_key=model_key,
                token_features=token_features,
                token_mask=token_mask,
                opponent_token_mask=opponent_token_mask,
                option_features=option_features,
                option_mask=option_mask,
                opponent_tag_id=opponent_tag_id,
                opponent_entity_id=opponent_entity_id,
                opponent_deck_id=opponent_deck_id,
                recurrent_state=recurrent_state,
                sample=sample,
                temperature=temperature,
                response_queue=response_queue,
            )
        )
        result = response_queue.get()
        if isinstance(result, BaseException):
            raise RuntimeError("inference server request failed") from result
        return result

    def stats(self) -> dict[str, float]:
        with self._lock:
            average_batch_size = (
                float(self._request_count) / float(self._batch_count)
                if self._batch_count
                else 0.0
            )
            return {
                "total_requests": float(self._request_count),
                "total_batches": float(self._batch_count),
                "avg_batch_size": average_batch_size,
                "last_batch_size": float(self._last_batch_size),
            }

    def close(self) -> None:
        self._queue.put(self._shutdown)
        self._worker.join(timeout=5)

    def _serve(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._shutdown:
                return
            assert isinstance(item, _InferenceRequest)
            pending = [item]
            deadline = time.perf_counter() + (self.config.max_wait_ms / 1000.0)
            while len(pending) < self.config.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    next_item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is self._shutdown:
                    self._queue.put(self._shutdown)
                    break
                assert isinstance(next_item, _InferenceRequest)
                pending.append(next_item)
            grouped: dict[str, list[_InferenceRequest]] = {}
            for request in pending:
                grouped.setdefault(request.model_key, []).append(request)
            for model_key, requests in grouped.items():
                try:
                    self._run_group(model_key=model_key, requests=requests)
                except BaseException as exc:
                    for request in requests:
                        request.response_queue.put(exc)

    def _run_group(self, *, model_key: str, requests: list[_InferenceRequest]) -> None:
        with self._lock:
            model = self._models[model_key]
        max_tokens = max(len(request.token_features) for request in requests)
        max_options = max(len(request.option_features) for request in requests)
        token_dim = len(requests[0].token_features[0])
        option_dim = len(requests[0].option_features[0])
        batch_size = len(requests)
        token_features = torch.zeros((batch_size, max_tokens, token_dim), dtype=torch.float32, device=self.device)
        token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=self.device)
        opponent_token_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=self.device)
        option_features = torch.zeros((batch_size, max_options, option_dim), dtype=torch.float32, device=self.device)
        option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool, device=self.device)
        opponent_tag_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        opponent_entity_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        opponent_deck_ids = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
        recurrent_state = None
        if model.config.recurrent_hidden_dim > 0:
            recurrent_state = torch.zeros(
                (batch_size, model.config.recurrent_hidden_dim),
                dtype=torch.float32,
                device=self.device,
            )
        for index, request in enumerate(requests):
            current_tokens = len(request.token_features)
            current_options = len(request.option_features)
            token_features[index, :current_tokens] = torch.as_tensor(request.token_features, dtype=torch.float32, device=self.device)
            token_mask[index, :current_tokens] = torch.as_tensor(request.token_mask, dtype=torch.bool, device=self.device)
            opponent_token_mask[index, :current_tokens] = torch.as_tensor(
                request.opponent_token_mask,
                dtype=torch.bool,
                device=self.device,
            )
            option_features[index, :current_options] = torch.as_tensor(request.option_features, dtype=torch.float32, device=self.device)
            option_mask[index, :current_options] = torch.as_tensor(request.option_mask, dtype=torch.bool, device=self.device)
            opponent_tag_ids[index] = int(request.opponent_tag_id)
            opponent_entity_ids[index] = int(request.opponent_entity_id)
            opponent_deck_ids[index] = int(request.opponent_deck_id)
            if recurrent_state is not None and request.recurrent_state is not None:
                current_state = torch.as_tensor(request.recurrent_state, dtype=torch.float32, device=self.device)
                recurrent_state[index, : current_state.shape[0]] = current_state[: model.config.recurrent_hidden_dim]
        with torch.no_grad():
            output = model(
                token_features,
                token_mask,
                option_features,
                option_mask,
                opponent_token_mask=opponent_token_mask,
                opponent_tag_ids=opponent_tag_ids,
                opponent_entity_ids=opponent_entity_ids,
                opponent_deck_ids=opponent_deck_ids,
                recurrent_state=recurrent_state,
            )
            logits = output.policy_logits
            temperatures = torch.tensor(
                [request.temperature if request.temperature > 0 else 1.0 for request in requests],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            logits = logits / temperatures
            log_probs = masked_log_softmax(logits, option_mask)
            probs = torch.exp(log_probs)
            entropies = (-(probs * log_probs).sum(dim=-1)).detach().cpu()
            for index, request in enumerate(requests):
                if request.sample:
                    action_index = int(torch.multinomial(probs[index], num_samples=1).item())
                else:
                    action_index = int(torch.argmax(log_probs[index]).item())
                request.response_queue.put(
                    InferenceResult(
                        action_index=action_index,
                        selected_logprob=float(log_probs[index, action_index].item()),
                        partial_value=float(output.partial_value[index].item()),
                        policy_entropy=float(entropies[index].item()),
                        batch_size=batch_size,
                        recurrent_state=(
                            tuple(float(value) for value in output.recurrent_state[index].detach().cpu().tolist())
                            if output.recurrent_state is not None
                            else None
                        ),
                        log_probs=tuple(float(value) for value in log_probs[index].detach().cpu().tolist()),
                        belief_histogram=tuple(
                            float(value) for value in output.belief_histogram[index].detach().cpu().tolist()
                        ),
                        belief_remaining_deck_histogram=tuple(
                            float(value)
                            for value in output.belief_remaining_deck_histogram[index].detach().cpu().tolist()
                        ),
                        high_policy_logits=tuple(
                            float(value) for value in output.high_policy_logits[index].detach().cpu().tolist()
                        ),
                    )
                )
        with self._lock:
            self._request_count += batch_size
            self._batch_count += 1
            self._last_batch_size = batch_size
