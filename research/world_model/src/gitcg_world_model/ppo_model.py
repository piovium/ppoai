from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PpoModelConfig:
    token_dim: int
    option_dim: int
    high_action_dim: int
    privileged_dim: int
    belief_histogram_dim: int
    belief_deck_histogram_dim: int
    max_tokens: int
    belief_group_dim: int = 10
    opponent_tag_vocab_size: int = 7
    opponent_entity_vocab_size: int = 257
    deck_vocab_size: int = 65
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    option_hidden_dim: int = 512
    privileged_hidden_dim: int = 512
    recurrent_hidden_dim: int = 512
    search_range_summary_dim: int = 0
    search_context_dim: int = 16


@dataclass(frozen=True)
class PpoForwardOutput:
    policy_logits: torch.Tensor
    high_policy_logits: torch.Tensor
    partial_value: torch.Tensor
    belief_histogram: torch.Tensor
    belief_hand_size: torch.Tensor
    belief_remaining_deck_histogram: torch.Tensor
    belief_burst_ready_count: torch.Tensor
    belief_group_hand_logits: torch.Tensor
    belief_group_deck_logits: torch.Tensor
    token_summary: torch.Tensor
    opponent_embedding: torch.Tensor
    search_public_summary: torch.Tensor
    recurrent_state: torch.Tensor | None


class PpoTransformerPolicy(nn.Module):
    def __init__(self, config: PpoModelConfig):
        super().__init__()
        self.config = config
        self.token_projection = nn.Sequential(
            nn.Linear(config.token_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.position_embedding = nn.Embedding(config.max_tokens, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        transformer_kwargs = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            # Pre-norm encoders do not use nested-tensor fastpaths in current PyTorch.
            # Disable the incompatible optimization explicitly instead of relying on
            # the default and paying the warning cost every time the model is built.
            transformer_kwargs["enable_nested_tensor"] = False
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            **transformer_kwargs,
        )
        self.opponent_tag_embedding = nn.Embedding(max(1, config.opponent_tag_vocab_size), config.d_model)
        self.opponent_entity_embedding = nn.Embedding(max(1, config.opponent_entity_vocab_size), config.d_model)
        self.opponent_deck_embedding = nn.Embedding(max(1, config.deck_vocab_size), config.d_model)
        self.opponent_behavior_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.summary_fusion = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.recurrent_cell = (
            nn.GRUCell(config.d_model, config.recurrent_hidden_dim)
            if config.recurrent_hidden_dim > 0
            else None
        )
        self.recurrent_projection = (
            nn.Sequential(
                nn.Linear(config.recurrent_hidden_dim, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.SiLU(),
            )
            if config.recurrent_hidden_dim > 0
            else nn.Identity()
        )
        self.option_projection = nn.Sequential(
            nn.Linear(config.option_dim, config.option_hidden_dim),
            nn.LayerNorm(config.option_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.option_hidden_dim, config.d_model),
            nn.SiLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.high_policy_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.high_action_dim),
        )
        self.partial_value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.belief_histogram_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.belief_histogram_dim),
        )
        self.belief_remaining_deck_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.belief_deck_histogram_dim),
        )
        self.belief_hand_size_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.belief_burst_ready_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )
        self.belief_group_hand_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.belief_group_dim),
        )
        self.belief_group_deck_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, config.belief_group_dim),
        )
        self.oracle_encoder = nn.Sequential(
            nn.Linear(config.privileged_dim, config.privileged_hidden_dim),
            nn.LayerNorm(config.privileged_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.privileged_hidden_dim, config.d_model),
            nn.SiLU(),
        )
        self.oracle_value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        search_range_summary_dim = (
            int(config.search_range_summary_dim)
            if int(config.search_range_summary_dim) > 0
            else (config.belief_histogram_dim * 2) + 12
        )
        self.search_range_summary_dim = search_range_summary_dim
        self.search_context_dim = max(1, int(config.search_context_dim))
        self.search_self_range_projection = nn.Sequential(
            nn.Linear(self.search_range_summary_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.search_opponent_range_projection = nn.Sequential(
            nn.Linear(self.search_range_summary_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.search_context_projection = nn.Sequential(
            nn.Linear(self.search_context_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
        )
        self.search_option_head = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )
        self.search_value_head = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
        option_features: torch.Tensor,
        option_mask: torch.Tensor,
        *,
        opponent_token_mask: torch.Tensor | None = None,
        opponent_tag_ids: torch.Tensor | None = None,
        opponent_entity_ids: torch.Tensor | None = None,
        opponent_deck_ids: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> PpoForwardOutput:
        batch_size, token_count, _ = token_features.shape
        position_ids = torch.arange(token_count, device=token_features.device).unsqueeze(0).expand(batch_size, -1)
        token_hidden = self.token_projection(token_features) + self.position_embedding(position_ids)
        encoded = self.transformer(token_hidden, src_key_padding_mask=~token_mask)
        summary = encoded[:, 0, :]
        if opponent_token_mask is None:
            opponent_token_mask = torch.zeros_like(token_mask)
        valid_opponent_mask = opponent_token_mask & token_mask
        opponent_counts = valid_opponent_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled_opponent = (encoded * valid_opponent_mask.unsqueeze(-1)).sum(dim=1) / opponent_counts
        if opponent_tag_ids is None:
            opponent_tag_ids = torch.zeros((batch_size,), dtype=torch.long, device=token_features.device)
        opponent_tag_ids = opponent_tag_ids.clamp(min=0, max=max(0, self.config.opponent_tag_vocab_size - 1))
        if opponent_entity_ids is None:
            opponent_entity_ids = torch.zeros((batch_size,), dtype=torch.long, device=token_features.device)
        opponent_entity_ids = opponent_entity_ids.clamp(
            min=0,
            max=max(0, self.config.opponent_entity_vocab_size - 1),
        )
        if opponent_deck_ids is None:
            opponent_deck_ids = torch.zeros((batch_size,), dtype=torch.long, device=token_features.device)
        opponent_deck_ids = opponent_deck_ids.clamp(min=0, max=max(0, self.config.deck_vocab_size - 1))
        opponent_tag_embedding = self.opponent_tag_embedding(opponent_tag_ids)
        opponent_entity_embedding = self.opponent_entity_embedding(opponent_entity_ids) + opponent_tag_embedding
        opponent_deck_embedding = self.opponent_deck_embedding(opponent_deck_ids)
        opponent_behavior_embedding = self.opponent_behavior_projection(pooled_opponent)
        fused_summary = self.summary_fusion(
            torch.cat(
                [
                    summary,
                    opponent_behavior_embedding,
                    opponent_entity_embedding,
                    opponent_deck_embedding,
                ],
                dim=-1,
            )
        )
        if self.recurrent_cell is not None:
            if recurrent_state is None:
                recurrent_state = fused_summary.new_zeros((batch_size, self.config.recurrent_hidden_dim))
            next_recurrent_state = self.recurrent_cell(fused_summary, recurrent_state)
            head_summary = self.recurrent_projection(next_recurrent_state)
        else:
            next_recurrent_state = None
            head_summary = fused_summary
        option_hidden = self.option_projection(option_features)
        summary_expanded = head_summary.unsqueeze(1).expand(-1, option_hidden.shape[1], -1)
        policy_logits = self.policy_head(torch.cat([summary_expanded, option_hidden], dim=-1)).squeeze(-1)
        masked_policy_logits = policy_logits.masked_fill(~option_mask, _masked_fill_value(policy_logits))
        return PpoForwardOutput(
            policy_logits=masked_policy_logits,
            high_policy_logits=self.high_policy_head(head_summary),
            partial_value=self.partial_value_head(head_summary).squeeze(-1),
            belief_histogram=self.belief_histogram_head(head_summary),
            belief_hand_size=self.belief_hand_size_head(head_summary).squeeze(-1),
            belief_remaining_deck_histogram=self.belief_remaining_deck_head(head_summary),
            belief_burst_ready_count=self.belief_burst_ready_head(head_summary).squeeze(-1),
            belief_group_hand_logits=self.belief_group_hand_head(head_summary),
            belief_group_deck_logits=self.belief_group_deck_head(head_summary),
            token_summary=head_summary,
            opponent_embedding=opponent_behavior_embedding + opponent_entity_embedding + opponent_deck_embedding,
            search_public_summary=fused_summary,
            recurrent_state=next_recurrent_state,
        )

    def oracle_value(self, privileged_state: torch.Tensor) -> torch.Tensor:
        hidden = self.oracle_encoder(privileged_state)
        return self.oracle_value_head(hidden).squeeze(-1)

    def search_option_logits(
        self,
        *,
        option_features: torch.Tensor,
        self_range_summary: torch.Tensor,
        opponent_range_summary: torch.Tensor,
        search_context_summary: torch.Tensor,
    ) -> torch.Tensor:
        option_hidden = self.option_projection(option_features)
        self_range_hidden = self.search_self_range_projection(self_range_summary).unsqueeze(1).expand_as(option_hidden)
        opponent_range_hidden = self.search_opponent_range_projection(opponent_range_summary).unsqueeze(1).expand_as(option_hidden)
        search_context_hidden = self.search_context_projection(search_context_summary).unsqueeze(1).expand_as(option_hidden)
        return self.search_option_head(
            torch.cat(
                [
                    option_hidden,
                    self_range_hidden,
                    opponent_range_hidden,
                    search_context_hidden,
                ],
                dim=-1,
            )
        ).squeeze(-1)

    def search_state_value(
        self,
        *,
        self_range_summary: torch.Tensor,
        opponent_range_summary: torch.Tensor,
        search_context_summary: torch.Tensor,
    ) -> torch.Tensor:
        return self.search_value_head(
            torch.cat(
                [
                    self.search_self_range_projection(self_range_summary),
                    self.search_opponent_range_projection(opponent_range_summary),
                    self.search_context_projection(search_context_summary),
                ],
                dim=-1,
            )
        ).squeeze(-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, _masked_fill_value(logits))
    return torch.log_softmax(masked_logits, dim=-1)


def _masked_fill_value(tensor: torch.Tensor) -> float:
    if torch.is_floating_point(tensor):
        return float(torch.finfo(tensor.dtype).min)
    return -1e9
