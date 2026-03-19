from __future__ import annotations

from importlib import import_module

from .agents import HeuristicAgent, LegalRandomAgent, build_baseline_agent
from .decks import SAMPLE_DECK_A, SAMPLE_DECK_B, SMALL_DECK_MATCHUPS, SMALL_DECK_POOL
from .env import GitcgDecisionEnv
from .replay import (
    append_episode_records,
    deserialize_episode_record,
    group_episode_records_by_matchup,
    load_episode_records,
    read_episode_records,
    resolve_replay_paths,
    sample_episode_records,
    serialize_episode_record,
)
from .rollout import run_episode
from .schema import (
    ActionChoice,
    DecisionContext,
    DecisionType,
    DeckSpec,
    EnvConfig,
    EpisodeRecord,
    Matchup,
    OptionKind,
    StateSnapshot,
    TrajectoryStep,
)

__all__ = [
    "ActionChoice",
    "append_episode_records",
    "build_baseline_agent",
    "deserialize_episode_record",
    "DecisionContext",
    "DecisionType",
    "DeckSpec",
    "EnvConfig",
    "EpisodeRecord",
    "GitcgDecisionEnv",
    "group_episode_records_by_matchup",
    "HeuristicAgent",
    "load_episode_records",
    "LegalRandomAgent",
    "Matchup",
    "OptionKind",
    "read_episode_records",
    "resolve_replay_paths",
    "run_episode",
    "SAMPLE_DECK_A",
    "SAMPLE_DECK_B",
    "sample_episode_records",
    "SMALL_DECK_MATCHUPS",
    "SMALL_DECK_POOL",
    "StateSnapshot",
    "TrajectoryStep",
    "serialize_episode_record",
    "PpoAgent",
    "PpoModelConfig",
    "PpoTransformerPolicy",
    "TokenObservationEncoder",
    "ScriptedBootstrapAgent",
    "PpoTrainArtifacts",
    "bootstrap_pretrain_from_episodes",
    "train_ppo_from_episodes",
    "load_ppo_train_artifacts",
    "save_ppo_training_checkpoint",
    "load_ppo_training_checkpoint",
    "save_ppo_deployment_checkpoint",
    "load_ppo_deployment_checkpoint",
    "PpoManagedRoundResult",
    "PpoManagedLoopArtifacts",
    "run_ppo_managed_loop",
]

_LAZY_EXPORTS = {
    "PpoAgent": (".ppo_agent", "PpoAgent"),
    "TokenObservationEncoder": (".ppo_features", "TokenObservationEncoder"),
    "PpoModelConfig": (".ppo_model", "PpoModelConfig"),
    "PpoTransformerPolicy": (".ppo_model", "PpoTransformerPolicy"),
    "PpoTrainArtifacts": (".ppo_training", "PpoTrainArtifacts"),
    "bootstrap_pretrain_from_episodes": (".ppo_training", "bootstrap_pretrain_from_episodes"),
    "train_ppo_from_episodes": (".ppo_training", "train_ppo_from_episodes"),
    "load_ppo_train_artifacts": (".ppo_training", "load_ppo_train_artifacts"),
    "ScriptedBootstrapAgent": (".bootstrap_agent", "ScriptedBootstrapAgent"),
    "save_ppo_training_checkpoint": (".ppo_checkpoint", "save_ppo_training_checkpoint"),
    "load_ppo_training_checkpoint": (".ppo_checkpoint", "load_ppo_training_checkpoint"),
    "save_ppo_deployment_checkpoint": (".ppo_checkpoint", "save_ppo_deployment_checkpoint"),
    "load_ppo_deployment_checkpoint": (".ppo_checkpoint", "load_ppo_deployment_checkpoint"),
    "PpoManagedRoundResult": (".ppo_pipeline", "PpoManagedRoundResult"),
    "PpoManagedLoopArtifacts": (".ppo_pipeline", "PpoManagedLoopArtifacts"),
    "run_ppo_managed_loop": (".ppo_pipeline", "run_ppo_managed_loop"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
