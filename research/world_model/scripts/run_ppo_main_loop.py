from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PPO + Oracle Guiding + SePoT + P2SRO loop for GI-TCG.",
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--bootstrap-episodes-per-matchup", type=int, default=8)
    parser.add_argument("--self-play-episodes-per-matchup", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--draw-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--bootstrap-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--policy-coef", type=float, default=4.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--belief-coef", type=float, default=0.2)
    parser.add_argument("--oracle-coef", type=float, default=0.2)
    parser.add_argument("--reference-kl-coef", type=float, default=0.005)
    parser.add_argument("--target-kl-low", type=float, default=0.01)
    parser.add_argument("--target-kl-high", type=float, default=0.05)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inference-max-batch-size", type=int, default=192)
    parser.add_argument("--inference-max-wait-ms", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--retain-round-directories", type=int, default=6)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--status-print", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    if platform.system() == "Windows":
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    from gitcg_world_model.ppo_pipeline import run_ppo_managed_loop

    args = parse_args()
    artifacts = run_ppo_managed_loop(
        workspace=args.workspace,
        rounds=args.rounds,
        init_checkpoint=args.init_checkpoint,
        bootstrap_episodes_per_matchup=args.bootstrap_episodes_per_matchup,
        self_play_episodes_per_matchup=args.self_play_episodes_per_matchup,
        max_rounds=args.max_rounds,
        max_decisions=args.max_decisions,
        draw_penalty=args.draw_penalty,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        epochs=args.epochs,
        bootstrap_epochs=args.bootstrap_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        policy_coef=args.policy_coef,
        value_coef=args.value_coef,
        belief_coef=args.belief_coef,
        oracle_coef=args.oracle_coef,
        reference_kl_coef=args.reference_kl_coef,
        target_kl_low=args.target_kl_low,
        target_kl_high=args.target_kl_high,
        use_amp=args.use_amp,
        device=args.device,
        workers=args.workers,
        inference_max_batch_size=args.inference_max_batch_size,
        inference_max_wait_ms=args.inference_max_wait_ms,
        seed_start=args.seed_start,
        resume=args.resume,
        enable_status_print=args.status_print,
        retain_round_directories=args.retain_round_directories,
    )
    payload = {
        "workspace": artifacts.workspace,
        "working_checkpoint": artifacts.working_checkpoint,
        "deployment_checkpoint": artifacts.deployment_checkpoint,
        "status_path": artifacts.status_path,
        "history_path": artifacts.history_path,
        "state_path": artifacts.state_path,
        "rounds": [
            {
                "round": round_result.round_index,
                "bootstrap_path": round_result.bootstrap_path,
                "self_play_path": round_result.self_play_path,
                "training_checkpoint": round_result.train_artifacts.training_checkpoint,
                "deployment_checkpoint": round_result.deployment_checkpoint,
                "working_checkpoint": round_result.working_checkpoint,
                "summary_path": round_result.train_artifacts.summary_path,
                "analysis_path": round_result.analysis_path,
            }
            for round_result in artifacts.rounds
        ],
    }
    destination = Path(args.workspace) / "event_loop_result.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
