# GI-TCG PPO-CTDE-Belief-SePoT-P2SRO

This package now uses the report-backed `PPO + CTDE + belief + dual critic + SePoT + P2SRO` stack as the training path.

## Default Method

The current mainline follows the research report's recommended scheme:

- problem view: `POSG / POMDP`
- policy optimization: `PPO`
- training paradigm: `CTDE`
- hidden information handling: `belief head`
- value estimation: `partial critic + oracle critic`
- opponent robustness: `P2SRO frozen-population training + meta-mixture deployment`

The main loop is:

1. optional scripted bootstrap warm start
2. seed or load frozen `P2SRO` population
3. train `active_main_slot` and `active_br_slot` against the current meta-policy
4. update the empirical payoff matrix and recompute the meta-strategy
5. deploy and benchmark the current meta-mixture population policy

## Why The Switch

The previous Dreamer-style event world model had three practical problems for this task:

- decision quality depended on latent imagination quality, which was hard to stabilize
- rollout speed and deployability were poor
- the mainline was drifting away from the report's recommended method

The new default is simpler, easier to deploy, and matches the report's primary recommendation more closely.

## Environment Boundaries

The environment and action interfaces are intentionally preserved:

- `GitcgDecisionEnv`
- `DecisionContext`
- `HierarchicalActionCodebook`
- `PolicyAgent`
- `EpisodeRecord`

The actor path must not read `full_state` during decision making.

Current split:

- actor / partial critic / belief head: use `player_view + history`
- oracle critic: training only, may use `full_state`

## Model Overview

The default model is a lightweight Transformer actor-critic.

Input encoding:

- public tokens: characters, summons, supports, statuses, round/request/global state
- private tokens: current player's hand cards and dice
- history tokens: recent decision events
- legal actions: live legal option list encoded separately and scored with masking

Output heads:

- `policy head`
- `partial critic`
- `belief head`
- `oracle critic`

Current opponent modeling is no longer just tag conditioning. The deployed actor now fuses:

- public/global summary from visible tokens
- opponent behavior embedding pooled from opponent-owned/history tokens
- learned opponent entity embedding
- learned opponent deck embedding

Current belief supervision has two layers:

- base hidden-info targets
  - opponent hand histogram
  - opponent hand size
  - opponent remaining deck histogram
  - opponent burst-ready count
- game-specific belief groups
  - `burst_finisher`
  - `direct_damage`
  - `heal_or_revive`
  - `shield_or_damage_mitigation`
  - `dice_fix_or_ramp`
  - `draw_or_search`
  - `tempo_switch`
  - `equipment_weapon_or_artifact`
  - `support_engine`
  - `summon_or_board_pressure`

## Rewards

The default training semantics are:

- win: `+1`
- loss: `-1`
- 15-round forced draw / both-fail outcome: `-1`

Training reward combines:

- terminal reward
- PBRS-style potential shaping
- anti-draw time penalty on rounds `13-15`

Analysis files still keep draw counts explicitly; they are not hidden.

## Bootstrap

Cold start uses a scripted bootstrap agent instead of human data.

Priority order:

1. use skill
2. play card
3. switch active
4. elemental tuning
5. declare end

And `declare_end` is delayed whenever any non-end action exists.

Bootstrap data currently comes from:

- scripted vs scripted
- scripted vs random

## P2SRO Population

The active population loop is now `P2SRO`.

Current outer-loop structure:

- frozen population
- `active_main_slot`
- `active_br_slot`
- empirical payoff matrix
- replicator-dynamics meta solver
- per-episode meta-mixture deployment

Locked semantics:

- active training slots: `2`
- meta solver: `Replicator Dynamics`
- train-time opponents: sampled from frozen population by meta probability
- deploy-time agent: meta-mixture over frozen population
- baselines: evaluation-only, not part of the population

Selection and promotion are now payoff-driven:

- PPO loss is still only an optimizer signal
- epoch shortlist is still filtered by safe `approx_kl`
- active-slot epoch choice is by `expected payoff vs current meta-policy`
- freezing/promotion is by payoff margin against the frozen reference of the same lineage

## Current Status

The active mainline has changed again.

What is in the current hot path now:

- recurrent PPO actor-critic (`Transformer + GRU`)
- strict actor/public-information boundary at runtime
- oracle critic and oracle distillation during training
- explicit `PublicBeliefState` search inputs:
  - public state
  - self explicit range
  - opponent explicit range
- independent search heads:
  - transformed search-policy head
  - search critic / value-summary head
- phase-triggered SePoT-style runtime search takeover:
  - default acting path is PPO
  - triggered `ACTION` states can be re-scored by SePoT search
- training-time perfect-information shallow look-ahead teacher kept only as training-side guiding
- payoff-driven `P2SRO` population loop:
  - frozen population
  - `active_main_slot`
  - `active_br_slot`
  - replicator meta-policy
  - meta-mixture deployment
- current-round PPO training plus recent-round replay for `critic / belief / oracle`
- hierarchical DouZero-style action layer:
  - fixed low-level executable action codebook
  - taxonomy-driven high-level action classes
  - exact payment binding
  - public-slot target binding
  - shared legality engine for PPO / SePoT / teacher search
- richer public-state encoding in the actor path:
  - character aura
  - character-attached public entities
  - public supports, summons, and combat statuses
  - opponent public dice count

Current public/private boundary in the deployed actor path:

- included:
  - own hand cards
  - own dice
  - public board state
  - public action history
  - opponent public dice count
- excluded:
  - opponent hidden hand
  - opponent exact dice colors
  - any direct `full_state` read during actor decisions

What was removed from the active runtime:

- the failed online belief-guided planner path
- the legacy event/gate stack

## Current Limitations

The most important current limitations are now:

- runtime SePoT is active and simulator-backed, but it is still expensive on CPU and needs engineering/performance work rather than algorithm simplification
- long-term evaluation diagnostics (`TrueSkill / alpha-rank / diversity / exploitability`) are no longer the control path; they can still exist as offline diagnostics
- true multi-process CPU actor collection is still deferred

Important interpretation:

- runtime search is no longer absent; it is now `PPO default + triggered simulator-backed SePoT takeover`
- perfect-information guidance remains useful, but only on the training side
- the current route is now:
  - `PPO mainline`
  - `perfect-information distillation / oracle guiding`
  - `triggered SePoT runtime search`
  - `training-time look-ahead search teacher`
  - `P2SRO frozen-population self-play`
  - `hierarchical DouZero-style action codebook`

## Next Step

The next step is:

- speed up the current SePoT path without weakening its public-belief/runtime honesty
- strengthen the training-time search teacher
- keep improving `P2SRO` throughput and payoff-evaluation efficiency
- expand and tune the taxonomy/codebook without reintroducing the legacy option-list interface

## Not Doing Yet

The current mainline is explicitly **not** doing these things:

- no runtime cheating planner
- no direct hidden-truth inputs to the deployed actor
- no return to the Dreamer/world-model route
- no requirement that checkpoint progression be gated by removed preservation layers

The actor/training-time-search split remains strictly honest:

- no direct live `full_state` / `full_state_json` reads in the deployed actor path
- no leaking opponent hidden hand
- no leaking opponent exact dice colors

## Not Doing Yet

These are explicitly not the current next step:

- switching the mainline back to Dreamer / RSSM planning
- making MCTS / ISMCTS / ReBeL the default acting path
- using hidden truth in search-teacher reconstruction
- turning held-out standards into training opponents
- using stale cross-round replay for PPO policy updates

## Default Entrypoints

Default PPO mainline:

```powershell
$env:PYTHONPATH='E:\Coding\WorldModel\research\world_model\src'
python research/world_model/scripts/run_ppo_main_loop.py `
  --workspace D:\WorldModelTemp\ppo_oracle_search_psro_live `
  --rounds 20 `
  --device cuda `
  --bootstrap-episodes-per-matchup 8 `
  --self-play-episodes-per-matchup 8 `
  --max-decisions 512
```

## Resume

Windows double-click resume helper:

- [resume_event_loop.cmd](/E:/Coding/WorldModel/resume_event_loop.cmd)

The helper:

- checks `run_manifest.json`
- avoids launching a duplicate process if the run is already alive
- resumes the PPO mainline workspace by default

## Workspace Outputs

Workspace root:

- `event_loop_status.json`
- `event_loop_history.jsonl`
- `event_loop_state.json`
- `event_loop_result.json`
- `run_manifest.json`
- `p2sro_state.json`
- `meta_deployment.json`

Per round:

- `bootstrap.jsonl` when cold-start bootstrap runs
- `bootstrap_analysis.json`
- `bootstrap_train/`
- `self_play.jsonl`
- `br_self_play.jsonl`
- `self_play_analysis.json`
- `train/last.pt`
- `train/deployment.pt`
- `train/training_summary.json`
- `train/selection_summary.json`
- `br_train/selection_summary.json`
- `analysis.json`

Deployment checkpoint rule:

- `train/deployment.pt` contains actor + belief path only
- oracle critic and optimizer state stay out of deployment artifacts

Current engineering limitation:

- the algorithmic mainline is complete for scheme-one rating/opponent-modeling work
- true multi-process CPU actors are still not enabled because the current simulator binding is not thread-safe
- rollout therefore still falls back to the safe serial collection path

## Validation

Run the targeted PPO tests:

```powershell
$env:PYTHONPATH='E:\Coding\WorldModel\research\world_model\src'
python -m unittest `
  research.world_model.tests.test_bootstrap_agent `
  research.world_model.tests.test_ppo_checkpoint `
  research.world_model.tests.test_ppo_training `
  research.world_model.tests.test_ppo_pipeline
```

Run the full test suite:

```powershell
$env:PYTHONPATH='E:\Coding\WorldModel\research\world_model\src'
python -m unittest discover -s research/world_model/tests
```
