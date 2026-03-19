# Training Methods

This file records the current primary method after the refactor.

## Current Primary Method

The active training path is now:

- `PPO`
- `CTDE`
- `belief modeling`
- `dual critic`
- `P2SRO frozen-population training`

This is the report-backed current primary method.

## Problem Framing

GI-TCG is treated as a two-player partially observable stochastic game.

Engineering simplification:

- execution-time policy acts from `player_view`
- training may use `full_state` only through privileged heads and labels

This keeps the deployed policy honest while still using CTDE to stabilize learning.

## Architecture

The default actor-critic stack is:

- token/entity observation encoder
- tiny Transformer backbone
- legal-option scoring head with masking
- partial critic
- belief head
- oracle critic

Current opponent-modeling path:

- learned opponent entity embedding
- learned opponent deck embedding
- opponent behavior embedding pooled from visible opponent/history tokens
- fallback `opponent_tag` only for backward compatibility

Current belief targets:

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

Current oracle target:

- privileged full-state value estimate

Current recurrent/long-horizon support:

- recurrent actor backbone: `Transformer + GRU`
- explicit recent-history tokens
- recent-round replay used only for `critic / belief / oracle`
- explicit-range SePoT search inputs:
  - public state
  - self explicit range
  - opponent explicit range
- independent search heads:
  - transformed search-policy head
  - search critic / value-summary head
- phase-triggered runtime SePoT takeover on key `ACTION` states
- training-time shallow look-ahead search teacher built from perfect-information replay states

This means the current system now has both:

- runtime search: `PPO default + triggered SePoT takeover`
- training-side search guidance: perfect-information shallow teacher

## PPO Objective

The optimizer uses PPO with:

- clipped policy objective
- partial value regression
- entropy regularization
- belief supervised loss
- oracle value loss
- oracle-to-partial distillation loss

The current implementation keeps the actor path separate from the oracle path in deployment artifacts.

## Reward Design

Main outcome semantics:

- win: `+1`
- loss: `-1`
- draw / both-fail at round 15: `-1`

Training shaping:

- PBRS-style potential over HP, alive characters, energy readiness, and board presence
- anti-draw penalties on rounds `13`, `14`, and `15+`

Important distinction:

- shaped rewards are for training only
- gate evaluation still reports actual wins, losses, and draws

## Bootstrap

Cold start uses a scripted bootstrap agent.

Priority order:

1. skill
2. card
3. switch
4. tuning
5. end turn

And end turn is delayed whenever another action exists.

Bootstrap is used only for warm start. The main loop after that is PPO self-play.

## P2SRO Population Loop

The current population controller is now `P2SRO`.

Formal members:

- frozen population
- `active_main_slot`
- `active_br_slot`
- empirical payoff matrix
- replicator-dynamics meta-strategy

Locked semantics:

- active slots: `2`
- training opponents: sampled from the frozen population by meta probability
- deployment agent: `meta-mixture agent`
- baselines: evaluation-only and excluded from the population

Selection and promotion are payoff-driven:

1. shortlist safe PPO epochs
2. evaluate each active-slot candidate against the current meta-policy
3. choose the epoch with the best `expected payoff vs meta`
4. freeze/promote only if it clears the lineage-specific payoff margin
5. recompute the meta-strategy with replicator dynamics

## Current Public And Private Information Boundary

The current PPO actor path is intended to be honest at deployment time.

Directly encoded private/public information:

- own private information
  - own hand cards
  - own dice
- public information
  - characters, HP, energy, aura
  - character-attached public entities
  - summons, supports, combat statuses
  - public action history
  - opponent public dice count
  - legal action semantics

Directly excluded from actor input:

- opponent hidden hand
- opponent exact dice colors
- oracle-only full state

Current action semantic coverage in the actor path includes:

- `use_skill` with skill definition ids
- `play_card` with card definition ids
- `choose_active` and `select_card` with candidate definition ids
- `switch_hands` with removed hand-card definition ids
- `elemental_tuning` with removed-card definition ids
- `reroll` with explicit rerolled-dice subset encoding
- target semantics for targeted actions

Current training-time search/public boundary:

- opponent public dice count is tracked and injected as public information
- opponent exact dice colors remain hidden
- runtime search uses public belief state plus explicit ranges
- perfect-information search guidance is produced only during training annotation
- the deployed actor may run triggered SePoT search, but it does not read hidden truth

## Checkpoints

Training checkpoint includes:

- full model
- oracle critic
- optimizer state
- metadata

Deployment checkpoint includes only:

- token encoder config
- actor backbone
- policy head
- belief path

It excludes oracle parameters on purpose.

## Managed Loop

The default managed loop is:

1. bootstrap if no checkpoint exists
2. seed a frozen population if needed
3. train `active_main_slot` against the current meta-policy
4. train `active_br_slot` against the current meta-policy
5. evaluate both active slots against the frozen population
6. update the payoff matrix and recompute the meta-strategy
7. deploy and benchmark the current meta-mixture population policy

Default entrypoint:

- [run_ppo_main_loop.py](/E:/Coding/WorldModel/research/world_model/scripts/run_ppo_main_loop.py)

## What Is Intentionally Not In The Current Mainline

These are intentionally not part of the current default path:

- ISMCTS mainline acting
- Dreamer/RSSM latent imagination
- world-model replay reanalysis as the default trainer
- removed retention layers as active control logic for the live lineage

Those are possible second-stage enhancements, not the default algorithm.

## Current Limitations

The main current limitations are now:

- runtime SePoT is now in the acting path and simulator-backed; the remaining issues are mainly performance and runtime engineering, not missing search components
- the action layer is now hierarchical and codebook-first; remaining work is taxonomy tuning and performance, not another interface rewrite
- long-term evaluation metrics are no longer the control path; they remain optional diagnostics
- actor collection is still not true multi-process

Important practical meaning:

- the system is no longer bottlenecked mainly by feature omissions
- the current bottleneck is now mainly engineering/performance around runtime SePoT and payoff evaluation, not missing top-level control structure

## Current Search-Teacher Layer

The current search component is training-time guidance:

- `PublicStateTracker`
  - derives only from `player_view`, public history, and known decks
- `LookaheadSearchConfig` / shallow search annotation
  - uses recorded perfect-information replay states during training only
- search-teacher policy/value targets
  - supervise the PPO actor-critic during training
- oracle critic / oracle distillation
  - remain the privileged route for perfect-information guidance

Current search scope:

- runtime SePoT:
  - default acting path remains PPO
  - only triggered `ACTION` states are taken over by SePoT search
  - runtime search uses explicit public belief state + explicit ranges + independent search heads
- hierarchical action layer:
  - fixed low-level executable action codebook
  - taxonomy-driven high-level action classes
  - exact payment binding
  - public-slot target binding
  - shared legality engine across PPO / SePoT / teacher search
- training-time search teacher:
  - annotate a limited number of key states per episode
  - use shallow look-ahead from replay-time perfect-information states
  - provide policy/value teacher targets
  - never act directly at deployment time

Current search-teacher scope:

- annotate a limited number of key states per episode
- use shallow look-ahead from replay-time perfect-information states
- provide policy/value teacher targets
- never act directly at deployment time

## Next Algorithmic Step

Expected next work:

- speed up runtime SePoT without weakening the public-belief/runtime honesty boundary
- strengthen perfect-information distillation / oracle guiding
- improve the training-time search teacher
- tune and extend the active taxonomy/codebook without restoring the legacy option-list interface

## Explicit Non-Goals For The Next Step

The next step is not:

- a return to Dreamer as the mainline
- a switch to MCTS / ISMCTS / ReBeL as the default trainer
- a cheating runtime planner that reads `full_state` or `full_state_json`
- a move to held-out-target overfitting by mixing held-out opponents into training
- a random preserved-model replacement of the main lineage every round

## Still Deferred

One scheme-one engineering item is still deferred:

- true multi-process CPU actors with a single GPU learner/inference server

The code exposes the interface, but the simulator binding still falls back to safe serial collection in the current implementation.
