# Action Hierarchy Guide

## Purpose

This file documents the **currently active** high-level to low-level action bridge for the new codebook-first pipeline:

- `PPO`
- `Oracle Guiding`
- `SePoT`
- `P2SRO`
- hierarchical action layer

It is meant to answer two practical questions:

1. `现在的高级语义是什么？`
2. `一个高级语义是怎么接到低级可执行动作上的？`

This is the file to review before changing the current high-level categories.

## Current Files That Matter

- Category names:
  - [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
- Thresholds / key names / subjective hierarchy priors:
  - [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)
- Runtime mapping logic:
  - [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
- Low-level executable action enumeration:
  - [action_adapter.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_adapter.py)

## The Bridge: High-Level Does Not Replace Low-Level

The current system does **not** work like:

- model outputs one abstract label
- rule code guesses which concrete move to execute

That would be too loose.

The current bridge is:

1. Enumerate all legal **low-level executable actions** for the current decision.
2. For each low-level action, assign **one primary high-level label**.
3. Build a per-state `high_to_low_map`.
4. Train and search still execute on the **low-level action space**.
5. The high-level head only provides:
   - structure
   - grouping
   - auxiliary supervision
   - candidate compression bias

So the active relation is:

`visible state + low-level action spec -> one high-level key`

not:

`high-level label -> guess a move`

## Low-Level Action Space

The low-level layer is the real executable layer.

Each low-level action is a [LowLevelActionSpec](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/schema.py) with:

- request type
- action kind
- subject definition id
- exact payment
- public slot targets
- reroll mask / switch mask / select card definition / tuning discard info

This layer is the one used by:

- PPO execution
- SePoT rollout
- teacher search low-level targets
- replay serialization

## Current High-Level Categories

These names come from [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml).

### ACTION

- `skill_commit`
- `skill_setup`
- `card_resource`
- `card_draw_search`
- `card_buff_equip`
- `card_board_setup`
- `card_defense_heal`
- `switch_tempo`
- `tune_fix`
- `declare_end`

### REROLL_DICE

- `keep_all`
- `reroll_minor`
- `reroll_major`
- `reroll_all`

### CHOOSE_ACTIVE

- `choose_mainline`
- `choose_safety`
- `choose_setup`

### SELECT_CARD

- `keep_core`
- `keep_curve`
- `keep_resource`
- `keep_flex`

### SWITCH_HANDS

- `switch_none`
- `switch_minor`
- `switch_major`
- `switch_all`

## Current Mapping Rules

These rules are implemented in `_high_level_key_for_spec(...)` inside [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py).

The rule inputs are:

- the low-level action spec itself
- current visible state
- acting player
- thresholds and key names from [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)

### ACTION rules

1. `ACTION_DECLARE_END`
   - always maps to `declare_end`

2. `ACTION_ELEMENTAL_TUNING`
   - always maps to `tune_fix`

3. `ACTION_SWITCH_ACTIVE`
   - always maps to `switch_tempo`

4. `ACTION_USE_SKILL`
   - if the skill has at least one target **and**
   - `len(used_dice) >= skill_commit_min_used_dice`
   - then map to `skill_commit`
   - else map to `skill_setup`

5. `ACTION_PLAY_CARD`
   - current version uses a **definition-id threshold split**
   - the thresholds are:
     - `defense_heal_min`
     - `board_setup_min`
     - `buff_equip_min`
     - `resource_min`
   - priority is checked from high threshold to low threshold:
     - `>= defense_heal_min` -> `card_defense_heal`
     - else `>= board_setup_min` -> `card_board_setup`
     - else `>= buff_equip_min` -> `card_buff_equip`
     - else `>= resource_min` -> `card_resource`
     - else -> `card_draw_search`

### REROLL_DICE rules

Use reroll subset size relative to current visible dice count:

- no dice rerolled -> `keep_all`
- reroll count == total dice -> `reroll_all`
- reroll count <= `reroll_minor_max` -> `reroll_minor`
- otherwise -> `reroll_major`

### CHOOSE_ACTIVE rules

Look at the chosen slot's visible character state:

- if `health / max_health <= choose_safety_health_ratio`
  - -> `choose_safety`
- else if `energy >= max_energy - choose_setup_energy_gap`
  - -> `choose_setup`
- else
  - -> `choose_mainline`

### SELECT_CARD rules

Use the selected card definition id and threshold bands:

- `>= resource_min` -> `keep_resource`
- else `>= curve_min` -> `keep_curve`
- else `>= core_min` -> `keep_core`
- else -> `keep_flex`

### SWITCH_HANDS rules

Use removed-hand count relative to current hand size:

- remove none -> `switch_none`
- remove all -> `switch_all`
- remove count <= `switch_minor_max` -> `switch_minor`
- else -> `switch_major`

## How High-Level Connects to Low-Level During Training

The actual bridge is built at runtime in [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py):

1. `ActionLegalityEngine.encode_legal_actions(...)`
   - registers legal low-level specs
   - assigns low-level action codes
   - assigns one high-level code to each low-level code
   - builds:
     - `legal_low_level_codes`
     - `legal_high_level_codes`
     - `high_to_low_map`

2. `aggregate_high_policy(...)`
   - takes a low-level teacher/search policy
   - sums probability mass over the low-level actions belonging to each high-level bucket
   - produces the current high-level training target

So the current system does:

- low-level teacher/search distribution first
- high-level target second

This is why the high-level head is structurally tied to the low-level layer instead of floating independently.

## Why One Low-Level Action Gets One Primary High-Level Label

Current design forces **one primary high-level label per low-level action per state**.

Reason:

- search stays on the low-level executable layer
- legality remains closed
- PPO / teacher / SePoT targets stay aligned

This is a deliberate engineering choice.

It means the current high-level layer is:

- structured
- auditable
- stable

but not yet meant to perfectly capture every fuzzy human intention.

## What To Edit If You Want To Change The Current High-Level Layer

### If you only want to rename categories

Edit:

- [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
- [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)

You must keep names aligned between:

- taxonomy category names
- `action_hierarchy.keys.*`

### If you want to change thresholds but keep the same branch structure

Edit:

- [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)

Specifically:

- `[action_hierarchy.thresholds]`
- `[action_hierarchy.action_card_thresholds]`
- `[action_hierarchy.select_card_thresholds]`

### If you want to change the actual classification logic

Edit:

- [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)

Function to edit:

- `_high_level_key_for_spec(...)`

This is the place where:

- branch order
- visible-state conditions
- label assignment logic

are defined.

## Recommendation: Keep Current High-Level Layer Mechanistic

Right now the safest role for the active high-level layer is:

- mechanism-oriented grouping
- one primary bucket per low-level action
- execution-safe mapping

Examples:

- `skill_commit`
- `card_resource`
- `switch_tempo`
- `tune_fix`

This is more stable than directly making the active high-level layer be:

- `保命`
- `斩杀`
- `骗反应`
- `压节奏`

because those are often fuzzy and overlapping.

## Draft: Extra Intent Layer (Auxiliary Only, Not Active)

The following idea is **not active in the current runtime or training path**.

It is a draft for future auxiliary supervision only.

### Intended role

Use a separate multi-label intent layer for fuzzy abstract tags such as:

- `保命`
- `斩杀`
- `铺场`
- `资源修复`
- `过牌/检索`
- `节奏切人`
- `能量准备`
- `附着/反应准备`

### Important boundary

This intent layer should:

- help supervision
- help analysis
- possibly help search priors later

but it should **not** replace the current executable low-level action space.

### Suggested label source

If added later, the best source is **soft labels**, not hard single labels.

Recommended sources:

- SePoT rollout deltas
- teacher search deltas
- critic value deltas

Examples:

- `保命` score:
  - how much the action improves future survival value
- `斩杀` score:
  - how much the action improves kill probability or lethal line
- `资源修复` score:
  - how much the action improves future legal high-value action coverage

### Recommended form

If added later:

- use **multi-label** or multi-score intent targets
- do **not** force one exclusive intent label
- keep it auxiliary

### Current status

- not wired into runtime
- not wired into active PPO objective
- safe to discuss and hand-edit later without blocking the current mainline

## Practical Edit Workflow

If you want to revise the active high-level layer later, use this order:

1. edit category names in [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
2. edit thresholds and label keys in [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)
3. edit `_high_level_key_for_spec(...)` in [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
4. rerun action-hierarchy and pipeline tests

This keeps the system understandable for:

- humans
- future edits
- other coding agents
