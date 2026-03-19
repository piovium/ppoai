# Action Hierarchy Guide

## Purpose

This file documents the **currently active** high-level to low-level action
bridge for the new codebook-first pipeline:

- `PPO`
- `Oracle Guiding`
- `SePoT`
- `P2SRO`
- hierarchical action layer

It answers four practical questions:

1. `当前生效的高级语义是什么？`
2. `它们怎么和低级可执行动作接起来？`
3. `当前 ACTION 高级层是按什么标准判断的？`
4. `以后如果要改，我应该改哪里？`

## Current Files That Matter

- Category names:
  - [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
- Active ACTION result-semantics rules:
  - [action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_semantic_rules.toml)
- Active non-ACTION functional rules:
  - [non_action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/non_action_semantic_rules.toml)
- Auxiliary intent draft labels:
  - [intent_supervision.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/intent_supervision.toml)
- Runtime mapping logic:
  - [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
- Runtime ACTION result relabel path:
  - [action_adapter.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_adapter.py)
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

- `chip_frontline_hp`
- `chip_backline_hp`
- `force_frontline_lethal`
- `fortify_frontline`
- `fortify_team`
- `switch_character`
- `cycle_for_cards`
- `yield_initiative`
- `burst_setup`
- `invest_tech`
- `invest_delayed_damage`
- `invest_engine_card`
- `elemental_tuning`
- `other_action`

### REROLL_DICE

- `keep_all`
- `keep_active_and_two_backline`
- `keep_active_and_one_backline`
- `keep_active_only`
- `keep_two_backline`
- `keep_one_backline`
- `reroll_active_and_two_backline`
- `reroll_active_and_one_backline`
- `reroll_active_only`
- `reroll_two_backline`
- `reroll_one_backline`

### CHOOSE_ACTIVE

- `choose_attack`
- `choose_defense`
- `choose_heal`
- `choose_absorb`
- `choose_support`
- `choose_other`

### SELECT_CARD

- `select_resource`
- `select_draw_search`
- `select_defense_heal`
- `select_buff_equip`
- `select_board_setup`
- `select_engine_card`
- `select_damage_pressure`
- `select_core`
- `select_other`

### SWITCH_HANDS

- `switch_keep_all`
- `switch_resource`
- `switch_draw_search`
- `switch_defense_heal`
- `switch_buff_equip`
- `switch_board_setup`
- `switch_engine_card`
- `switch_damage_pressure`
- `switch_core`
- `switch_mixed`
- `switch_other`

## Current ACTION Mapping Rules

There are now **two layers** of ACTION mapping:

1. a provisional fallback in `_high_level_key_for_spec(...)`
2. the active runtime result-based relabel in [action_adapter.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_adapter.py)

The authoritative ACTION semantics are defined by:

- [action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_semantic_rules.toml)
- `classify_action_outcome_key(...)` in [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)

### Runtime flow

For `ACTION` requests, the active path is:

1. enumerate legal low-level actions
2. build the low-level codebook view
3. for each legal ACTION low-level action, simulate one step from the current
   root state
4. read the pre/post state delta
5. assign one primary ACTION high-level label
6. rebuild `legal_high_level_codes` and `high_to_low_map`

So the active ACTION high-level layer is now **result-based**, not mainly
definition-id threshold based.

### Delta features currently used

- opponent frontline HP loss
- opponent backline HP loss total
- opponent frontline lethal
- self frontline tank delta
- self team tank delta
- self team HP gain count
- self frontline energy delta
- self burst-ready delta
- self hand-count delta
- self support delta
- self summon delta
- self durable-status delta
- future damage asset delta

### ACTION labels and how they are recognized

- `force_frontline_lethal`
  - opponent frontline changes from alive to defeated
  - this has higher priority than ordinary frontline chip
- `chip_frontline_hp`
  - opponent frontline loses HP
- `chip_backline_hp`
  - opponent backline characters lose HP
- `fortify_team`
  - self team tankiness rises and at least two characters gain HP
- `fortify_frontline`
  - self frontline tankiness rises
- `switch_character`
  - `ACTION_SWITCH_ACTIVE`
- `cycle_for_cards`
  - own hand count rises, or the played card is in the editable cycling card list
- `yield_initiative`
  - currently mainly `ACTION_DECLARE_END`
- `burst_setup`
  - self frontline energy rises, or burst-ready count rises, or card is in the editable burst-setup card list
- `invest_delayed_damage`
  - future damage assets rise, or card is in the editable delayed-damage card list
- `invest_engine_card`
  - card is in the editable engine-card list
- `invest_tech`
  - card is in the editable tech-card list, or support / durable setup rises
- `elemental_tuning`
  - `ACTION_ELEMENTAL_TUNING`
- `other_action`
  - narrow fallback only when nothing else matches

Note:

- `ACTION_DECLARE_END`, `ACTION_SWITCH_ACTIVE`, and `ACTION_ELEMENTAL_TUNING`
  still have real downstream game consequences.
- The current implementation skips the extra one-step relabel simulation only
  because, under the **current ACTION taxonomy**, their primary high-level label
  is defined directly by the action type itself:
  - `ACTION_DECLARE_END` -> `yield_initiative`
  - `ACTION_SWITCH_ACTIVE` -> `switch_character`
  - `ACTION_ELEMENTAL_TUNING` -> `elemental_tuning`
- This is an implementation shortcut for the current label system, not a claim
  that those actions have no state impact.

### Editable card / entity lists

The subjective card-group parts are **not** hidden in Python. They live in:

- [action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_semantic_rules.toml)

Current editable groups include:

- `cycle_card_ids`
- `burst_setup_card_ids`
- `engine_card_ids`
- `tech_card_ids`
- `delayed_damage_card_ids`
- `defensive_entity_definition_ids`
- `offensive_entity_definition_ids`

If you disagree with any current card-to-category seed, edit this file first.

## Current Non-ACTION Mapping Rules

Non-`ACTION` request types now use the request-specific functional rule path in
`_high_level_key_for_spec(...)` inside [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py),
backed by:

- [non_action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/non_action_semantic_rules.toml)

### REROLL_DICE rules

Use current visible dice plus the current team color profile:

- `keep_all`
  - reroll mask is empty
- the remaining 10 labels are chosen by comparing:
  - which team-color dice are mostly kept
  - which team-color dice are mostly rerolled
  - whether the kept/rerolled focus is:
    - active only
    - active + one backline color
    - active + two backline colors
    - one backline color
    - two backline colors
- backline order is ignored
- duplicate backline colors collapse to distinct color classes
- off-team / omni dice are treated as "other colors" and help break ties
- this keeps the semantics close to:
  - 保留哪些己方角色元素色
  - 或者反过来换掉哪些己方角色元素色

### CHOOSE_ACTIVE rules

Look up the chosen character definition id in the editable role groups:

- `choose_attack`
- `choose_defense`
- `choose_heal`
- `choose_absorb`
- `choose_support`
- otherwise -> `choose_other`

These role groups are explicit and editable in
[non_action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/non_action_semantic_rules.toml).

### SELECT_CARD rules

Use the selected card definition id and map it to an explicit card-function
group:

- `select_resource`
- `select_draw_search`
- `select_defense_heal`
- `select_buff_equip`
- `select_board_setup`
- `select_engine_card`
- `select_damage_pressure`
- `select_core`
- otherwise -> `select_other`

### SWITCH_HANDS rules

Use the removed hand cards' function groups:

- remove none -> `switch_keep_all`
- all removed cards in one function group -> `switch_<group>`
- removed cards span multiple groups -> `switch_mixed`
- all removed cards unknown to the current function table -> `switch_other`

## How High-Level Connects to Low-Level During Training

The bridge is built at runtime in [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py):

1. `ActionLegalityEngine.encode_legal_actions(...)`
   - registers legal low-level specs
   - assigns low-level action codes
   - assigns provisional high-level codes
2. `build_decision_context(...)` in [action_adapter.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_adapter.py)
   - for `ACTION`, optionally relabels high-level codes using one-step outcome deltas
   - rebuilds `high_to_low_map`
3. `aggregate_high_policy(...)`
   - takes a low-level teacher/search policy
   - sums probability mass over the low-level actions belonging to each high-level bucket
   - produces the current high-level training target

So the current system still does:

- low-level teacher/search distribution first
- high-level target second

This is why the high-level head is structurally tied to the low-level layer
instead of floating independently.

## Why One Low-Level Action Gets One Primary High-Level Label

Current design forces **one primary high-level label per low-level action per state**.

Reason:

- search stays on the low-level executable layer
- legality remains closed
- PPO / teacher / SePoT targets stay aligned

This is a deliberate engineering choice.

It means the current active high-level layer is:

- structured
- auditable
- state-aware

but still not meant to perfectly capture every fuzzy human intention.

## Draft: Extra Intent Layer (Auxiliary Only, Not Active)

The following idea is **not active in the current runtime or training path**.

It is a draft for future auxiliary supervision only.

The editable draft labels live in:

- [intent_supervision.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/intent_supervision.toml)

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

### Current status

- taxonomy exists
- intent draft labels exist
- but the numeric soft-label computation is **not implemented yet**

## Practical Edit Workflow

If you want to revise the active ACTION high-level layer later, use this order:

1. edit category names in [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
2. edit result-based rules, priorities, and card/entity lists in [action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_semantic_rules.toml)
3. edit `classify_action_outcome_key(...)` and related feature code in [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
4. rerun action-hierarchy and pipeline tests

If you want to revise the active non-`ACTION` high-level layer later, use:

1. edit category names in [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
2. edit role / card-function / reroll-color rules in [non_action_semantic_rules.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/non_action_semantic_rules.toml)
3. edit the request-specific rule functions in [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
4. rerun action-hierarchy tests

If you want to revise the auxiliary fuzzy intent layer later, edit:

- [intent_supervision.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/intent_supervision.toml)

This keeps the system understandable for:

- humans
- future edits
- other coding agents
