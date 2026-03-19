# Semantic Priors Inventory

For the active high-level action categories, mapping rules, and future auxiliary intent-layer notes, see:

- [ACTION_HIERARCHY_GUIDE.md](/E:/Coding/WorldModel/research/world_model/ACTION_HIERARCHY_GUIDE.md)

This file lists the current **hand-authored / subjective semantic priors** that
still exist in the active mainline. The goal is to make them reviewable instead
of hiding them across Python files.

## Already Externalized

### Hierarchical Action Taxonomy

- File:
  - [action_taxonomy.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_taxonomy.toml)
- Used by:
  - [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
- Role:
  - defines the current high-level action category names
  - drives the high-level action vocabulary
- Status:
  - active mainline dependency
  - human-reviewable and editable

### Shared Semantic Thresholds / Prior Knobs

- File:
  - [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)
- Used by:
  - [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
  - [action_quality.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_quality.py)
  - [agents.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/agents.py)
  - [bootstrap_agent.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/bootstrap_agent.py)
- Role:
  - heuristic priority order
  - bootstrap priority order
  - action-quality severities
  - productive-action kind list
  - high-level action mapping thresholds
- Status:
  - active mainline dependency
  - hand-authored prior
  - now externalized for review

### Belief Group Definitions

- File:
  - [belief_groups.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/belief_groups.toml)
- Used by:
  - [belief_groups.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/belief_groups.py)
  - [ppo_features.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/ppo_features.py)
- Role:
  - groups hidden cards into semantic buckets for belief supervision
- Status:
  - active mainline dependency
  - hand-authored semantic prior
  - now externalized for review

## Still Hand-Authored In Code

### High-Level Action Mapping Logic

- File:
  - [action_hierarchy.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_hierarchy.py)
- Main function:
  - `_high_level_key_for_spec(...)`
- Role:
  - maps a concrete low-level executable action plus current visible state to a
    single high-level category key
- Why it is still partly subjective:
  - the category names and numeric cutoffs are now externalized
  - but the control-flow shape of the mapping is still coded in Python
- Status:
  - active mainline dependency
  - partially externalized

### Action-Quality Logic

- File:
  - [action_quality.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/action_quality.py)
- Role:
  - defines what counts as a "productive" non-end action
  - defines early-end severity and tuning-follow-up severity
- Why it is still partly subjective:
  - the value knobs and productive-kind list are externalized
  - the control-flow logic is still coded in Python
- Status:
  - active mainline dependency
  - partially externalized

### Baseline / Bootstrap Rule Agents

- Files:
  - [agents.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/agents.py)
  - [bootstrap_agent.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/bootstrap_agent.py)
- Role:
  - define heuristic and scripted baseline behavior
- Why it is subjective:
  - action priority order is human-authored
- Status:
  - not the main PPO/SePoT policy
  - still used for bootstrap and baselines
  - priority tables are now externalized in [semantic_priors.toml](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/semantic_priors.toml)

## Not Semantic Priors (Even If They Use Labels)

### Opponent Identity Tags

- File:
  - [opponent_identity.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/opponent_identity.py)
- Labels like:
  - `active_population_slot`
  - `frozen_population`
  - `meta_population`
  - `baseline_random`
  - `baseline_heuristic`
  - `baseline_scripted`
- Status:
  - still metadata labels
  - not semantic card/gameplay priors in the same sense as belief groups or
    action taxonomy

## Removed Legacy Files

- Removed old world-model file:
  - `event_model.py`
- Removed old event/gate stack:
  - `event_pipeline.py`
  - `random_gate.py`

## Naming Note

Some remaining filenames still contain old wording but are part of the current
mainline:

- [event_rollout.py](/E:/Coding/WorldModel/research/world_model/src/gitcg_world_model/event_rollout.py)

This file is currently a live PPO episode rollout helper, not the removed
legacy event-model algorithm.
