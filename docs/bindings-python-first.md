# Python-First Binding Guide

This guide maps the repository from the perspective of external bindings and the first Python-based simulator milestone.

## Repository Map

The repository is easiest to work with as four layers:

1. **Core and data**
   - `packages/core`: game flow, pause points, action resolution, state transitions
   - `packages/data`: official card and character definition data
2. **Language bindings**
   - `packages/cbinding`: native C API over the simulator and official data
   - `packages/pybinding`: Python wrapper over the C API using `cffi`
   - `packages/csbinding`: C# wrapper over the C API with protobuf-based callbacks
3. **Local simulator and UI**
   - `packages/standalone`: local browser simulator/debug UI
   - `packages/web-ui` and `packages/web-ui-core`: embeddable visualization and interaction UI
4. **Battle platform**
   - `packages/server`: backend for online battles
   - `packages/web-client`: battle-platform frontend

Use the bindings for programmatic control and AI integration. Use the UI packages when the goal is human-facing play, visualization, or browser interaction.

## Binding Comparison

The C header at [`packages/cbinding/include/gitcg/gitcg.h`](../packages/cbinding/include/gitcg/gitcg.h) is the native source of truth for lifecycle and exposed capabilities.

| Binding | Main role | Build/install path | Runtime model | Callback model | Current maturity |
| --- | --- | --- | --- | --- | --- |
| C | Native truth source and lowest-level integration surface | In `packages/cbinding`: `cmake -B build -G Ninja`, `cmake --build build --config Release`, `cmake --install build --config Release --prefix install` | Explicit lifecycle: global init, thread init, create state/create game, set handlers, step, cleanup | Raw protobuf bytes delivered through function pointers | Most authoritative |
| Python | Best fit for AI agents, self-play, cloud training, and experimentation | Build/install C binding first, then package `packages/pybinding` so the wheel can generate the `cffi` shim and bundle `libgitcg` | High-level objects over the C API: `CreateParam`, `State`, `Game`, `Player`, plus `low_level` escape hatch | `Player` methods receive generated protobuf message classes | Best path for first external automation |
| C# | .NET wrapper for the same native core | Build/install C binding, generate `NativeMethods.cs` with `packages/csbinding-gen`, then `dotnet build packages/csbinding -c Release` | Managed wrappers around native handles: `GiTcg`, `CreateParam`, `State`, `Game`, `IPlayer` | `IPlayer` methods receive generated protobuf message classes | Secondary parity path; useful, but less mature than Python |

## What The Bindings Actually Expose

All three bindings center on simulator control, not rich card-database browsing.

They expose:

- state creation from deck definitions
- game stepping and winner/status inspection
- state serialization to JSON
- entity queries and entity variables
- protobuf request/response and notification traffic for player IO

They do **not** directly act as the main source for rich card text or detailed definition metadata. For that, use `packages/data` and the development docs under [`docs/development/data`](./development/data/README.md).

At binding level, “reading cards” usually means reading card entity ids or definition ids from state/query results, then resolving those ids in the data layer if card text or static metadata is needed.

## Public Surface To Learn First

### C binding

Read the exported functions in [`packages/cbinding/include/gitcg/gitcg.h`](../packages/cbinding/include/gitcg/gitcg.h) in this order:

1. `gitcg_initialize`, `gitcg_thread_initialize`, cleanup counterparts
2. `gitcg_state_createparam_*`
3. `gitcg_state_*`
4. `gitcg_game_*`
5. `gitcg_state_query` and `gitcg_entity_*`

The native lifecycle is:

1. initialize library
2. initialize the current thread
3. build a `createparam`
4. create a `state`
5. create a `game`
6. install player handlers
7. call `gitcg_game_step` until status is no longer running
8. inspect/export/query final state
9. free objects and clean up thread/library state

### Python binding

The Python wrapper surface is concentrated in:

- `Game`: start, step, status, winner, current state
- `Player`: callback interface for all player decisions and notifications
- `State`: JSON export, entity query, dice and round helpers
- `CreateParam` and `Deck`: initial game construction
- `Entity`: `id`, `definition_id`, and numeric variables
- `low_level`: direct C-level access when the high-level wrapper is not enough

The first demo lives at [`packages/pybinding/examples/agent_vs_agent.py`](../packages/pybinding/examples/agent_vs_agent.py). It is deterministic, uses fixed decks, exports final state JSON, and demonstrates entity query results in a form that can be reused for later self-play tooling.

### C# binding

The public wrapper surface is concentrated in:

- `GiTcg`: library version access
- `CreateParam`: deck construction
- `State`: JSON round-trip and state handle wrapper
- `Game`: player attachment, stepping, state access
- `IPlayer`: protobuf callback interface

The sample app is [`packages/csbinding/GiTcgTest/Program.cs`](../packages/csbinding/GiTcgTest/Program.cs).

## First Python Milestone

The first milestone is intentionally narrow:

- run a deterministic scripted agent vs scripted agent game
- keep both agents programmatic, not human-facing
- prove the full lifecycle from decks to final winner
- inspect both structured game output and queried entities

This is the right starting point for a world-model or self-play environment because it validates the simulator boundary before adding training loops, model adapters, or UI.

## Training Direction

Python should be treated as the main environment API for later AI work.

Use:

- full state JSON for internal rollouts, replay, and world-model state capture
- player notifications for partial-observation training signals
- action requests as the legal-action frontier

A later adapter should:

1. convert protobuf action requests into a model-friendly action space
2. keep enough metadata to reconstruct the exact chosen action
3. convert the selected action back into a legal protobuf response

Training infrastructure, cloud execution, and human-vs-AI UI should stay out of scope until the environment wrapper is stable and reproducible.
