# Genius Invokation TCG (Python binding)

This Python binding of GI-TCG is based on the C binding and Python `cffi`.

It is the best external integration path in this repository for AI agents, scripted self-play, and later cloud training workflows.

## Build From This Repository

The Python package depends on the native C binding. Build that first:

```sh
cd ../cbinding
cmake -B build -G Ninja
cmake --build build --config Release
cmake --install build --config Release --prefix install
```

Then build or install the Python package from `packages/pybinding`. The build step generates the `cffi` shim from the C header and bundles the native library into the wheel:

```sh
uv build
```

For local smoke tests, install the generated wheel from `dist/` into a virtual environment and then run the demo at [`examples/agent_vs_agent.py`](./examples/agent_vs_agent.py).

A very simple usage example:

```py
from gitcg import Deck, Player, Game, CreateParam

# Set players initial deck
DECK0 = Deck(characters=[1411, 1510, 2103], cards=[214111, 311503, ...])
DECK1 = Deck(characters=[1609, 2203, 1608], cards=[312025, 321002, ...])

class MyPlayer(Player):
    # implements on_notify, on_action, etc.
    # See `gitcg.Player`'s documentation for detail.
    pass

# Initialize the game
game = Game(create_param=CreateParam(deck0=DECK0, deck1=DECK1))
game.set_player(0, MyPlayer())
game.set_player(1, MyPlayer())

# Start and step the game until end
game.start()
while game.is_running():
    game.step()
```

## Public Surface

The Python wrapper is intentionally small:

- `Deck` and `CreateParam` build initial states
- `State` exposes JSON export, entity query, dice access, and round helpers
- `Game` exposes start/step/status/state/winner operations
- `Player` is the typed callback interface for agent logic
- `Entity` exposes `id`, `definition_id`, and numeric variables
- `low_level` exposes the C-level binding directly when the high-level wrapper is too small

The bindings expose simulator state, ids, variables, JSON, and protobuf IO. Rich card metadata still belongs to the data layer in `packages/data`.

## Demo

[`examples/agent_vs_agent.py`](./examples/agent_vs_agent.py) is the first deterministic Python-first milestone:

- fixed decks
- fixed seed and no-shuffle setup
- scripted agent vs scripted agent
- final-state JSON export
- entity query examples

It is shaped to be reused later as the base of a self-play environment instead of a one-off script.

## Threading Note

Importing `gitcg` on the main thread initializes the native library automatically. If you run games from a non-main thread, call `gitcg.thread_initialize()` before using the binding and `gitcg.thread_cleanup()` when the thread exits.

Here is [a generated API documentation](https://pybinding.gi-tcg.guyutongxue.site).
