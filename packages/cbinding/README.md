# Genius Invokation TCG (C binding)

`@gi-tcg/cbinding` is the native C entry point for the simulator and official card data.

The public API is defined in [`include/gitcg/gitcg.h`](./include/gitcg/gitcg.h). Treat that header as the source of truth for lifecycle, exported constants, and function names.

## Build From This Repository

From `packages/cbinding`:

```sh
cmake -B build -G Ninja
cmake --build build --config Release
cmake --install build --config Release --prefix install
```

This produces an installed header plus the shared library under `install/`.

## Runtime Lifecycle

The native lifecycle is explicit:

1. `gitcg_initialize()`
2. `gitcg_thread_initialize()`
3. create `gitcg_state_createparam_t`
4. fill decks and optional attributes
5. create `gitcg_state_t`
6. create `gitcg_game_t`
7. set player data and IO handlers
8. call `gitcg_game_step()` until status is no longer running
9. inspect state, JSON, queried entities, winner, or error
10. free objects, clean up thread state, then clean up library state

`gitcg_state_query()` and `gitcg_entity_*()` are the main debugging/introspection tools exposed by the native API.

## IO Model

The C binding uses callbacks:

- `gitcg_rpc_handler`: receives a protobuf request payload and must write a protobuf response payload
- `gitcg_notification_handler`: receives protobuf notifications
- `gitcg_io_error_handler`: receives an error string when the player response is invalid

The protobuf schemas live in [`/proto`](../../proto/README.md).

## Sample

See [`test/main.c`](./test/main.c) for a minimal end-to-end example that:

- initializes the library
- constructs decks and state
- creates a game
- installs handlers
- steps the simulator
- exports final state JSON
- queries entities from the resulting state
