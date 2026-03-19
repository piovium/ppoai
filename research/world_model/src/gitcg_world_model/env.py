from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from queue import Queue
from threading import Event, Thread
from typing import Any

from .action_adapter import BuiltDecisionContext, build_decision_context, encode_choice
from .decks import deck_by_name
from .schema import ActionChoice, DecisionContext, DecisionType, EnvConfig, EpisodeRecord, Matchup, TrajectoryStep
from .snapshot import snapshot_notification, snapshot_state


def _safe_release(resource: Any) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()
        return
    finalize = getattr(resource, "_finalize", None)
    if callable(finalize):
        finalize()
        return
    handle_specs = (
        ("_createparam_handle", "state_createpram_free"),
        ("_state_handle", "state_free"),
        ("_game_handle", "game_free"),
    )
    for handle_name, free_name in handle_specs:
        if not hasattr(resource, handle_name):
            continue
        try:
            module_root = resource.__class__.__module__.split(".", maxsplit=1)[0]
            low_level = import_module(f"{module_root}.low_level")
            handle = getattr(resource, handle_name)
            null_handle = getattr(low_level, "NULL", None)
            if handle is None or handle == null_handle:
                return
            getattr(low_level, free_name)(handle)
            if null_handle is not None:
                setattr(resource, handle_name, null_handle)
            return
        except Exception:
            return


def _patch_gitcg_instance_callback_storage(gitcg: Any) -> None:
    try:
        game_module = import_module(f"{gitcg.__name__}.game")
        low_level = import_module(f"{gitcg.__name__}.low_level")
    except Exception:
        return
    game_cls = getattr(game_module, "Game", None)
    callback_cls = getattr(game_module, "_GameCallback", None)
    if game_cls is None or callback_cls is None:
        return
    if getattr(game_cls, "_world_model_instance_callback_patch", False):
        return

    def _instance_safe_set_player(self, who: int, player: Any):
        assert who == 0 or who == 1
        callback = callback_cls(player)
        if not hasattr(self, "_wm_players"):
            self._wm_players = [None, None]
        if not hasattr(self, "_wm_player_callbacks"):
            self._wm_player_callbacks = [None, None]
        if not hasattr(self, "_wm_player_callback_handles"):
            self._wm_player_callback_handles = [None, None]
        self._wm_player_callbacks[who] = callback
        handle = low_level.game_set_handlers(self._game_handle, who, callback)
        self._wm_player_callback_handles[who] = handle
        self._wm_players[who] = player

    game_cls.set_player = _instance_safe_set_player
    game_cls._world_model_instance_callback_patch = True


class _EnvironmentClosed(RuntimeError):
    pass


class _Shutdown:
    pass


@dataclass(frozen=True)
class _DecisionEvent:
    context: DecisionContext


@dataclass(frozen=True)
class _TerminalEvent:
    context: DecisionContext


@dataclass(frozen=True)
class _CrashEvent:
    error: BaseException


class GitcgDecisionEnv:
    def __init__(self, config: EnvConfig, matchup: Matchup, *, initial_state_json: str | None = None):
        self._config = config
        self._matchup = matchup
        self._initial_state_json = initial_state_json
        self._event_queue: Queue[Any] = Queue()
        self._response_queue: Queue[Any] = Queue()
        self._worker: Thread | None = None
        self._shutdown_requested = Event()
        self._current_context: DecisionContext | None = None
        self._current_built_context: BuiltDecisionContext | None = None
        self._terminal_context: DecisionContext | None = None
        self._steps: list[TrajectoryStep] = []
        self._seed: int | None = None

    def reset(self, seed: int | None = None, *, state_json: str | None = None) -> DecisionContext:
        self.close()
        self._event_queue = Queue()
        self._response_queue = Queue()
        self._shutdown_requested.clear()
        self._current_context = None
        self._current_built_context = None
        self._terminal_context = None
        self._steps = []
        self._seed = seed if seed is not None else self._config.seed
        initial_state_json = state_json if state_json is not None else self._initial_state_json
        self._worker = Thread(
            target=self._run_worker,
            args=(self._seed, initial_state_json),
            daemon=True,
            name=f"gitcg-world-model-{self._matchup.key}",
        )
        self._worker.start()
        event = self._next_event()
        if isinstance(event, _DecisionEvent):
            self._current_context = event.context
            return event.context
        if isinstance(event, _TerminalEvent):
            self._terminal_context = event.context
            raise RuntimeError("game terminated before producing a decision context")
        raise RuntimeError(f"unexpected event on reset: {type(event).__name__}")

    def step(self, choice: ActionChoice) -> tuple[DecisionContext | None, TrajectoryStep, bool]:
        if self._current_context is None or self._current_context.terminal:
            raise RuntimeError("environment is not awaiting an action")
        chosen_action_code = int(choice.action_code)
        if chosen_action_code < 0:
            chosen_action_code = _fallback_action_code(self._current_context)
        self._response_queue.put(chosen_action_code)
        event = self._next_event()
        if isinstance(event, _DecisionEvent):
            next_context = event.context
            done = False
        elif isinstance(event, _TerminalEvent):
            next_context = event.context
            self._terminal_context = next_context
            done = True
        else:
            raise RuntimeError(f"unexpected event on step: {type(event).__name__}")
        reward = (
            _terminal_reward(
                next_context.full_state.winner,
                self._current_context.acting_player,
                draw_penalty=self._config.draw_penalty,
            )
            if done
            else 0.0
        )
        terminal_context = replace(next_context, reward=reward)
        normalized_choice = ActionChoice(action_code=chosen_action_code)
        trajectory_step = TrajectoryStep(
            acting_player=self._current_context.acting_player,
            request_type=self._current_context.request_type,
            choice=normalized_choice,
            pre_state=self._current_context.full_state,
            post_state=terminal_context.full_state,
            reward=reward,
            done=done,
            legal_low_level_codes=self._current_context.legal_low_level_codes,
            legal_low_level_mask=self._current_context.legal_low_level_mask,
            legal_high_level_codes=self._current_context.legal_high_level_codes,
            high_to_low_map=self._current_context.high_to_low_map,
            chosen_high_level_code=(
                self._current_built_context.low_to_high_code.get(chosen_action_code)
                if self._current_built_context is not None
                else None
            ),
            player_view=self._current_context.player_view,
            full_state_json_before=self._current_context.full_state_json,
            full_state_json_after=terminal_context.full_state_json,
            metadata={
                "matchup": self._matchup.key,
                "action_label": _action_label(self._current_built_context, chosen_action_code),
                "chosen_action_code": chosen_action_code,
                **terminal_context.metadata,
            },
        )
        self._steps.append(trajectory_step)
        self._current_context = None if done else terminal_context
        self._current_built_context = None if done else self._current_built_context
        if done:
            self._terminal_context = terminal_context
            return None, trajectory_step, True
        return terminal_context, trajectory_step, False

    def episode_record(self) -> EpisodeRecord:
        if self._terminal_context is None:
            raise RuntimeError("episode is not complete")
        return EpisodeRecord(
            matchup=self._matchup.key,
            seed=self._seed,
            winner=self._terminal_context.full_state.winner,
            steps=tuple(self._steps),
            final_state=self._terminal_context.full_state,
            final_state_json=self._terminal_context.full_state_json,
            metadata=self._terminal_context.metadata,
        )

    def partial_episode_record(self, *, reason: str | None = None) -> EpisodeRecord:
        final_context = self._terminal_context or self._current_context
        if final_context is None:
            raise RuntimeError("episode has not started")
        metadata = dict(final_context.metadata)
        if reason is not None:
            metadata["truncated"] = True
            metadata["truncation_reason"] = reason
            metadata["terminal_value"] = 0.0
        return EpisodeRecord(
            matchup=self._matchup.key,
            seed=self._seed,
            winner=final_context.full_state.winner,
            steps=tuple(self._steps),
            final_state=final_context.full_state,
            final_state_json=final_context.full_state_json,
            metadata=metadata,
        )

    def close(self):
        if self._worker is None:
            return
        if self._terminal_context is not None:
            self._current_built_context = None
            self._worker = None
            return
        if not self._worker.is_alive():
            self._current_built_context = None
            self._worker = None
            return
        if self._worker.is_alive():
            self._shutdown_requested.set()
            self._response_queue.put(_Shutdown())
            self._worker.join(timeout=5)
        self._current_built_context = None
        self._worker = None

    def _next_event(self) -> _DecisionEvent | _TerminalEvent:
        event = self._event_queue.get()
        if isinstance(event, _CrashEvent):
            raise RuntimeError("world-model worker failed") from event.error
        return event

    def _run_worker(self, seed: int | None, initial_state_json: str | None):
        try:
            self._run_worker_inner(seed, initial_state_json)
        except _EnvironmentClosed:
            pass
        except BaseException as exc:
            self._event_queue.put(_CrashEvent(exc))

    def _run_worker_inner(self, seed: int | None, initial_state_json: str | None):
        import gitcg

        gitcg.thread_initialize()
        _patch_gitcg_instance_callback_storage(gitcg)
        create_param = None
        state = None
        game = None
        final_state_ref = None
        try:
            if initial_state_json is None:
                create_param = _build_create_param(gitcg, self._config, self._matchup, seed)
                state = gitcg.State(create_param=create_param)
            else:
                state = gitcg.State(json=initial_state_json)
            game = gitcg.Game(state=state)
            step_index_ref = {"value": 0}

            env = self

            class BridgePlayer(gitcg.Player):
                def __init__(self, who: int):
                    self.who = who
                    self.last_notification = None
                    self.last_error: str | None = None

                def on_notify(self, notification):
                    self.last_notification = notification

                def on_io_error(self, error_msg: str):
                    self.last_error = error_msg

                def on_choose_active(self, request):
                    payload = self._await_payload(DecisionType.CHOOSE_ACTIVE, request)
                    return gitcg.ChooseActiveResponse(**payload)

                def on_reroll_dice(self, request):
                    payload = self._await_payload(DecisionType.REROLL_DICE, request)
                    return gitcg.RerollDiceResponse(**payload)

                def on_select_card(self, request):
                    payload = self._await_payload(DecisionType.SELECT_CARD, request)
                    return gitcg.SelectCardResponse(**payload)

                def on_switch_hands(self, request):
                    payload = self._await_payload(DecisionType.SWITCH_HANDS, request)
                    return gitcg.SwitchHandsResponse(**payload)

                def on_action(self, request):
                    payload = self._await_payload(DecisionType.ACTION, request)
                    return gitcg.ActionResponse(**payload)

                def _await_payload(self, request_type: DecisionType, request: Any) -> dict[str, Any]:
                    state_ref = game.state()
                    try:
                        raw_state_json = state_ref.json()
                        full_state = snapshot_state(state_ref, state_json=raw_state_json)
                    finally:
                        _safe_release(state_ref)
                    full_state_json = raw_state_json if env._config.record_full_state_json else None
                    visible_snapshot = None
                    if self.last_notification is not None:
                        visible_snapshot = snapshot_notification(self.last_notification)
                    built_context = build_decision_context(
                        acting_player=self.who,
                        request_type=request_type,
                        request=request,
                        full_state=full_state,
                        player_view=visible_snapshot,
                        full_state_json=full_state_json,
                        step_index=step_index_ref["value"],
                        metadata={
                            "io_error": self.last_error,
                            "matchup": env._matchup.key,
                            "seed": env._seed,
                        },
                    )
                    context = built_context.context
                    step_index_ref["value"] += 1
                    if not env._config.record_player_view:
                        context = replace(context, player_view=None)
                        built_context = BuiltDecisionContext(
                            context=context,
                            payload_by_low_level_code=built_context.payload_by_low_level_code,
                            label_by_low_level_code=built_context.label_by_low_level_code,
                            low_to_high_code=built_context.low_to_high_code,
                        )
                    env._current_built_context = built_context
                    env._event_queue.put(_DecisionEvent(context))
                    if env._shutdown_requested.is_set():
                        return encode_choice(built_context, _fallback_action_code(context))
                    action_code = env._response_queue.get()
                    if isinstance(action_code, _Shutdown):
                        env._shutdown_requested.set()
                        return encode_choice(built_context, _fallback_action_code(context))
                    return encode_choice(built_context, int(action_code))

            game.set_player(0, BridgePlayer(0))
            game.set_player(1, BridgePlayer(1))
            game.start()
            while game.is_running():
                game.step()
                if self._shutdown_requested.is_set():
                    break
            if self._shutdown_requested.is_set():
                return
            final_state_ref = game.state()
            try:
                raw_state_json = final_state_ref.json()
                final_state = snapshot_state(final_state_ref, state_json=raw_state_json)
                final_state_json = raw_state_json if self._config.record_full_state_json else None
                terminal_context = DecisionContext(
                    acting_player=-1,
                    request_type=DecisionType.TERMINAL,
                    step_index=step_index_ref["value"],
                    full_state=final_state,
                    legal_low_level_codes=(),
                    legal_low_level_mask=(),
                    legal_high_level_codes=(),
                    high_to_low_map=(),
                    player_view=None,
                    full_state_json=final_state_json,
                    terminal=True,
                    metadata={
                        "matchup": self._matchup.key,
                        "seed": self._seed,
                        "status": game.status().name,
                        "winner": game.winner(),
                        "error": game.error() if game.status().name == "ABORTED" else None,
                    },
                )
            finally:
                _safe_release(final_state_ref)
            self._event_queue.put(_TerminalEvent(terminal_context))
        finally:
            if final_state_ref is not None:
                _safe_release(final_state_ref)
            if game is not None:
                _safe_release(game)
            if state is not None:
                _safe_release(state)
            if create_param is not None:
                _safe_release(create_param)
            gitcg.thread_cleanup()


def _build_create_param(gitcg: Any, config: EnvConfig, matchup: Matchup, seed: int | None):
    deck0 = deck_by_name(matchup.deck0)
    deck1 = deck_by_name(matchup.deck1)
    create_param = gitcg.CreateParam(
        deck0=gitcg.Deck(characters=list(deck0.characters), cards=list(deck0.cards)),
        deck1=gitcg.Deck(characters=list(deck1.characters), cards=list(deck1.cards)),
        version=config.version,
    )
    if seed is not None:
        create_param.set_attr(gitcg.low_level.ATTR_STATE_CONFIG_RANDOM_SEED, seed)
    if config.max_rounds is not None:
        create_param.set_attr(gitcg.low_level.ATTR_STATE_CONFIG_MAX_ROUNDS_COUNT, config.max_rounds)
    return create_param


def _terminal_reward(
    winner: int | None,
    acting_player: int,
    *,
    draw_penalty: float = 2.0,
) -> float:
    if acting_player not in (0, 1):
        return 0.0
    if winner is None:
        return -abs(float(draw_penalty))
    return 1.0 if winner == acting_player else -1.0


def _action_label(built_context: BuiltDecisionContext | None, action_code: int) -> str:
    if built_context is None:
        return f"action:{action_code}"
    return str(built_context.label_by_low_level_code.get(int(action_code), f"action:{action_code}"))


def _fallback_action_code(context: DecisionContext) -> int:
    if not context.legal_low_level_codes:
        raise RuntimeError("cannot synthesize fallback choice without legal codes")
    return int(context.legal_low_level_codes[0])
