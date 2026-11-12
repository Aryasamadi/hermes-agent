"""``PluginContext.ask_choice``: a plugin slash command asks its user to pick one option.

The surface that dispatches the command binds an asker around the handler call (:func:`bind_asker`).
Each asker is that surface's existing clarify prompt, so no surface grows a new kind of card:

* a messaging chat: ``adapter.send_clarify`` (buttons on Telegram, Discord and Slack, the numbered
  list answered by a typed reply elsewhere): ``gateway/plugin_command_choice.py``;
* a local session on the gateway (classic CLI, Ink TUI, Desktop): the ``clarify`` server request to
  its attached viewers (same module);
* the in-process classic CLI and the legacy ``hermes serve`` backend: their own clarify picker.

An asker takes ``(question, labels)`` and returns the label the user picked, or None. Passing
``labels=None`` asks for free text through the same clarify UI. With no asker bound (a hook, a cron
job, or a handler on an event loop) the call returns None.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

Asker = Callable[[str, Optional[list]], Optional[str]]
Choice = str | dict

_ASKER: ContextVar[Optional[Asker]] = ContextVar("plugin_choice_asker", default=None)


@contextmanager
def bind_asker(asker: Asker) -> Iterator[None]:
    """Make *asker* answer ``ask_choice`` for the calls in this context (and the executor thread a
    sync handler runs on: callers carry the context over)."""
    token = _ASKER.set(asker)
    try:
        yield
    finally:
        _ASKER.reset(token)


def asker_bound() -> bool:
    return _ASKER.get() is not None


def clarify_asker(callback: Callable[[list], dict]) -> Asker:
    """An asker over a clarify-tool callback (``questions -> {answers: {qid: raw}, outcome}``): the
    classic CLI's clarify panel, the legacy backend's ``clarify`` server request."""
    def ask(question: str, labels: Optional[list]) -> Optional[str]:
        reply = callback([{"qid": "q0", "question": question, "choices": labels, "multi_select": False}])
        answer = ((reply or {}).get("answers") or {}).get("q0")
        return answer if isinstance(answer, str) else None
    return ask


def _options(choices: list[Choice]) -> list[tuple[str, str, str]]:
    """``[(value, plain label, label shown)]``: a string is all three; a dict's description rides on the shown
    label only, so a typed plain label still picks it."""
    options = []
    for choice in choices:
        if isinstance(choice, str):
            options.append((choice, choice, choice))
            continue
        value = str(choice["value"])
        label = str(choice.get("label") or value)
        description = str(choice.get("description") or "").strip()
        options.append((value, label, f"{label} — {description}" if description else label))
    return options


def _running_on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True

 
def ask_text(question: str) -> Optional[str]:
    """Ask the current slash-command user for free text via the same clarify prompt."""
    asker = _ASKER.get()
    if asker is None:
        return None
    if _running_on_loop():
        logger.warning("ask_text needs a sync slash-command handler: returning None")
        return None
    answer = asker(str(question), None)
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    return answer or None



def ask_choice(question: str, choices: list[Choice]) -> Optional[str]:
    """Ask the user who ran the current slash command to pick one of *choices*; the picked choice's
    value, or None when nothing can ask here, the user cancels or types something else, or the
    clarify timeout passes. A typed value or label (any case) also counts as that choice."""
    asker = _ASKER.get()
    options = _options(choices)
    if asker is None or not options:
        return None
    if _running_on_loop():
        logger.warning("ask_choice needs a sync slash-command handler: an async one would block the "
                       "event loop that delivers the answer; returning None")
        return None
    picked = asker(str(question), [shown for _value, _label, shown in options])
    typed = (picked or "").strip().casefold()
    return next((value for value, label, shown in options
                 if typed and typed in (shown.casefold(), label.casefold(), value.casefold())), None)
