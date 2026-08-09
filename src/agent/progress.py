"""How long-running phases talk to whoever is watching.

Two small channels between the pipeline and a UI:
  * `_say` streams progress text (the TUI captures it via `set_say_sink`;
    without a sink it prints, which is what the bare CLI wants).
  * the skip flag lets the user abandon in-flight research workers.
"""

_say_sink = None


def set_say_sink(sink) -> None:
    """Redirect progress output. The TUI captures it; None restores print."""
    global _say_sink
    _say_sink = sink


def _say(message: str) -> None:
    """Progress output for long-running phases.

    The build runs for minutes with nothing in state until it finishes, so it
    streams as it goes. The research workers also use this to report a degraded
    worker, which would otherwise be invisible.
    """
    if _say_sink is None:
        print(message, flush=True)
    else:
        _say_sink(message)


# Set from the UI to abandon in-flight research and let the graph move on.
# A plain flag rather than an asyncio.Event: an Event created at import time
# binds its internal futures to whichever loop first waits on it, which misfires
# across loops. A polled flag has no loop affinity.
_skip = False


class SkippedByUser(Exception):
    """Raised in a research worker when the user asks to move on."""


def request_skip() -> None:
    """Abandon in-flight research workers. Each degrades and the phase ends."""
    global _skip
    _skip = True


def clear_skip() -> None:
    """Re-arm skipping once the phase has moved on."""
    global _skip
    _skip = False


def skip_requested() -> bool:
    return _skip
