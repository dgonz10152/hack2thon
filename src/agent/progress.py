"""How long-running phases talk to whoever is watching.

Two small channels between the pipeline and the user:
  * `_say` prints progress text.
  * the skip flag lets the user abandon in-flight research workers.
"""


def _say(message: str) -> None:
    """Progress output for long-running phases.

    The build runs for minutes with nothing in state until it finishes, so it
    streams as it goes. The research workers also use this to report a degraded
    worker, which would otherwise be invisible.
    """
    print(message, flush=True)


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
