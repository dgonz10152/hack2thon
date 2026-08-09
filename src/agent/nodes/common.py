"""Shared machinery for the two research fan-outs (judges and ideas).

Both fan-outs run one ReAct agent per item, and both need the same three
protections: a bounded runtime, a way for the user to skip a stuck phase, and
degradation of a failed worker so one blip cannot cost the whole run.
`run_researcher` owns that policy in one place.
"""

import asyncio
import contextlib
import os

from langchain_core.messages import HumanMessage, SystemMessage

from agent.progress import SkippedByUser, _say, skip_requested

# LangGraph's default recursion limit is 25 supersteps, roughly a dozen tool
# calls, which a research agent can legitimately exceed. Raised, but still
# bounded so a genuinely stuck agent fails instead of running forever.
RESEARCH_CONFIG = {"recursion_limit": int(os.getenv("RESEARCH_RECURSION_LIMIT", "40"))}

# Nothing else bounds a model call, so a hung request stalls the phase forever:
# retries and degradation only fire on errors, never on silence.
RESEARCH_TIMEOUT = float(os.getenv("RESEARCH_TIMEOUT", "300"))

# Placeholders written when a worker gives up. Non-empty on purpose: the state
# reducers treat "" as "no update", so an empty string would be
# indistinguishable from research that never ran. They also state plainly that
# nothing was found, so the downstream compiler does not treat silence as a
# finding.
UNAVAILABLE = "({kind} unavailable after retries: {err}. No information gathered.)"
SKIPPED = "({kind} skipped at the user's request. No information gathered.)"


async def _watch_for_skip(poll: float = 0.05) -> None:
    while not skip_requested():
        await asyncio.sleep(poll)


async def bounded_research(coro, timeout: float):
    """Await a research call, but give up on a timeout or a skip request.

    Races the call against a skip watcher so a hung request can be abandoned
    mid-flight, not merely checked before it starts.
    """
    task = asyncio.ensure_future(coro)
    watcher = asyncio.ensure_future(_watch_for_skip())
    try:
        done, _ = await asyncio.wait(
            {task, watcher}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        # Decide from the flag, never from which task landed in `done`: a
        # watcher that errors or is cancelled also lands there, and would then
        # be misreported as a deliberate skip.
        if skip_requested():
            raise SkippedByUser("skipped at your request")
        raise TimeoutError(f"no response within {timeout:.0f}s")
    finally:
        watcher.cancel()
        with contextlib.suppress(BaseException):
            await watcher


async def run_researcher(
    agent, system_prompt: str, user_content: str, semaphore, kind: str, label: str
) -> str:
    """Run one fan-out research worker and return its summary text.

    Failure policy, in order: a skip request returns a "skipped" placeholder,
    any other failure (network, recursion limit, timeout) returns an
    "unavailable" one. One worker out of N must never lose the whole run, and
    every degradation is reported through `_say` so it stays visible.
    """
    try:
        async with semaphore:
            result = await bounded_research(
                agent.ainvoke(
                    {
                        "messages": [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_content),
                        ]
                    },
                    config=RESEARCH_CONFIG,
                ),
                RESEARCH_TIMEOUT,
            )
        return result["messages"][-1].content
    except SkippedByUser:
        _say(f"  - skipped {kind.lower()} for {label}")
        return SKIPPED.format(kind=kind)
    except Exception as exc:
        _say(f"  ! {kind.lower()} failed for {label}: {exc!r} - continuing")
        return UNAVAILABLE.format(kind=kind, err=type(exc).__name__)
