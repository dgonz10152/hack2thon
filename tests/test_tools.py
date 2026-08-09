"""Self-check for the tools, mainly that a failing web search is survivable.

Hermetic - DuckDuckGo is mocked, nothing hits the network.

Regression guard for a real failure: a DNS error inside search_web propagated
out of the ReAct loop and killed a pipeline that was ~20 minutes in. The tool is
bound into both judge_researcher and deep_researcher, so every judge and every
idea worker calls it; one transient blip anywhere aborted the whole run.

Run:  uv run python tests/test_tools.py
"""

import unittest.mock as mock

from agent.tools import fetch_html, search_web


class DDGSException(Exception):
    """Stands in for the ddgs error class seen in the wild."""


def test_search_failure_is_returned_not_raised():
    failures = [
        DDGSException("ConnectError: dns error > no records found"),
        TimeoutError("timed out"),
        RuntimeError("rate limited"),
    ]
    for exc in failures:
        with mock.patch("agent.tools.DuckDuckGoSearchRun") as fake:
            fake.return_value.run.side_effect = exc
            result = search_web.invoke({"query": "anything"})

        assert isinstance(result, str), result
        assert "SEARCH FAILED" in result, result
        assert type(exc).__name__ in result, result
    print("ok: search failures come back as a tool result instead of raising")


def test_failure_message_bounds_retries_and_forbids_fabrication():
    """The message must not invite an unbounded retry loop.

    An earlier version said "try a different query" with no limit. Under
    DuckDuckGo rate limiting the agent retried forever and blew the graph
    recursion limit, which surfaced as GraphRecursionError per idea.
    """
    with mock.patch("agent.tools.DuckDuckGoSearchRun") as fake:
        fake.return_value.run.side_effect = DDGSException("boom")
        result = search_web.invoke({"query": "x"})

    lowered = result.lower()
    assert "at most once" in lowered, f"retries must be bounded: {result}"
    assert "stop searching" in lowered, result
    assert "do not invent" in lowered, result
    print("ok: the failure message bounds retries and forbids fabrication")


def test_successful_search_is_untouched():
    with mock.patch("agent.tools.DuckDuckGoSearchRun") as fake:
        fake.return_value.run.return_value = "real results here"
        result = search_web.invoke({"query": "x"})

    assert result == "real results here", result
    print("ok: a successful search passes straight through")


def test_fetch_html_still_fails_loudly():
    """Deliberate asymmetry: an unreachable DevPost URL has no graceful path.

    There is nothing to analyse without the page, so this must abort the run
    rather than feed an empty string downstream. The TUI reports it and offers
    the resume command.
    """
    try:
        fetch_html.invoke({"url": "http://127.0.0.1:1/nope"})
    except Exception:
        print("ok: fetch_html still raises, so a bad URL fails fast")
        return
    raise AssertionError("fetch_html should not swallow an unreachable URL")


if __name__ == "__main__":
    test_search_failure_is_returned_not_raised()
    test_failure_message_bounds_retries_and_forbids_fabrication()
    test_successful_search_is_untouched()
    test_fetch_html_still_fails_loudly()
    print("\nAll tool checks passed.")
