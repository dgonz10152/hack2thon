"""Self-check for surviving flaky infrastructure.

Hermetic - no network, no model calls.

Regression guard for two real failures, both of which ended runs that were
already 20+ minutes in:
  * `DDGSException: ConnectError ... dns error` from search_web
  * `Pipeline failed: ReadError:` (empty message) from an Ollama call during
    rank_ideas, on a resumed run

Run:  uv run python tests/test_resilience.py
"""

import asyncio

import httpx
from langchain_core.runnables import RunnableLambda

from agent import nodes
from agent.state import Idea, Judge


def test_transient_errors_are_retried():
    """A dropped connection should be retried, not fatal."""
    attempts = []

    def flaky(_payload):
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadError("")  # empty message, exactly as seen
        return "recovered"

    result = nodes._resilient(RunnableLambda(flaky)).invoke({})
    assert result == "recovered", result
    assert len(attempts) == 3, f"expected 2 retries then success, got {attempts}"
    print("ok: a transient ReadError is retried and can recover")


def test_every_transport_error_is_covered():
    """ReadError was the one seen; siblings must be handled too."""
    for exc_type in (
        httpx.ReadError,
        httpx.ConnectError,
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    ):
        assert issubclass(exc_type, nodes._TRANSIENT), exc_type
    print("ok: all httpx transport failures count as transient")


def test_non_transient_errors_are_not_retried():
    """A bug should surface immediately, not be retried three times."""
    attempts = []

    def broken(_payload):
        attempts.append(1)
        raise ValueError("this is a bug, not a blip")

    try:
        nodes._resilient(RunnableLambda(broken)).invoke({})
    except ValueError:
        assert len(attempts) == 1, f"a ValueError must not be retried: {attempts}"
        print("ok: non-transient errors fail fast")
        return
    raise AssertionError("expected the ValueError to propagate")


class _AlwaysFails:
    def __init__(self, exc):
        self.exc = exc

    async def ainvoke(self, _payload, config=None):
        raise self.exc


def test_one_failed_judge_does_not_kill_the_run():
    original = nodes.judge_researcher
    said: list[str] = []
    nodes.set_say_sink(said.append)
    nodes.judge_researcher = _AlwaysFails(httpx.ReadError(""))
    try:
        out = asyncio.run(
            nodes.research_one_judge_node(
                {
                    "judge": Judge(name="Ada", blurb="b"),
                    "hackathon_synopsis": "syn",
                }
            )
        )
    finally:
        nodes.judge_researcher = original
        nodes.set_say_sink(None)

    judge = out["judges"][0]
    assert judge.name == "Ada", out
    # Non-empty on purpose: merge_judges treats "" as "no update", so an empty
    # placeholder would be indistinguishable from research never running.
    assert judge.online_summary, "placeholder must not be empty"
    assert "unavailable" in judge.online_summary.lower(), judge.online_summary
    assert "ReadError" in judge.online_summary, judge.online_summary
    assert any("Ada" in line for line in said), f"failure must be visible: {said}"
    print("ok: a failed judge degrades to a placeholder and is reported")


def test_one_failed_idea_does_not_kill_the_run():
    original = nodes.idea_researcher
    said: list[str] = []
    nodes.set_say_sink(said.append)
    nodes.idea_researcher = _AlwaysFails(httpx.ConnectError("boom"))
    try:
        out = asyncio.run(
            nodes.research_one_idea_node(
                {
                    "idea": Idea(title="Idea 1", pitch="p"),
                    "hackathon_synopsis": "syn",
                    "judge_bias": "bias",
                }
            )
        )
    finally:
        nodes.idea_researcher = original
        nodes.set_say_sink(None)

    idea = out["ideas"][0]
    assert idea.title == "Idea 1", out
    assert idea.research and "unavailable" in idea.research.lower(), idea.research
    assert any("Idea 1" in line for line in said), f"failure must be visible: {said}"
    print("ok: a failed idea degrades to a placeholder and is reported")


def test_single_nodes_still_propagate():
    """rank_ideas has no peer to degrade to, so it must still fail loudly.

    This is the node that actually broke: a ReadError there means no ranking,
    and inventing one would be worse than stopping.
    """
    original = nodes.idea_ranker

    class Boom:
        def invoke(self, _messages):
            raise httpx.ReadError("")

    nodes.idea_ranker = Boom()
    try:
        nodes.rank_ideas_node({"idea_candidates": [], "judge_bias": "b"})
    except httpx.ReadError:
        print("ok: rank_ideas still fails loudly, since there is nothing to degrade")
        return
    finally:
        nodes.idea_ranker = original
    raise AssertionError("rank_ideas should not silently continue")


class _CapturingAgent:
    """Records the config a research node passes in."""

    def __init__(self):
        self.config = None

    async def ainvoke(self, _payload, config=None):
        self.config = config
        return {"messages": [type("M", (), {"content": "summary"})()]}


def test_research_agents_get_a_raised_recursion_limit():
    """25 supersteps is ~12 tool calls, which real research can exceed."""
    original = nodes.idea_researcher
    agent = _CapturingAgent()
    nodes.idea_researcher = agent
    try:
        asyncio.run(
            nodes.research_one_idea_node(
                {
                    "idea": Idea(title="I", pitch="p"),
                    "hackathon_synopsis": "s",
                    "judge_bias": "b",
                }
            )
        )
    finally:
        nodes.idea_researcher = original

    assert agent.config, "the research call must pass a config"
    limit = agent.config.get("recursion_limit")
    assert limit and limit > 25, f"limit should exceed LangGraph's default 25: {limit}"
    print(f"ok: research agents run with recursion_limit={limit}, above the default 25")


def test_recursion_error_degrades_rather_than_aborting():
    """A stuck agent must cost one idea, not the run - and must not be retried."""
    from langgraph.errors import GraphRecursionError

    assert not issubclass(
        GraphRecursionError, nodes._TRANSIENT
    ), "a recursion loop is not transient; retrying it would waste three runs"

    original = nodes.idea_researcher
    said: list[str] = []
    nodes.set_say_sink(said.append)
    nodes.idea_researcher = _AlwaysFails(GraphRecursionError("limit of 25 reached"))
    try:
        out = asyncio.run(
            nodes.research_one_idea_node(
                {
                    "idea": Idea(title="The Unboxing", pitch="p"),
                    "hackathon_synopsis": "s",
                    "judge_bias": "b",
                }
            )
        )
    finally:
        nodes.idea_researcher = original
        nodes.set_say_sink(None)

    idea = out["ideas"][0]
    assert idea.title == "The Unboxing", out
    assert "GraphRecursionError" in idea.research, idea.research
    assert any("The Unboxing" in line for line in said), said
    print("ok: a looping research agent degrades one idea and is reported")


class _Hangs:
    """A research agent that never returns, like a stalled Ollama request."""

    def __init__(self):
        self.cancelled = False

    async def ainvoke(self, _payload, config=None):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def test_a_hung_worker_times_out():
    """Retries and degradation only fire on errors; silence needs a timeout."""
    original_agent = nodes.idea_researcher
    original_timeout = nodes._RESEARCH_TIMEOUT
    agent = _Hangs()
    nodes.idea_researcher = agent
    nodes._RESEARCH_TIMEOUT = 0.3
    said: list[str] = []
    nodes.set_say_sink(said.append)
    try:
        out = asyncio.run(
            nodes.research_one_idea_node(
                {
                    "idea": Idea(title="Stuck", pitch="p"),
                    "hackathon_synopsis": "s",
                    "judge_bias": "b",
                }
            )
        )
    finally:
        nodes.idea_researcher = original_agent
        nodes._RESEARCH_TIMEOUT = original_timeout
        nodes.set_say_sink(None)

    assert agent.cancelled, "the hung call must actually be cancelled"
    assert "TimeoutError" in out["ideas"][0].research, out["ideas"][0].research
    assert any("Stuck" in line for line in said), said
    print("ok: a hung research worker times out, is cancelled, and degrades")


def test_skip_abandons_in_flight_research():
    """The `s` binding must interrupt a call already in flight, not just queued."""
    original_agent = nodes.idea_researcher
    original_timeout = nodes._RESEARCH_TIMEOUT
    agent = _Hangs()
    nodes.idea_researcher = agent
    nodes._RESEARCH_TIMEOUT = 3600  # long: only the skip can end this
    said: list[str] = []
    nodes.set_say_sink(said.append)

    async def scenario():
        task = asyncio.ensure_future(
            nodes.research_one_idea_node(
                {
                    "idea": Idea(title="Slow", pitch="p"),
                    "hackathon_synopsis": "s",
                    "judge_bias": "b",
                }
            )
        )
        await asyncio.sleep(0.2)  # let it get in flight
        nodes.request_skip()
        return await asyncio.wait_for(task, timeout=5)

    try:
        out = asyncio.run(scenario())
    finally:
        nodes.idea_researcher = original_agent
        nodes._RESEARCH_TIMEOUT = original_timeout
        nodes.clear_skip()
        nodes.set_say_sink(None)

    assert agent.cancelled, "skip must cancel the in-flight call"
    research = out["ideas"][0].research
    assert "skipped" in research.lower(), research
    assert "unavailable after retries" not in research, (
        "a deliberate skip should read differently from a failure: " + research
    )
    assert any("skipped" in line for line in said), said
    print("ok: skip abandons an in-flight worker and is labelled as a skip")


def test_skip_is_rearmed_after_the_phase():
    nodes.clear_skip()
    assert not nodes.skip_requested()
    nodes.request_skip()
    assert nodes.skip_requested()
    nodes.clear_skip()
    assert not nodes.skip_requested(), "otherwise every later phase skips too"
    print("ok: skip re-arms so it does not leak into the next phase")


if __name__ == "__main__":
    test_transient_errors_are_retried()
    test_every_transport_error_is_covered()
    test_non_transient_errors_are_not_retried()
    test_one_failed_judge_does_not_kill_the_run()
    test_one_failed_idea_does_not_kill_the_run()
    test_single_nodes_still_propagate()
    test_research_agents_get_a_raised_recursion_limit()
    test_recursion_error_degrades_rather_than_aborting()
    test_a_hung_worker_times_out()
    test_skip_abandons_in_flight_research()
    test_skip_is_rearmed_after_the_phase()
    print("\nAll resilience checks passed.")
