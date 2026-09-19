"""Self-check for the CLI driver loop and failure handling.

Hermetic - no model calls, no network, no real graph. Uses a tiny in-memory
graph whose middle node interrupts twice, which is the shape choose_build_dir
takes on a bad path or a non-empty directory.

Run:  uv run python tests/test_cli_driver.py
"""

import asyncio
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent import cli


class S(TypedDict, total=False):
    a: str
    b: str


def _twice_interrupting_graph():
    """Middle node interrupts, and interrupts a second time on a bad answer."""

    def one(s):
        return {"a": "one"}

    def two(s):
        answer = interrupt({"instructions": "path?"})
        if answer == "bad":
            confirmed = interrupt({"warning": "not empty"})
            return {"b": f"{answer}/{confirmed}"}
        return {"b": answer}

    def three(s):
        return {"a": "done"}

    g = StateGraph(S)
    for name, fn in [("one", one), ("two", two), ("three", three)]:
        g.add_node(name, fn)
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", "three")
    g.add_edge("three", END)
    return g.compile(checkpointer=MemorySaver())


def scripted(answers):
    """An `ask` that replays fixed answers and records the questions asked."""
    questions = []
    replies = list(answers)

    def ask(payload):
        questions.append(payload)
        return replies.pop(0)

    return ask, questions


def _drive(graph, config, answers):
    ask, questions = scripted(answers)
    printed = io.StringIO()
    with redirect_stdout(printed):
        asyncio.run(cli.drive(graph, config, {}, ask))
    return questions, printed.getvalue()


def test_resumes_through_two_interrupts_in_one_node():
    """The trap: after a node's second interrupt, state.next is empty."""
    graph = _twice_interrupting_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "t-twice"}}

    questions, printed = _drive(graph, config, ["bad", "yes"])

    assert len(questions) == 2, f"expected two questions, got {questions}"
    assert "warning" in questions[1], "second question should be the confirmation"
    for node in ("one", "two", "three"):
        assert f"=== {node} ===" in printed, printed

    state = asyncio.run(graph.aget_state(config))
    assert state.values["b"] == "bad/yes"
    print("ok: resumes through a second interrupt inside the same node")


def test_state_next_is_empty_at_second_interrupt():
    """Pins the trap itself, so nobody 'simplifies' the loop to use state.next."""
    graph = _twice_interrupting_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "t-next"}}

    async def run():
        async for _ in graph.astream({}, config, stream_mode="updates"):
            pass
        first = await graph.aget_state(config)
        async for _ in graph.astream(
            Command(resume="bad"), config, stream_mode="updates"
        ):
            pass
        return first, await graph.aget_state(config)

    first, second = asyncio.run(run())
    assert first.next == ("two",), first.next
    assert second.next == (), (
        "LangGraph reports no next node at the second interrupt; a driver keyed "
        f"on state.next would call this finished. Got {second.next}"
    )
    assert second.interrupts, "but the graph is still waiting"
    print("ok: state.next is empty mid-node, so the driver must use __interrupt__")


def test_completes_with_one_interrupt():
    graph = _twice_interrupting_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "t-clean"}}
    questions, _ = _drive(graph, config, ["good"])
    assert questions == [{"instructions": "path?"}], questions
    print("ok: a run with one interrupt finishes cleanly")


def test_a_failing_node_prints_how_to_resume():
    """A node raising must be reported with a resume hint, not a traceback."""

    def boom(state):
        raise RuntimeError("ollama unreachable")

    def failing_graph(checkpointer=None):
        g = StateGraph(S)
        g.add_node("fetch_html", boom)
        g.add_edge(START, "fetch_html")
        g.add_edge("fetch_html", END)
        return g.compile(checkpointer=checkpointer)

    original_build = cli.build_graph
    cli.build_graph = failing_graph
    printed = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with redirect_stdout(printed):
                asyncio.run(
                    cli.run(
                        "https://x.devpost.com",
                        "fail1",
                        db_path=str(Path(tmp) / "t.db"),
                    )
                )
        finally:
            cli.build_graph = original_build

    text = printed.getvalue()
    assert "ollama unreachable" in text, text
    assert "--thread fail1" in text, "must tell the user how to resume"
    print("ok: a failing node is reported with a resume hint")


if __name__ == "__main__":
    test_resumes_through_two_interrupts_in_one_node()
    test_state_next_is_empty_at_second_interrupt()
    test_completes_with_one_interrupt()
    test_a_failing_node_prints_how_to_resume()
    print("\nAll CLI driver checks passed.")
