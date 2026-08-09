"""Self-check for the TUI: driver loop, dialog dispatch, and output capture.

Hermetic - no model calls, no network, no real graph. Uses a tiny in-memory
graph whose middle node interrupts twice, which is the shape choose_build_dir
takes on a bad path or a non-empty directory.

Run:  uv run python tests/test_tui_driver.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from textual import work
from textual.app import App

from agent import nodes, tui
from agent.tui import (
    BiasScreen,
    ConfirmScreen,
    IdeaScreen,
    PathScreen,
    prompt_for,
)


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


async def _drive(graph, config, answers, payload):
    """The driver loop from HackathonApp.run_pipeline, minus the widgets."""
    seen_nodes, prompts = [], []
    replies = list(answers)
    while True:
        pending = None
        async for event in graph.astream(payload, config, stream_mode="updates"):
            if "__interrupt__" in event:
                pending = event["__interrupt__"][0].value
            else:
                seen_nodes.extend(event.keys())
        if pending is None:
            break
        prompts.append(pending)
        payload = Command(resume=replies.pop(0))
    return seen_nodes, prompts


def test_resumes_through_two_interrupts_in_one_node():
    """The trap: after a node's second interrupt, state.next is empty."""
    graph = _twice_interrupting_graph()
    config = {"configurable": {"thread_id": "t-twice"}}

    nodes_seen, prompts = asyncio.run(_drive(graph, config, ["bad", "yes"], {}))

    assert len(prompts) == 2, f"expected two prompts, got {prompts}"
    assert "warning" in prompts[1], "second prompt should be the confirmation"
    assert nodes_seen == ["one", "two", "three"], nodes_seen

    state = asyncio.run(graph.aget_state(config))
    assert state.values["b"] == "bad/yes"
    print("ok: resumes through a second interrupt inside the same node")


def test_state_next_is_empty_at_second_interrupt():
    """Pins the trap itself, so nobody 'simplifies' the loop to use state.next."""
    graph = _twice_interrupting_graph()
    config = {"configurable": {"thread_id": "t-next"}}

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


def test_completes_without_interrupts():
    graph = _twice_interrupting_graph()
    config = {"configurable": {"thread_id": "t-clean"}}
    nodes_seen, prompts = asyncio.run(_drive(graph, config, ["good"], {}))
    assert prompts == [{"instructions": "path?"}], prompts
    assert nodes_seen == ["one", "two", "three"], nodes_seen
    print("ok: a run with one interrupt finishes cleanly")


def test_dialog_dispatch():
    """Each interrupt payload reaches the right dialog."""
    cases = [
        ({"draft_bias": "x", "instructions": "edit"}, BiasScreen),
        ({"ideas": "1. A - p\n2. B - q", "instructions": "pick"}, IdeaScreen),
        ({"warning": "not empty", "instructions": "yes?"}, ConfirmScreen),
        ({"instructions": "path", "tasks": ["t"]}, PathScreen),
        ({"instructions": "path", "error": "not absolute"}, PathScreen),
    ]
    for payload, expected in cases:
        got = prompt_for(payload)
        assert isinstance(got, expected), f"{payload} -> {type(got)}, want {expected}"

    # Both choose_build_dir payloads must route differently despite sharing keys.
    assert isinstance(prompt_for({"instructions": "p"}), PathScreen)
    assert isinstance(prompt_for({"warning": "w", "instructions": "p"}), ConfirmScreen)
    print("ok: payload shape routes to the right dialog")


def test_idea_screen_parses_options():
    """select_idea passes one newline-joined string, not a list."""
    screen = IdeaScreen({"ideas": "1. Alpha - a\n2. Beta - b\n", "instructions": ""})
    assert screen.options == ["1. Alpha - a", "2. Beta - b"], screen.options
    print("ok: idea list is split from the joined string")


def test_say_sink_redirects_and_restores():
    captured = []
    nodes.set_say_sink(captured.append)
    nodes._say("hello")
    assert captured == ["hello"], captured

    nodes.set_say_sink(None)
    assert nodes._say_sink is None, "None must restore printing for the CLI path"
    print("ok: _say redirects to the TUI and restores the default")


class _Harness(App):
    """Minimal app that pushes one dialog and records what it returns."""

    def __init__(self, screen):
        super().__init__()
        self._screen = screen
        self.result = "UNSET"

    @work
    async def on_mount(self):
        self.result = await self.push_screen_wait(self._screen)
        self.exit()


async def _drive_screen(screen, keys):
    app = _Harness(screen)
    async with app.run_test() as pilot:
        await pilot.pause()
        for key in keys:
            await pilot.press(key)
        await pilot.pause()
    return app.result


def test_dialogs_return_expected_values():
    """Drive each dialog headlessly and check what it hands back to the graph."""

    async def run():
        path = await _drive_screen(
            PathScreen({"instructions": "path"}), list("/tmp/x") + ["enter"]
        )
        assert path == "/tmp/x", path

        yes = await _drive_screen(
            ConfirmScreen({"warning": "w", "instructions": "i"}), ["enter"]
        )
        assert yes == "yes", yes

        no = await _drive_screen(
            ConfirmScreen({"warning": "w", "instructions": "i"}), ["tab", "enter"]
        )
        assert no == "no", no

        # Selecting the second idea must yield "2": select_idea_node indexes
        # final_ideas from 1, matching the numbering it rendered.
        picked = await _drive_screen(
            IdeaScreen({"ideas": "1. A - a\n2. B - b", "instructions": "pick"}),
            ["down", "tab", "tab", "enter"],
        )
        assert picked == "2", picked

        own = await _drive_screen(
            IdeaScreen({"ideas": "1. A - a", "instructions": "pick"}),
            ["tab", *list("my own"), "enter"],
        )
        assert own == "my own", own

        bias = await _drive_screen(
            BiasScreen({"draft_bias": "draft", "instructions": "edit"}),
            ["tab", "enter"],
        )
        assert bias == "draft", f"approving unchanged should return the draft: {bias}"

    asyncio.run(run())
    print("ok: dialogs return the values the graph expects")


def _log_text(app) -> str:
    lines = []
    for strip in app.query_one("#output").lines:
        lines.append("".join(seg.text for seg in strip._segments).rstrip())
    return "\n".join(lines)


def test_app_survives_a_failing_node():
    """A node raising must not tear down the UI; the run is still resumable."""

    def boom(_):
        raise RuntimeError("ollama unreachable")

    def stub(checkpointer=None):
        g = StateGraph(S)
        g.add_node("fetch_html", boom)
        g.add_edge(START, "fetch_html")
        g.add_edge("fetch_html", END)
        return g.compile(checkpointer=checkpointer)

    original_build, original_db = tui.build_graph, tui.DB_PATH
    tmp = tempfile.mkdtemp()
    tui.build_graph = stub
    tui.DB_PATH = str(Path(tmp) / "t.db")

    async def run():
        app = tui.HackathonApp(url="https://x.devpost.com", thread_id="fail1")
        async with app.run_test() as pilot:
            await pilot.pause()
            await asyncio.sleep(1.5)
            await pilot.pause()
            assert app.is_running, "a failing node must not kill the app"
            text = _log_text(app)
        return text

    try:
        text = asyncio.run(run())
    finally:
        tui.build_graph, tui.DB_PATH = original_build, original_db
        shutil.rmtree(tmp, ignore_errors=True)

    assert "ollama unreachable" in text, text
    assert "--thread fail1" in text, "must tell the user how to resume"
    print("ok: a failing node is reported, not fatal")


if __name__ == "__main__":
    try:
        for case in (
            test_resumes_through_two_interrupts_in_one_node,
            test_state_next_is_empty_at_second_interrupt,
            test_completes_without_interrupts,
            test_dialog_dispatch,
            test_idea_screen_parses_options,
            test_say_sink_redirects_and_restores,
            test_dialogs_return_expected_values,
            test_app_survives_a_failing_node,
        ):
            case()
    finally:
        nodes.set_say_sink(None)
    print("\nAll TUI driver checks passed.")
