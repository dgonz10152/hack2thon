"""End-to-end test: the real pipeline through the real TUI, with dummy data.

Hermetic - no Ollama, no DuckDuckGo, no opencode, no network. Stubs are placed
at the *model* boundary (the model-bound names in agent.nodes.*), not at the
node boundary, so every real node body still runs: BeautifulSoup parsing, prompt
assembly, the merge_judges/merge_ideas reducers, _resolve_selected_idea, the
choose_build_dir validation loop, and the build loop.

Run:  uv run python tests/test_tui_e2e.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

from agent import progress, tui
from agent.nodes import build, common, ideas, judges
from agent.prompts import BIAS_COMPILER_SYSTEM, THEME_EXTRACTOR_SYSTEM
from agent.tui import app as tui_app
from agent.state import (
    BuildPlan,
    BuildTask,
    FinalIdea,
    FinalIdeas,
    Idea,
    IdeaCandidates,
    Judge,
    Judges,
)

DUMMY_HTML = """
<html><body>
  <h1>Dummy Hackathon 2026</h1>
  <script>ignored()</script>
  <p>Build something with agents. Judged on impact and technical depth.</p>
  <div class="judges"><span>Ada Lovelace</span><span>Alan Turing</span></div>
</body></html>
"""

CANDIDATE_TITLES = [f"Idea {n}" for n in range(1, 7)]
RANKED_TITLES = CANDIDATE_TITLES[:3]

# Recorded so assertions can inspect what the real nodes passed to the model.
SEEN: dict = {}


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeTool:
    def __init__(self, result):
        self.result = result

    def invoke(self, _args):
        return self.result


class FakeModel:
    """Stands in for ChatOllama for the two direct model.invoke call sites.

    Dispatches on the exact prompt constant rather than guessing from substrings:
    THEME_EXTRACTOR_SYSTEM and BIAS_COMPILER_SYSTEM share vocabulary, so a
    keyword match silently routes both to the same branch.
    """

    def invoke(self, messages):
        system = str(messages[0].content)
        if system == THEME_EXTRACTOR_SYSTEM:
            return AIMessage(content="SYNOPSIS: agents, impact, technical depth")
        assert system == BIAS_COMPILER_SYSTEM, f"unexpected prompt: {system[:80]}"
        SEEN["bias_prompt"] = str(messages[-1].content)
        return AIMessage(content="BIAS: likes agentic systems")


class FakeStructured:
    """Stands in for model.with_structured_output(Model)."""

    def __init__(self, result, record_key=None):
        self.result = result
        self.record_key = record_key

    def invoke(self, messages):
        if self.record_key:
            SEEN[self.record_key] = str(messages[-1].content)
        return self.result(messages) if callable(self.result) else self.result


class FakeReactAgent:
    """Stands in for create_react_agent; nodes read result['messages'][-1]."""

    def __init__(self, text, record_key=None):
        self.text = text
        self.record_key = record_key

    async def ainvoke(self, payload, config=None):
        if self.record_key:
            SEEN.setdefault(self.record_key, []).append(
                str(payload["messages"][-1].content)
            )
        return {"messages": [AIMessage(content=self.text)]}


class FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class FakeProc:
    def __init__(self):
        self.returncode = 0
        self.stdout = FakeStdout([b"<- Write main.py", b"Wrote file successfully."])

    def kill(self):
        pass

    async def wait(self):
        return 0


def install_stubs() -> dict:
    """Patch the model boundary. Returns the originals for restoration."""
    # Each stub goes on the module the code lives in: node functions read the
    # globals of their own module, so patching the package would do nothing.
    homes = {
        "fetch_html": judges,
        "chat": judges,
        "judge_extractor": judges,
        "judge_researcher": judges,
        "idea_generator": ideas,
        "idea_ranker": ideas,
        "idea_compiler": ideas,
        "idea_researcher": ideas,
        "build_planner": build,
        "_git": build,
        "_has_changes": build,
    }
    original = {name: (mod, getattr(mod, name)) for name, mod in homes.items()}
    original["create_subprocess_exec"] = (asyncio, asyncio.create_subprocess_exec)
    original["which"] = (shutil, shutil.which)

    judges.fetch_html = FakeTool(DUMMY_HTML)
    judges.chat = FakeModel()  # the retrying wrapper the nodes actually call
    judges.judge_extractor = FakeStructured(
        Judges(
            judges=[
                Judge(name="Ada Lovelace", blurb="analytical engine"),
                Judge(name="Alan Turing", blurb="computation"),
            ]
        )
    )
    judges.judge_researcher = FakeReactAgent(
        "Researched profile.", record_key="judge_prompts"
    )
    ideas.idea_generator = FakeStructured(
        IdeaCandidates(
            ideas=[Idea(title=t, pitch=f"pitch {t}") for t in CANDIDATE_TITLES]
        )
    )
    # Titles match the candidates, as the real prompt requires. Note this does
    # not by itself exercise merge_ideas' title matching: the orchestrator fans
    # out from whatever rank_ideas returned and each worker model_copy's that
    # object, so seed and update always share a title. What the merge assertion
    # below actually catches is research failing to land at all.
    ideas.idea_ranker = FakeStructured(
        IdeaCandidates(ideas=[Idea(title=t, pitch=f"pitch {t}") for t in RANKED_TITLES])
    )
    ideas.idea_researcher = FakeReactAgent("Feasible. No close prior art.")
    ideas.idea_compiler = FakeStructured(
        FinalIdeas(
            ideas=[
                FinalIdea(
                    title=t,
                    problem=f"problem for {t}",
                    target_users="devs",
                    core_features=["a", "b"],
                    tech_stack=["python"],
                    differentiation="d",
                    bias_alignment="b",
                    winning_formula_notes="w",
                    feasibility_notes="f",
                )
                for t in RANKED_TITLES
            ]
        ),
        record_key="compiler_prompt",
    )
    build.build_planner = FakeStructured(
        BuildPlan(
            project_summary="A dummy app.",
            tech_stack=["python"],
            setup_notes="none",
            tasks=[
                BuildTask(title="Scaffold", description="make it", acceptance="runs"),
                BuildTask(title="Verify", description="check it", acceptance="works"),
            ],
        ),
        record_key="planner_prompt",
    )

    commits: list = []

    def fake_git(_dir, *args):
        if args[:1] == ("commit",):
            commits.append(args[-1])

        class R:
            stdout = " main.py | 1 +"

        return R()

    build._git = fake_git
    build._has_changes = lambda _dir: True
    SEEN["commits"] = commits

    async def fake_exec(*_args, **_kwargs):
        return FakeProc()

    asyncio.create_subprocess_exec = fake_exec
    shutil.which = lambda _: "/usr/bin/opencode"
    return original


def restore(original: dict) -> None:
    for name, (module, value) in original.items():
        setattr(module, name, value)


# --------------------------------------------------------------------------
# Pilot helpers - polling, never bare sleeps, so the test does not flake
# --------------------------------------------------------------------------


async def wait_for_screen(app, screen_type, timeout=15.0):
    """Wait until `screen_type` is on top, or fail with what is there instead."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if isinstance(app.screen, screen_type):
            return app.screen
        if "Pipeline failed" in log_text(app):
            raise AssertionError(f"pipeline failed early:\n{log_text(app)}")
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"timed out waiting for {screen_type.__name__}; "
        f"current screen is {type(app.screen).__name__}\n{log_text(app)}"
    )


async def wait_for_text(app, needle, timeout=15.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if needle in log_text(app):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {needle!r}\n{log_text(app)}")


def log_text(app) -> str:
    lines = []
    for strip in app.query_one("#output").lines:
        lines.append("".join(seg.text for seg in strip._segments).rstrip())
    return "\n".join(lines)


async def type_text(pilot, text):
    for char in text:
        await pilot.press("underscore" if char == "_" else char)


# --------------------------------------------------------------------------
# The test
# --------------------------------------------------------------------------


async def _run(tmp: Path):
    build_dir = tmp / "built"
    app = tui.HackathonApp(url="https://dummy.devpost.com", thread_id="e2e")

    async with app.run_test(size=(120, 40)) as pilot:
        # 1. Bias review: edit the draft, then approve.
        bias = await wait_for_screen(pilot.app, tui.BiasScreen)
        assert "BIAS: likes agentic systems" in bias.payload["draft_bias"]
        await type_text(pilot, "EDITED. ")
        await pilot.press("tab", "enter")

        # 2. Idea selection: pick the second idea.
        ideas = await wait_for_screen(pilot.app, tui.IdeaScreen)
        assert len(ideas.options) == len(RANKED_TITLES), ideas.options
        await pilot.press("down")
        await pilot.press("tab", "tab", "enter")

        # 3. Build directory: a relative path first, which must be rejected and
        #    re-prompted. This is the second-interrupt-in-one-node case.
        path_screen = await wait_for_screen(pilot.app, tui.PathScreen)
        assert "error" not in path_screen.payload
        await type_text(pilot, "relative/path")
        await pilot.press("enter")

        retry = await wait_for_screen(pilot.app, tui.PathScreen)
        deadline = asyncio.get_event_loop().time() + 10
        while "error" not in retry.payload:
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError("relative path was accepted without an error")
            await asyncio.sleep(0.05)
            retry = pilot.app.screen
        assert "absolute" in retry.payload["error"], retry.payload

        # 4. Now a real absolute path.
        await type_text(pilot, str(build_dir))
        await pilot.press("enter")

        await wait_for_text(pilot.app, "Done. Press q to quit.")
        text = log_text(pilot.app)
        state = await _final_state(pilot.app)
        phases = set(pilot.app.query_one(tui.PhasePanel).done)

    return text, state, phases, build_dir


async def _final_state(app):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(tui_app.DB_PATH) as saver:
        graph = tui_app.build_graph(saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "e2e"}})
        return snapshot.values


def test_full_pipeline_through_the_tui():
    tmp = Path(tempfile.mkdtemp())
    original_db = tui_app.DB_PATH
    tui_app.DB_PATH = str(tmp / "e2e.db")
    SEEN.clear()
    # install_stubs must be called exactly once: a second call would capture the
    # stubs as the "originals" and restore() would leave agent.nodes patched.
    original = install_stubs()
    try:
        text, state, phases, build_dir = asyncio.run(_run(tmp))
    finally:
        restore(original)
        tui_app.DB_PATH = original_db
        shutil.rmtree(tmp, ignore_errors=True)

    # run_pipeline logs failures instead of raising, so check this FIRST or a
    # broken stub would look like a quiet pass.
    assert "Pipeline failed" not in text, text

    # Every phase reached.
    expected = {node for node, _ in tui.PHASES}
    assert phases == expected, f"missing phases: {sorted(expected - phases)}"

    # The bias edit round-tripped through the interrupt.
    assert state["judge_bias"].startswith("EDITED. "), state["judge_bias"]

    # Judge research really ran per judge, and the reducer merged the summaries.
    assert len(state["judges"]) == 2, state["judges"]
    assert all(j.online_summary for j in state["judges"]), state["judges"]

    # A degraded worker in the happy path means a stub broke and the broad
    # except in the research nodes swallowed it as "research unavailable".
    assert not any("unavailable" in (i.research or "") for i in state["ideas"]), [
        i.research for i in state["ideas"]
    ]
    assert not any(
        "unavailable" in (j.online_summary or "") for j in state["judges"]
    ), [j.online_summary for j in state["judges"]]

    # Every ranked idea came back enriched: each worker ran and merge_ideas
    # upserted onto its seed rather than appending a duplicate.
    assert len(state["ideas"]) == len(RANKED_TITLES), state["ideas"]
    assert all(i.research for i in state["ideas"]), (
        "research did not land on the ranked ideas - a worker did not run, or "
        "merge_ideas failed to match the seed by title: "
        f"{[(i.title, bool(i.research)) for i in state['ideas']]}"
    )

    # The second idea was selected and carried into the build plan.
    assert state["selected_idea"] == "2", state["selected_idea"]
    assert RANKED_TITLES[1] in SEEN["planner_prompt"], SEEN["planner_prompt"]

    # Build ran both tasks, streamed output, and committed each one.
    assert state["build_dir"] == str(build_dir), state["build_dir"]
    assert "Build finished" in state["build_result"], state["build_result"]
    assert [t.status for t in state["build_tasks"]] == ["ok", "ok"]
    assert len(SEEN["commits"]) == 2, SEEN["commits"]

    # Sub-agent output must reach the log, and in order: it goes through the
    # same queue as the phase headers, so it cannot arrive after "Done".
    assert "Wrote file successfully." in text, "sub-agent output should stream"
    assert text.index("Wrote file successfully.") < text.index("Done. Press q"), (
        "build output arrived after the done message; the log has two write "
        f"paths again:\n{text}"
    )

    print("ok: full pipeline runs through the TUI on dummy data")


def test_judge_workers_get_the_synopsis():
    """Guards the silent bug: workers were researched with an empty synopsis."""
    prompts = SEEN.get("judge_prompts", [])
    assert len(prompts) == 2, prompts
    for prompt in prompts:
        assert "SYNOPSIS:" in prompt, (
            "judge worker received no hackathon synopsis; extract_theme must run "
            f"before the orchestrator: {prompt[:200]}"
        )
    print("ok: every judge worker received the hackathon synopsis")


def test_completed_phases_maps_state_to_sidebar():
    """Unit check for the resume mapping, including the review_bias special case."""
    done, counts = tui.completed_phases({})
    assert done == set(), done

    partial = {
        "html": "x",
        "hackathon_synopsis": "s",
        "judges": [
            Judge(name="A", blurb="b", online_summary="done"),
            Judge(name="B", blurb="b"),
        ],
        "judge_bias": "bias",
    }
    done, counts = tui.completed_phases(partial)
    assert "research_one_judge" in done and counts["research_one_judge"] == 1
    assert "compile_bias" in done
    # review_bias has no output of its own, so it is only "done" once the next
    # phase has produced something.
    assert "review_bias" not in done, done

    partial["idea_candidates"] = [Idea(title="I", pitch="p")]
    done, _ = tui.completed_phases(partial)
    assert "review_bias" in done, done
    print("ok: checkpoint state maps to the right completed phases")


async def _resume_scenario():
    # Run 1: stop at the first dialog without answering.
    app = tui.HackathonApp(url="https://dummy.devpost.com", thread_id="resume")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_screen(pilot.app, tui.BiasScreen)
        pilot.app.exit()

    # Run 2: resume by thread id alone, as `--thread <id>` does.
    app2 = tui.HackathonApp(url=None, thread_id="resume")
    async with app2.run_test(size=(120, 40)) as pilot:
        screen = await wait_for_screen(pilot.app, tui.BiasScreen)
        # The dialog can appear before the queued log lines have been drained,
        # so wait for the banner rather than reading the log immediately.
        await wait_for_text(pilot.app, "Resumed thread")
        phases = set(pilot.app.query_one(tui.PhasePanel).done)
        text = log_text(pilot.app)
        url = pilot.app.url
        assert screen.payload.get("draft_bias"), "the pending interrupt should return"

    # Run 3: an id that was never checkpointed.
    app3 = tui.HackathonApp(url=None, thread_id="nope")
    async with app3.run_test(size=(120, 40)) as pilot:
        await wait_for_text(pilot.app, "No checkpoint found")
        unknown = log_text(pilot.app)

    return phases, text, url, unknown


def test_resume_restores_visible_progress():
    """Resume worked, but showed an empty sidebar and log, so it looked broken."""
    tmp = Path(tempfile.mkdtemp())
    original_db = tui_app.DB_PATH
    tui_app.DB_PATH = str(tmp / "resume.db")
    SEEN.clear()
    original = install_stubs()
    try:
        phases, text, url, unknown = asyncio.run(_resume_scenario())
    finally:
        restore(original)
        tui_app.DB_PATH = original_db
        shutil.rmtree(tmp, ignore_errors=True)

    assert "Pipeline failed" not in text, text
    # The work already done must be visible, not silently re-presented as new.
    for phase in ("fetch_html", "extract_theme", "extract_judges", "compile_bias"):
        assert phase in phases, f"{phase} not restored to the sidebar: {phases}"
    assert "Resumed thread resume" in text, text
    assert "judges: 2/2 researched" in text, text
    assert url == "https://dummy.devpost.com", f"status bar lost the URL: {url}"

    # An unknown id must say so rather than sitting there doing nothing.
    assert "No checkpoint found" in unknown, unknown
    assert "Resume with" not in unknown, "must not suggest reusing a bad id"
    print("ok: resume restores the sidebar, log, and URL; bad ids are reported")


class _ExplodingGraph:
    """Wraps a real graph but blows up on one thread, like an old checkpoint."""

    def __init__(self, real, bad_thread):
        self.real = real
        self.bad = bad_thread

    async def aget_state(self, config):
        if config["configurable"]["thread_id"] == self.bad:
            raise RuntimeError("InvalidUpdateError: At key 'judge_bias'")
        return await self.real.aget_state(config)


async def _ls_scenario(tmp: Path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # Two runs, stopped at different points, so the listing has variety.
    for thread_id in ("run-one", "run-two"):
        app = tui.HackathonApp(url="https://dummy.devpost.com", thread_id=thread_id)
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_for_screen(pilot.app, tui.BiasScreen)
            pilot.app.exit()

    async with AsyncSqliteSaver.from_conn_string(tui_app.DB_PATH) as saver:
        graph = tui_app.build_graph(saver)
        rows = await tui.list_threads(saver, graph)
        # One unreadable thread must not take out the listing.
        degraded = await tui.list_threads(saver, _ExplodingGraph(graph, "run-two"))

    # Browsing mode: pick the top run and confirm it resumes that thread.
    app = tui.HackathonApp(url=None, thread_id="")
    async with app.run_test(size=(140, 44)) as pilot:
        picker = await wait_for_screen(pilot.app, tui.ThreadPicker)
        listed = [r["thread_id"] for r in picker.rows]
        await pilot.press("enter")
        await wait_for_screen(pilot.app, tui.BiasScreen)
        await wait_for_text(pilot.app, "Resumed thread")
        chosen = pilot.app.thread_id
        text = log_text(pilot.app)

    return rows, degraded, listed, chosen, text


def test_ls_lists_and_resumes_previous_runs():
    tmp = Path(tempfile.mkdtemp())
    original_db = tui_app.DB_PATH
    tui_app.DB_PATH = str(tmp / "ls.db")
    SEEN.clear()
    original = install_stubs()
    try:
        rows, degraded, listed, chosen, text = asyncio.run(_ls_scenario(tmp))
    finally:
        restore(original)
        tui_app.DB_PATH = original_db
        shutil.rmtree(tmp, ignore_errors=True)

    ids = [r["thread_id"] for r in rows]
    assert set(ids) == {"run-one", "run-two"}, ids
    assert ids == ["run-two", "run-one"], f"newest run must come first: {ids}"

    top = rows[0]
    assert top["url"] == "https://dummy.devpost.com", top
    assert top["status"] == "waiting for you", top
    assert 0 < top["done"] < top["total"], top
    assert top["ts"], "each row needs a timestamp"

    # A thread the current graph cannot read is flagged, not fatal.
    assert len(degraded) == 2, degraded
    broken = [r for r in degraded if r["status"] == "unreadable"]
    assert len(broken) == 1 and broken[0]["thread_id"] == "run-two", degraded
    assert "judge_bias" in broken[0]["error"], broken[0]

    # The picker showed both and resumed the one that was highlighted.
    assert set(listed) == {"run-one", "run-two"}, listed
    assert chosen == listed[0], f"picked {listed[0]} but resumed {chosen}"
    assert f"Resumed thread {chosen}" in text, text
    print("ok: ls lists previous runs, survives a bad one, and resumes a pick")


class _NeverReturns:
    """A research agent that hangs, like a stalled Ollama request."""

    async def ainvoke(self, _payload, config=None):
        await asyncio.sleep(3600)


async def _skip_scenario():
    app = tui.HackathonApp(url="https://dummy.devpost.com", thread_id="skip")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for_screen(pilot.app, tui.BiasScreen)
        await pilot.press("tab", "enter")  # approve the bias
        await wait_for_text(pilot.app, "Rank top 20")
        await asyncio.sleep(0.5)  # let the workers get in flight and stall
        stuck_screen = type(pilot.app.screen).__name__
        await pilot.press("s")
        # Skipping must carry the run forward to the next human step.
        await wait_for_screen(pilot.app, tui.IdeaScreen, timeout=25)
        return stuck_screen, log_text(pilot.app)


def test_skip_button_unsticks_a_hung_research_phase():
    """`s` must rescue a phase that is hung, not merely one that is erroring."""
    tmp = Path(tempfile.mkdtemp())
    original_db, tui_app.DB_PATH = tui_app.DB_PATH, str(tmp / "skip.db")
    SEEN.clear()
    original = install_stubs()
    original_agent = ideas.idea_researcher
    original_timeout = common.RESEARCH_TIMEOUT
    ideas.idea_researcher = _NeverReturns()
    common.RESEARCH_TIMEOUT = 3600  # long enough that only the key can help
    try:
        stuck_screen, text = asyncio.run(_skip_scenario())
    finally:
        ideas.idea_researcher = original_agent
        common.RESEARCH_TIMEOUT = original_timeout
        progress.clear_skip()
        restore(original)
        tui_app.DB_PATH = original_db
        shutil.rmtree(tmp, ignore_errors=True)

    assert stuck_screen == "Screen", f"should have been mid-run, not {stuck_screen}"
    assert "Pipeline failed" not in text, text
    assert "skip requested" in text, text
    # Not an exact count: how many workers are skipped depends on how many were
    # in flight when the key was pressed (IDEA_RESEARCH_CONCURRENCY gates them),
    # and the log drains on a timer. Reaching IdeaScreen is the real guarantee.
    assert "skipped idea research" in text, text
    print("ok: the skip key unsticks a hung research phase and moves on")


def test_ls_with_no_previous_runs():
    tmp = Path(tempfile.mkdtemp())
    original_db = tui_app.DB_PATH
    tui_app.DB_PATH = str(tmp / "empty.db")
    original = install_stubs()

    async def run():
        app = tui.HackathonApp(url=None, thread_id="")
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_for_text(pilot.app, "No previous runs")
            return log_text(pilot.app)

    try:
        text = asyncio.run(run())
    finally:
        restore(original)
        tui_app.DB_PATH = original_db
        shutil.rmtree(tmp, ignore_errors=True)

    assert "No previous runs" in text, text
    assert "python -m agent.tui <devpost-url>" in text, "should say how to start one"
    print("ok: ls on an empty database explains how to start a run")


if __name__ == "__main__":
    test_full_pipeline_through_the_tui()
    test_judge_workers_get_the_synopsis()
    test_completed_phases_maps_state_to_sidebar()
    test_resume_restores_visible_progress()
    test_ls_lists_and_resumes_previous_runs()
    test_skip_button_unsticks_a_hung_research_phase()
    test_ls_with_no_previous_runs()
    print("\nAll TUI end-to-end checks passed.")
