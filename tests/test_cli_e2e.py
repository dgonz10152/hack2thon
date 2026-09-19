"""End-to-end test: the real pipeline through the real CLI, with dummy data.

Hermetic - no Ollama, no DuckDuckGo, no opencode, no network. Stubs are placed
at the *model* boundary (the model-bound names in agent.nodes.*), not at the
node boundary, so every real node body still runs: BeautifulSoup parsing, prompt
assembly, the merge_judges/merge_ideas reducers, _resolve_selected_idea, the
choose_build_dir validation loop, and the build loop.

Run:  uv run python tests/test_cli_e2e.py
"""

import asyncio
import io
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from langchain_core.messages import AIMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import cli
from agent.nodes import build, ideas, judges
from agent.prompts import BIAS_COMPILER_SYSTEM, THEME_EXTRACTOR_SYSTEM
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
# Helpers
# --------------------------------------------------------------------------


class ScriptedUser:
    """Stands in for the person at the terminal: replays fixed answers.

    Records every question so the test can check what was asked. Raises
    KeyboardInterrupt when it runs out of answers, like pressing Ctrl+C.
    """

    def __init__(self, *answers):
        self.answers = list(answers)
        self.questions = []

    def __call__(self, payload):
        self.questions.append(payload)
        if not self.answers:
            raise KeyboardInterrupt
        return self.answers.pop(0)


def run_cli(url, thread_id, user, db_path) -> str:
    """Run the CLI end to end and return everything it printed."""
    printed = io.StringIO()
    with redirect_stdout(printed):
        asyncio.run(cli.run(url, thread_id, ask=user, db_path=db_path))
    return printed.getvalue()


async def final_state(db_path, thread_id):
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        graph = cli.build_graph(saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values


# --------------------------------------------------------------------------
# The tests
# --------------------------------------------------------------------------


def test_full_pipeline_through_the_cli():
    tmp = Path(tempfile.mkdtemp())
    db_path = str(tmp / "e2e.db")
    build_dir = tmp / "built"
    user = ScriptedUser(
        "EDITED. BIAS",  # review_bias: replace the draft
        "2",  # select_idea: the second idea
        "relative/path",  # choose_build_dir: rejected, must re-prompt
        str(build_dir),  # choose_build_dir: accepted
    )
    SEEN.clear()
    # install_stubs must be called exactly once: a second call would capture the
    # stubs as the "originals" and restore() would leave agent.nodes patched.
    original = install_stubs()
    try:
        text = run_cli("https://dummy.devpost.com", "e2e", user, db_path)
        state = asyncio.run(final_state(db_path, "e2e"))
    finally:
        restore(original)
        shutil.rmtree(tmp, ignore_errors=True)

    # run() prints failures instead of raising, so check this FIRST or a broken
    # stub would look like a quiet pass.
    assert "Stopped:" not in text, text

    # Every phase was reached and announced.
    for _, label in cli.PHASES:
        assert f"=== {label} ===" in text, f"missing phase {label}:\n{text}"

    # The questions came in order, and the bad path was re-asked with an error.
    bias, idea, path, retry = user.questions
    assert "BIAS: likes agentic systems" in bias["draft_bias"], bias
    assert len(idea["ideas"].splitlines()) == len(RANKED_TITLES), idea
    assert "error" not in path, path
    assert "absolute" in retry["error"], retry

    # The bias edit round-tripped through the interrupt.
    assert state["judge_bias"] == "EDITED. BIAS", state["judge_bias"]

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
    assert "Wrote file successfully." in text, "sub-agent output should stream"
    assert text.rstrip().endswith(state["build_result"].rstrip()), text
    print("ok: full pipeline runs through the CLI on dummy data")


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


def test_completed_phases_from_checkpoint_state():
    """Unit check for the resume mapping, including the review_bias special case."""
    done = cli.completed_phases({})
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
    done = cli.completed_phases(partial)
    assert "research_one_judge" in done, done
    assert "compile_bias" in done
    # review_bias has no output of its own, so it is only "done" once the next
    # phase has produced something.
    assert "review_bias" not in done, done

    partial["idea_candidates"] = [Idea(title="I", pitch="p")]
    done = cli.completed_phases(partial)
    assert "review_bias" in done, done
    print("ok: checkpoint state maps to the right completed phases")


def test_resume_picks_up_where_it_stopped():
    tmp = Path(tempfile.mkdtemp())
    db_path = str(tmp / "resume.db")
    SEEN.clear()
    original = install_stubs()
    try:
        # Run 1: Ctrl+C at the first question.
        first = run_cli("https://dummy.devpost.com", "resume", ScriptedUser(), db_path)
        # Run 2: resume by thread id alone, as `--thread <id>` does.
        user = ScriptedUser()
        resumed = run_cli(None, "resume", user, db_path)
        # Run 3: an id that was never checkpointed.
        unknown = run_cli(None, "nope", ScriptedUser(), db_path)
    finally:
        restore(original)
        shutil.rmtree(tmp, ignore_errors=True)

    assert "Resume with: python -m agent.cli --thread resume" in first, first
    # The work already done is summarised, and the pending question is asked again.
    assert "Resumed thread resume" in resumed, resumed
    assert "judges: 2/2 researched" in resumed, resumed
    assert "draft_bias" in user.questions[0], user.questions
    assert "=== Fetch page ===" not in resumed, "resume must not restart the run"

    # An unknown id must say so rather than sitting there doing nothing.
    assert "No checkpoint found" in unknown, unknown
    assert "Resume with" not in unknown, "must not suggest reusing a bad id"
    print("ok: resume summarises past work and re-asks the pending question")


class _ExplodingGraph:
    """Wraps a real graph but blows up on one thread, like an old checkpoint."""

    def __init__(self, real, bad_thread):
        self.real = real
        self.bad = bad_thread

    async def aget_state(self, config):
        if config["configurable"]["thread_id"] == self.bad:
            raise RuntimeError("InvalidUpdateError: At key 'judge_bias'")
        return await self.real.aget_state(config)


async def list_runs(db_path):
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        graph = cli.build_graph(saver)
        rows = await cli.list_threads(saver, graph)
        # One unreadable thread must not take out the listing.
        degraded = await cli.list_threads(saver, _ExplodingGraph(graph, "run-two"))
    return rows, degraded


def test_ls_lists_previous_runs():
    tmp = Path(tempfile.mkdtemp())
    db_path = str(tmp / "ls.db")
    SEEN.clear()
    original = install_stubs()
    try:
        # Two runs, both stopped at the first question.
        for thread_id in ("run-one", "run-two"):
            run_cli("https://dummy.devpost.com", thread_id, ScriptedUser(), db_path)
        rows, degraded = asyncio.run(list_runs(db_path))
        printed = io.StringIO()
        with redirect_stdout(printed):
            asyncio.run(cli.print_threads(db_path))
    finally:
        restore(original)
        shutil.rmtree(tmp, ignore_errors=True)

    ids = [r["thread_id"] for r in rows]
    assert ids == ["run-two", "run-one"], f"newest run must come first: {ids}"

    top = rows[0]
    assert top["url"] == "https://dummy.devpost.com", top
    assert top["status"] == "waiting for you", top
    assert 0 < top["done"] < len(cli.PHASES), top
    assert top["ts"], "each row needs a timestamp"

    # A thread the current graph cannot read is flagged, not fatal.
    statuses = {r["thread_id"]: r["status"] for r in degraded}
    assert statuses == {"run-one": "waiting for you", "run-two": "unreadable"}, degraded

    text = printed.getvalue()
    assert "run-one" in text and "run-two" in text, text
    print("ok: ls lists previous runs and survives an unreadable one")


def test_ls_with_no_previous_runs():
    with tempfile.TemporaryDirectory() as tmp:
        printed = io.StringIO()
        with redirect_stdout(printed):
            asyncio.run(cli.print_threads(str(Path(tmp) / "empty.db")))

    text = printed.getvalue()
    assert "No previous runs" in text, text
    assert "python -m agent.cli <devpost-url>" in text, "should say how to start one"
    print("ok: ls on an empty database explains how to start a run")


if __name__ == "__main__":
    test_full_pipeline_through_the_cli()
    test_judge_workers_get_the_synopsis()
    test_completed_phases_from_checkpoint_state()
    test_resume_picks_up_where_it_stopped()
    test_ls_lists_previous_runs()
    test_ls_with_no_previous_runs()
    print("\nAll CLI end-to-end checks passed.")
