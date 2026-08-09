import asyncio
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send, interrupt

from agent.model import model
from agent.prompts import (
    BIAS_COMPILER_SYSTEM,
    BUILD_PLANNER_SYSTEM,
    IDEA_COMPILER_SYSTEM,
    IDEA_GENERATOR_SYSTEM,
    IDEA_RANKER_SYSTEM,
    IDEA_RESEARCHER_SYSTEM,
    JUDGE_EXTRACTOR_SYSTEM,
    JUDGE_RESEARCHER_SYSTEM,
    THEME_EXTRACTOR_SYSTEM,
)
from agent.state import (
    AgentState,
    AgentStateUpdate,
    BuildPlan,
    BuildTask,
    FinalIdea,
    FinalIdeas,
    IdeaCandidates,
    IdeaResearchState,
    Judge,
    JudgeResearchState,
    Judges,
)
from agent.tools import fetch_html, search_web

# Ollama Cloud drops connections mid-response (`httpx.ReadError`, often with an
# empty message), which used to abort runs already 20+ minutes in. TransportError
# is the base for ReadError, ConnectError, the timeouts and RemoteProtocolError.
_TRANSIENT = (httpx.TransportError, ConnectionError, TimeoutError)


def _resilient(runnable):
    """Retry a model call on transient network failures.

    Applied to each bound object rather than to `model` itself: wrapping the
    chat model returns a RunnableRetry, which no longer has
    `with_structured_output`.
    """
    return runnable.with_retry(
        retry_if_exception_type=_TRANSIENT,
        stop_after_attempt=int(os.getenv("MODEL_RETRIES", "3")),
        wait_exponential_jitter=True,
    )


chat = _resilient(model)  # the two direct model.invoke call sites
judge_extractor = _resilient(model.with_structured_output(Judges))
judge_researcher = _resilient(create_react_agent(model, tools=[search_web]))

idea_generator = _resilient(model.with_structured_output(IdeaCandidates))
idea_ranker = _resilient(model.with_structured_output(IdeaCandidates))
idea_compiler = _resilient(model.with_structured_output(FinalIdeas))
build_planner = _resilient(model.with_structured_output(BuildPlan))
deep_researcher = _resilient(create_react_agent(model, tools=[search_web]))

# Provider + permission config for the opencode sub-agents. Passed via
# OPENCODE_CONFIG because they run with --dir set to the user's build directory,
# where this repo's project config would not be picked up.
_OPENCODE_CONFIG = Path(__file__).resolve().parents[2] / "opencode.json"

# Placeholder written when a fan-out worker exhausts its retries. Downstream
# prompts already handle a missing profile, and this states plainly that nothing
# was found so the compiler does not treat silence as a finding.
_UNAVAILABLE = "({kind} unavailable after retries: {err}. No information gathered.)"
_SKIPPED = "({kind} skipped at the user's request. No information gathered.)"

# LangGraph's default recursion limit is 25 supersteps, roughly a dozen tool
# calls, which a research agent can legitimately exceed. Raised, but still
# bounded so a genuinely stuck agent fails instead of running forever.
_RESEARCH_CONFIG = {"recursion_limit": int(os.getenv("RESEARCH_RECURSION_LIMIT", "40"))}

# Nothing else bounds a model call, so a hung request stalls the phase forever:
# retries and degradation only fire on errors, never on silence.
_RESEARCH_TIMEOUT = float(os.getenv("RESEARCH_TIMEOUT", "300"))

# Set from the UI to abandon in-flight research and let the graph move on.
# A plain flag rather than an asyncio.Event: an Event created at import time
# binds its internal futures to whichever loop first waits on it, which misfires
# across loops. A polled flag has no loop affinity.
_SKIP = False


class SkippedByUser(Exception):
    """Raised in a worker when the user asks to move on."""


def request_skip() -> None:
    """Abandon in-flight research workers. Each degrades and the phase ends."""
    global _SKIP
    _SKIP = True


def clear_skip() -> None:
    """Re-arm skipping once the phase has moved on."""
    global _SKIP
    _SKIP = False


def skip_requested() -> bool:
    return _SKIP


async def _watch_for_skip(poll: float = 0.05) -> None:
    while not _SKIP:
        await asyncio.sleep(poll)


async def _bounded_research(coro, timeout: float):
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


_JUDGE_RESEARCH_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("JUDGE_RESEARCH_CONCURRENCY", "2"))
)
_IDEA_RESEARCH_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("IDEA_RESEARCH_CONCURRENCY", "2"))
)


def fetch_html_node(state: AgentState) -> AgentStateUpdate:
    html = fetch_html.invoke({"url": state["devpost_url"]})
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return {"html": text}


def extract_theme_node(state: AgentState) -> AgentStateUpdate:
    system = SystemMessage(content=THEME_EXTRACTOR_SYSTEM)
    message = HumanMessage(content=state["html"])
    result = chat.invoke([system, message])
    return {"hackathon_synposis": result.content}


def extract_judges_node(state: AgentState) -> AgentStateUpdate:
    system = SystemMessage(content=JUDGE_EXTRACTOR_SYSTEM)
    user = HumanMessage(content=state["html"])
    result: Judges = judge_extractor.invoke([system, user])
    return {"judges": result.judges, "messages": [system, user]}


def research_judges_orchestrator(state: AgentState) -> list[Send]:
    synopsis = state.get("hackathon_synposis", "")
    return [
        Send("research_one_judge", {"judge": j, "hackathon_synposis": synopsis})
        for j in state["judges"]
    ]


async def research_one_judge_node(state: JudgeResearchState) -> AgentStateUpdate:
    judge = state["judge"]
    synopsis = state.get("hackathon_synposis", "")
    user_content = (
        f"Judge name: {judge.name}\n"
        f"Page blurb: {judge.blurb or '(none)'}\n\n"
        f"Hackathon synopsis (for context):\n{synopsis}\n\n"
        f"Research this judge and return the 4–8 sentence summary."
    )
    try:
        async with _JUDGE_RESEARCH_SEMAPHORE:
            result = await _bounded_research(
                judge_researcher.ainvoke(
                    {
                        "messages": [
                            SystemMessage(content=JUDGE_RESEARCHER_SYSTEM),
                            HumanMessage(content=user_content),
                        ]
                    },
                    config=_RESEARCH_CONFIG,
                ),
                _RESEARCH_TIMEOUT,
            )
        summary = result["messages"][-1].content
    except SkippedByUser:
        summary = _SKIPPED.format(kind="Judge research")
        _say(f"  - skipped judge research for {judge.name}")
    except Exception as exc:
        # One worker out of N must not lose the whole run. The placeholder is
        # non-empty on purpose: merge_judges treats "" as "no update", so an
        # empty string would look like the research simply never happened.
        summary = _UNAVAILABLE.format(kind="Judge research", err=type(exc).__name__)
        _say(f"  ! judge research failed for {judge.name}: {exc!r} - continuing")
    enriched = judge.model_copy(update={"online_summary": summary})
    return {"judges": [enriched]}


def compile_bias_node(state: AgentState) -> AgentStateUpdate:
    judges = state.get("judges", [])
    synopsis = state.get("hackathon_synposis", "")
    profiles = "\n\n".join(
        f"--- {j.name} ---\nPage blurb: {j.blurb or '(none)'}\nResearch summary: {j.online_summary or '(no research)'}"
        for j in judges
    )
    user_content = f"Hackathon synopsis:\n{synopsis}\n\n" f"Judge profiles:\n{profiles}"
    result = chat.invoke(
        [
            SystemMessage(content=BIAS_COMPILER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"judge_bias": result.content}


def review_bias_node(state: AgentState) -> AgentStateUpdate:
    draft = state.get("judge_bias", "")
    approved = interrupt(
        {
            "draft_bias": draft,
            "instructions": "Edit the analysis or return it unchanged to approve.",
        }
    )
    return {"judge_bias": approved}


def generate_idea_candidates_node(state: AgentState) -> AgentStateUpdate:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge bias analysis:\n{bias}\n\n"
        f"Generate the diverse pool of candidate app ideas."
    )
    result: IdeaCandidates = idea_generator.invoke(
        [
            SystemMessage(content=IDEA_GENERATOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"idea_candidates": result.ideas}


def rank_ideas_node(state: AgentState) -> AgentStateUpdate:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    candidates = state.get("idea_candidates", [])
    listing = "\n".join(
        f"- {i.title}: {i.pitch} (rationale: {i.rationale or 'n/a'})"
        for i in candidates
    )
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge bias analysis:\n{bias}\n\n"
        f"Candidate ideas:\n{listing}\n\n"
        f"Select the best 20, returning them unchanged."
    )
    result: IdeaCandidates = idea_ranker.invoke(
        [
            SystemMessage(content=IDEA_RANKER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"ideas": result.ideas}


def research_ideas_orchestrator(state: AgentState) -> list[Send]:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    return [
        Send(
            "research_one_idea",
            {"idea": i, "hackathon_synposis": synopsis, "judge_bias": bias},
        )
        for i in state["ideas"]
    ]


async def research_one_idea_node(state: IdeaResearchState) -> AgentStateUpdate:
    idea = state["idea"]
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    user_content = (
        f"Idea title: {idea.title}\n"
        f"Pitch: {idea.pitch}\n\n"
        f"Hackathon synopsis (for context):\n{synopsis}\n\n"
        f"Judge bias analysis (for context):\n{bias}\n\n"
        f"Research this idea and return the Feasibility / Existing Projects / "
        f"Winning Formula summary."
    )
    try:
        async with _IDEA_RESEARCH_SEMAPHORE:
            result = await _bounded_research(
                deep_researcher.ainvoke(
                    {
                        "messages": [
                            SystemMessage(content=IDEA_RESEARCHER_SYSTEM),
                            HumanMessage(content=user_content),
                        ]
                    },
                    config=_RESEARCH_CONFIG,
                ),
                _RESEARCH_TIMEOUT,
            )
        summary = result["messages"][-1].content
    except SkippedByUser:
        summary = _SKIPPED.format(kind="Idea research")
        _say(f"  - skipped idea research for {idea.title}")
    except Exception as exc:
        # Same as judge research: degrade this one idea, keep the other 19.
        summary = _UNAVAILABLE.format(kind="Idea research", err=type(exc).__name__)
        _say(f"  ! idea research failed for {idea.title}: {exc!r} - continuing")
    enriched = idea.model_copy(update={"research": summary})
    return {"ideas": [enriched]}


def compile_ideas_node(state: AgentState) -> AgentStateUpdate:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    ideas = state.get("ideas", [])
    researched = "\n\n".join(
        f"--- {i.title} ---\nPitch: {i.pitch}\nResearch:\n{i.research or '(no research)'}"
        for i in ideas
    )
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge bias analysis:\n{bias}\n\n"
        f"Researched ideas:\n{researched}"
    )
    result: FinalIdeas = idea_compiler.invoke(
        [
            SystemMessage(content=IDEA_COMPILER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"final_ideas": result.ideas}


def select_idea_node(state: AgentState) -> AgentStateUpdate:
    ideas = state.get("final_ideas", [])
    options = "\n".join(
        f"{n}. {i.title} — {i.problem}" for n, i in enumerate(ideas, start=1)
    )
    selected = interrupt(
        {
            "ideas": options,
            "instructions": (
                "Pick an idea by its number, or write your own idea description "
                "to use instead."
            ),
        }
    )
    return {"selected_idea": selected}


def _resolve_selected_idea(state: AgentState) -> str:
    """Turn the raw `select_idea` interrupt response into a planner description.

    `selected_idea` is either a 1-based number indexing `final_ideas`, or
    free-text the user wrote. For a number we serialize the full FinalIdea;
    otherwise we pass the text straight through.
    """
    selected = (state.get("selected_idea") or "").strip()
    final_ideas = state.get("final_ideas", [])
    if selected.isdigit():
        index = int(selected) - 1
        if 0 <= index < len(final_ideas):
            idea: FinalIdea = final_ideas[index]
            return (
                f"Title: {idea.title}\n"
                f"Problem: {idea.problem}\n"
                f"Target users: {idea.target_users}\n"
                f"Core features:\n- " + "\n- ".join(idea.core_features) + "\n"
                f"Tech stack: {', '.join(idea.tech_stack)}\n"
                f"Differentiation: {idea.differentiation}\n"
                f"Bias alignment: {idea.bias_alignment}\n"
                f"Winning-formula notes: {idea.winning_formula_notes}\n"
                f"Feasibility notes: {idea.feasibility_notes}"
            )
    return selected


def plan_build_node(state: AgentState) -> AgentStateUpdate:
    """Master agent: decompose the selected idea into an ordered task list."""
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    idea_description = _resolve_selected_idea(state)
    user_content = (
        f"Selected idea:\n{idea_description}\n\n"
        f"Hackathon synopsis (for context):\n{synopsis}\n\n"
        f"Judge bias analysis (for context):\n{bias}\n\n"
        f"Produce the build plan with an ordered list of atomic tasks."
    )
    plan: BuildPlan = build_planner.invoke(
        [
            SystemMessage(content=BUILD_PLANNER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"build_plan": plan, "build_tasks": plan.tasks}


def _git(build_dir: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", build_dir, *args], capture_output=True, text=True
    )


def choose_build_dir_node(state: AgentState) -> AgentStateUpdate:
    """Ask the user where to build, validate it, and git-init it.

    Trust boundary: the coding sub-agents get write and shell access to whatever
    path this returns, so a non-empty directory needs explicit confirmation. The
    `git init` means every task's output can be diffed and reverted.
    """
    plan = state.get("build_plan")
    tasks = state.get("build_tasks", [])
    payload = {
        "project_summary": plan.project_summary if plan else "",
        "tech_stack": ", ".join(plan.tech_stack) if plan else "",
        "tasks": [f"{n}. {t.title}" for n, t in enumerate(tasks, start=1)],
        "instructions": (
            "Absolute path to the directory to build in. It will be created if "
            "it does not exist, and git-initialized so you can diff and revert."
        ),
    }
    error = ""
    while True:
        raw = interrupt({**payload, "error": error} if error else payload)
        candidate = str(raw or "").strip()
        if not candidate:
            error = "No path given. Enter an absolute path."
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            error = f"'{candidate}' is not an absolute path."
            continue
        if path.exists() and not path.is_dir():
            error = f"'{path}' exists but is not a directory."
            continue
        if path.is_dir() and any(path.iterdir()):
            answer = interrupt(
                {
                    "warning": (
                        f"{path} is not empty. Coding agents will write into it "
                        f"and may overwrite existing files."
                    ),
                    "instructions": "Type 'yes' to use it anyway, or anything else to pick another path.",
                }
            )
            if str(answer or "").strip().lower() not in {"yes", "y"}:
                error = "Pick a different path."
                continue
        path.mkdir(parents=True, exist_ok=True)
        if not (path / ".git").is_dir():
            _git(str(path), "init", "-q")
        return {"build_dir": str(path)}


def _task_prompt(
    plan: BuildPlan, task: BuildTask, n: int, total: int, done: list[str]
) -> str:
    """Build the self-contained prompt for one coding sub-agent.

    Each task gets a fresh opencode session, so everything it needs must be here
    or already on disk.
    """
    parts = [
        f"You are building this project:\n{plan.project_summary}",
        f"Tech stack: {', '.join(plan.tech_stack)}",
    ]
    if plan.setup_notes:
        parts.append(f"Setup notes:\n{plan.setup_notes}")
    if done:
        parts.append(
            "Already completed by earlier agents (the files are on disk):\n"
            + "\n".join(f"- {t}" for t in done)
        )
    parts.append(f"Your task ({n} of {total}): {task.title}\n\n{task.description}")
    parts.append(f"Done when: {task.acceptance or 'the task is implemented and runs.'}")
    parts.append("Work in the current directory. Do only this task, then stop.")
    return "\n\n".join(parts)


_say_sink = None


def set_say_sink(sink) -> None:
    """Redirect build progress output. The TUI captures it; None restores print."""
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


def _has_changes(build_dir: str) -> bool:
    """True if the sub-agent actually touched the working tree."""
    return bool(_git(build_dir, "status", "--porcelain").stdout.strip())


async def _run_sub_agent(
    prompt: str, build_dir: str, model_id: str, timeout: int, env: dict
) -> tuple[str, str]:
    """Run one opencode session. Returns (outcome, output).

    Outcome is "ok", "timeout", or "exit:<code>".
    """
    proc = await asyncio.create_subprocess_exec(
        "opencode",
        "run",
        "--dir",
        build_dir,
        "--model",
        model_id,
        prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )

    # Collected outside the pump so partial output survives a timeout kill.
    lines: list[str] = []

    async def pump() -> None:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            _say(f"  | {line}")
            lines.append(line)
        await proc.wait()

    try:
        await asyncio.wait_for(pump(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "timeout", "\n".join(lines)
    output = "\n".join(lines)
    if proc.returncode != 0:
        return f"exit:{proc.returncode}", output
    return "ok", output


async def build_app_node(state: AgentState) -> AgentStateUpdate:
    """Run one opencode sub-agent per task, in order, in the user's build dir.

    Sequential on purpose: task N+1 reads the files task N wrote, so parallel
    agents in one directory would clobber each other. Stops at the first failure
    because everything after it would build on a broken base.

    A sub-agent can exit 0 having done nothing at all (observed with
    ollama/minimax-m3, which intermittently returns an empty response), so exit
    status alone is not proof of completion. Every task is verified against the
    git working tree and retried once if it changed no files.
    """
    plan = state.get("build_plan")
    tasks = [t.model_copy() for t in state.get("build_tasks", [])]
    build_dir = state.get("build_dir", "")
    if not plan or not tasks or not build_dir:
        return {"build_result": "No build plan, tasks, or build dir; skipped build."}
    if shutil.which("opencode") is None:
        return {"build_result": "Build failed: 'opencode' not found on PATH."}

    model_id = os.getenv("BUILD_MODEL", "ollama/minimax-m3")
    timeout = int(os.getenv("BUILD_TASK_TIMEOUT", "900"))
    env = {**os.environ, "OPENCODE_CONFIG": str(_OPENCODE_CONFIG)}

    total = len(tasks)
    _say(f"\n=== Building in {build_dir} ===")
    _say(f"{plan.project_summary}")
    _say(f"Stack: {', '.join(plan.tech_stack)}")
    _say(f"{total} tasks, model {model_id}, {timeout}s per task\n")

    log: list[str] = []
    done: list[str] = []
    for n, task in enumerate(tasks, start=1):
        prompt = _task_prompt(plan, task, n, total, done)
        _say(f"--- [{n}/{total}] {task.title} ---")
        outcome, output = await _run_sub_agent(
            prompt, build_dir, model_id, timeout, env
        )
        if outcome == "ok" and not _has_changes(build_dir):
            _say(f"[{n}/{total}] no files changed, retrying once")
            log.append(f"{n}. {task.title} - no changes on first try, retrying")
            outcome, output = await _run_sub_agent(
                prompt, build_dir, model_id, timeout, env
            )
            if outcome == "ok" and not _has_changes(build_dir):
                outcome = "no-op"

        if outcome != "ok":
            task.status = "failed"
            reason = {
                "timeout": f"TIMED OUT after {timeout}s",
                "no-op": "sub-agent made no file changes (twice)",
            }.get(outcome, f"FAILED ({outcome})")
            log.append(f"{n}. {task.title} - {reason}")
            _say(f"[{n}/{total}] {reason}")
            _say(f"=== Build stopped at task {n}. Output left in {build_dir} ===\n")
            return {
                "build_tasks": tasks,
                "build_result": _build_summary(
                    build_dir, log, failed=task.title, tail=output[-2000:]
                ),
            }

        task.status = "ok"
        done.append(task.title)
        log.append(f"{n}. {task.title} - ok")
        _git(build_dir, "add", "-A")
        _git(build_dir, "commit", "-q", "-m", f"task {n}: {task.title}")
        changed = _git(build_dir, "show", "--stat", "--format=", "HEAD").stdout.strip()
        _say(f"[{n}/{total}] done, committed:\n{changed}\n")

    _say(f"=== Build finished. {total} tasks, output in {build_dir} ===\n")
    return {"build_tasks": tasks, "build_result": _build_summary(build_dir, log)}


def _build_summary(
    build_dir: str, log: list[str], failed: str = "", tail: str = ""
) -> str:
    lines = [
        f"Build {'failed' if failed else 'finished'}. Output in {build_dir}.",
        "",
        "Tasks:",
        *log,
    ]
    if failed:
        lines += ["", f"Stopped at: {failed}"]
    if tail:
        lines += ["", "Last output:", tail]
    return "\n".join(lines)
