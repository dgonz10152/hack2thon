"""Phase 3: actually build the chosen idea.

plan_build (decompose into ordered tasks) → choose_build_dir (human names a
directory) → build_app (one opencode coding sub-agent per task, sequential).

Sequential on purpose: task N+1 reads the files task N wrote, so parallel
agents in one directory would clobber each other.
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from agent.model import model, resilient
from agent.progress import _say
from agent.prompts import BUILD_PLANNER_SYSTEM
from agent.state import AgentState, BuildPlan, BuildTask, FinalIdea

build_planner = resilient(model.with_structured_output(BuildPlan))

# Provider + permission config for the opencode sub-agents. Passed via
# OPENCODE_CONFIG because they run with --dir set to the user's build directory,
# where this repo's project config would not be picked up.
_OPENCODE_CONFIG = Path(__file__).resolve().parents[3] / "opencode.json"


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


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


def plan_build_node(state: AgentState) -> AgentState:
    """Master agent: decompose the selected idea into an ordered task list."""
    user_content = (
        f"Selected idea:\n{_resolve_selected_idea(state)}\n\n"
        f"Hackathon synopsis (for context):\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Judge bias analysis (for context):\n{state.get('judge_bias', '')}\n\n"
        f"Produce the build plan with an ordered list of atomic tasks."
    )
    plan: BuildPlan = build_planner.invoke(
        [
            SystemMessage(content=BUILD_PLANNER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"build_plan": plan, "build_tasks": plan.tasks}


# --------------------------------------------------------------------------
# Choosing where to build
# --------------------------------------------------------------------------


def _git(build_dir: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", build_dir, *args], capture_output=True, text=True
    )


def _validate_dir(raw) -> tuple[Path | None, str]:
    """Check one answer to the build-dir prompt. Returns (path, error).

    Exactly one of the two is meaningful: a usable Path, or the error text to
    re-prompt with.
    """
    candidate = str(raw or "").strip()
    if not candidate:
        return None, "No path given. Enter an absolute path."
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        return None, f"'{candidate}' is not an absolute path."
    if path.exists() and not path.is_dir():
        return None, f"'{path}' exists but is not a directory."
    return path, ""


def choose_build_dir_node(state: AgentState) -> AgentState:
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
        path, error = _validate_dir(raw)
        if error:
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


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


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


async def build_app_node(state: AgentState) -> AgentState:
    """Run one opencode sub-agent per task, in order, in the user's build dir.

    Stops at the first failure because everything after it would build on a
    broken base.

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
