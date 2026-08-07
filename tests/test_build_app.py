"""Self-check for the build phase: build_app_node and choose_build_dir_node.

Hermetic - no model calls, no real opencode, no network. The opencode subprocess
and the git helper are faked, and `interrupt` is replaced with a scripted queue.

Run:  uv run python tests/test_build_app.py
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

from agent import nodes
from agent.state import BuildPlan, BuildTask

PLAN = BuildPlan(
    project_summary="A tiny CRUD app.",
    tech_stack=["python", "fastapi"],
    setup_notes="pip install fastapi uvicorn",
    tasks=[
        BuildTask(title="Scaffold", description="Make main.py", acceptance="it runs"),
        BuildTask(title="Add routes", description="Add /items", acceptance="200 ok"),
        BuildTask(title="Verify", description="Run it", acceptance="starts"),
    ],
)


class FakeStdout:
    """Async-iterable stand-in for proc.stdout, which build_app_node streams."""

    def __init__(self, lines, hang=False):
        self._lines = list(lines)
        self._hang = hang

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._hang:
            await asyncio.sleep(3600)
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class FakeProc:
    def __init__(self, returncode=0, output=b"done", hang=False):
        self.returncode = returncode
        self.stdout = FakeStdout([output] if output else [], hang=hang)
        self.killed = False

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


class FakeGit:
    """subprocess.CompletedProcess stand-in; _git callers read .stdout."""

    def __init__(self, stdout=""):
        self.stdout = stdout


def _install_fakes(procs, changes=None):
    """Point build_app_node at fake processes; record commands and git calls.

    `changes` scripts what _has_changes returns per call, defaulting to always
    True (every sub-agent touched the tree).
    """
    calls, gits, made = [], [], list(procs)
    scripted = list(changes) if changes is not None else None

    async def fake_exec(*args, **kwargs):
        calls.append((args, kwargs))
        return made.pop(0)

    nodes.asyncio.create_subprocess_exec = fake_exec

    def fake_git(d, *a):
        gits.append(a)
        return FakeGit(" main.py | 2 +-" if a[:1] == ("show",) else "")

    nodes._git = fake_git
    nodes._has_changes = lambda d: scripted.pop(0) if scripted else True
    nodes.shutil.which = lambda _: "/usr/bin/opencode"
    return calls, gits


def _state(tmp, tasks=None):
    return {
        "build_plan": PLAN,
        "build_tasks": tasks if tasks is not None else PLAN.tasks,
        "build_dir": str(tmp),
    }


def test_all_tasks_succeed(tmp):
    calls, gits = _install_fakes([FakeProc(), FakeProc(), FakeProc()])
    out = asyncio.run(nodes.build_app_node(_state(tmp)))

    assert len(calls) == 3, f"expected 3 sub-agents, got {len(calls)}"
    assert [t.status for t in out["build_tasks"]] == ["ok", "ok", "ok"]
    assert "Build finished" in out["build_result"]

    # Tasks run in the planned order, each as its own opencode session.
    prompts = [c[0][-1] for c in calls]
    assert "Scaffold" in prompts[0] and "1 of 3" in prompts[0]
    assert "Add routes" in prompts[1]
    # Later agents are told what earlier ones already did.
    assert "Scaffold" in prompts[1], "task 2 should list completed work"
    assert "--dir" in calls[0][0] and str(tmp) in calls[0][0]

    commits = [g for g in gits if g[0] == "commit"]
    assert len(commits) == 3, f"expected a commit per task, got {commits}"
    assert "task 1: Scaffold" in commits[0][-1]
    print("ok: all tasks succeed, ordered, one commit each")


def test_stops_on_failure(tmp):
    calls, gits = _install_fakes([FakeProc(), FakeProc(returncode=1, output=b"boom")])
    out = asyncio.run(nodes.build_app_node(_state(tmp)))

    assert len(calls) == 2, "should not run task 3 after task 2 fails"
    assert [t.status for t in out["build_tasks"]] == ["ok", "failed", ""]
    assert "Build failed" in out["build_result"]
    assert "Stopped at: Add routes" in out["build_result"]
    assert "boom" in out["build_result"], "failure output should be surfaced"
    assert len([g for g in gits if g[0] == "commit"]) == 1
    print("ok: stops at first failure, surfaces output")


def test_timeout_kills_agent(tmp):
    import os

    os.environ["BUILD_TASK_TIMEOUT"] = "1"
    hung = FakeProc(hang=True)
    calls, _ = _install_fakes([hung])
    out = asyncio.run(nodes.build_app_node(_state(tmp)))

    assert hung.killed, "a hung sub-agent must be killed"
    assert "TIMED OUT" in out["build_result"]
    assert out["build_tasks"][0].status == "failed"
    del os.environ["BUILD_TASK_TIMEOUT"]
    print("ok: hung sub-agent is killed and the build stops")


def test_silent_noop_is_retried_then_succeeds(tmp):
    """A sub-agent can exit 0 having written nothing. One retry is allowed."""
    calls, gits = _install_fakes(
        [FakeProc(), FakeProc(), FakeProc(), FakeProc()],
        changes=[False, True, True, True],  # task 1 no-ops once, then works
    )
    out = asyncio.run(nodes.build_app_node(_state(tmp)))

    assert len(calls) == 4, f"expected 3 tasks + 1 retry, got {len(calls)}"
    assert [t.status for t in out["build_tasks"]] == ["ok", "ok", "ok"]
    assert "retrying" in out["build_result"]
    assert len([g for g in gits if g[0] == "commit"]) == 3, "retry is not a 4th commit"
    print("ok: silent no-op is retried and can recover")


def test_persistent_noop_fails(tmp):
    """Exit 0 twice with no file changes means the task did not happen."""
    calls, gits = _install_fakes([FakeProc(), FakeProc()], changes=[False, False])
    out = asyncio.run(nodes.build_app_node(_state(tmp)))

    assert len(calls) == 2, "one attempt plus one retry, then stop"
    assert out["build_tasks"][0].status == "failed"
    assert "no file changes" in out["build_result"]
    assert "Stopped at: Scaffold" in out["build_result"]
    assert not [g for g in gits if g[0] == "commit"], "nothing to commit"
    print("ok: persistent no-op fails loudly instead of reporting success")


def test_missing_prerequisites(tmp):
    out = asyncio.run(nodes.build_app_node({"build_plan": PLAN, "build_tasks": []}))
    assert "skipped build" in out["build_result"]
    print("ok: missing plan/dir skips cleanly")


def _script_interrupt(answers):
    """Replace interrupt() with a scripted queue, capturing payloads."""
    seen = []
    queue = list(answers)

    def fake(payload):
        seen.append(payload)
        return queue.pop(0)

    nodes.interrupt = fake
    return seen


def test_build_dir_validation(tmp):
    target = tmp / "newproj"
    seen = _script_interrupt(["   ", "relative/path", str(target)])
    nodes._git = lambda d, *a: None

    out = nodes.choose_build_dir_node({"build_plan": PLAN, "build_tasks": PLAN.tasks})

    assert out["build_dir"] == str(target)
    assert target.is_dir(), "missing directory should be created"
    assert "error" in seen[1] and "absolute" in seen[2]["error"]
    print("ok: blank and relative paths re-prompt, missing dir is created")


def test_non_empty_dir_needs_confirmation(tmp):
    occupied = tmp / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("mine")
    empty = tmp / "empty"
    nodes._git = lambda d, *a: None

    # Decline the non-empty dir, then give a clean one.
    seen = _script_interrupt([str(occupied), "no", str(empty)])
    out = nodes.choose_build_dir_node({"build_plan": PLAN, "build_tasks": PLAN.tasks})
    assert out["build_dir"] == str(empty), "declining should not use the occupied dir"
    assert any("not empty" in str(p.get("warning", "")) for p in seen)

    # Accepting it is allowed.
    _script_interrupt([str(occupied), "yes"])
    out = nodes.choose_build_dir_node({"build_plan": PLAN, "build_tasks": PLAN.tasks})
    assert out["build_dir"] == str(occupied)
    print("ok: non-empty dir requires explicit confirmation")


if __name__ == "__main__":
    originals = (
        nodes.interrupt,
        nodes._git,
        nodes._has_changes,
        nodes.shutil.which,
        asyncio.create_subprocess_exec,
    )
    for case in (
        test_all_tasks_succeed,
        test_stops_on_failure,
        test_silent_noop_is_retried_then_succeeds,
        test_persistent_noop_fails,
        test_timeout_kills_agent,
        test_missing_prerequisites,
        test_build_dir_validation,
        test_non_empty_dir_needs_confirmation,
    ):
        tmp = Path(tempfile.mkdtemp())
        try:
            case(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            (
                nodes.interrupt,
                nodes._git,
                nodes._has_changes,
                nodes.shutil.which,
                asyncio.create_subprocess_exec,
            ) = originals
    print("\nAll build-phase checks passed.")
