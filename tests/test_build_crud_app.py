"""Real end-to-end build: generates a small CRUD app into a directory you name.

Unlike tests/test_build_app.py (hermetic, mocked), this runs the actual coding
sub-agents. It costs tokens, takes minutes, and writes real files, so it only
runs when you give it a target directory:

    uv run python tests/test_build_crud_app.py ~/code/pantry
    BUILD_OUTPUT_DIR=~/code/pantry uv run python tests/test_build_crud_app.py

Needs `opencode` on PATH and a reachable Ollama (OLLAMA_* in .env, BUILD_MODEL).
The directory is created if missing and git-initialized. Without a directory it
skips, so a test collector cannot trigger a real build by accident.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from agent import nodes
from agent.state import BuildPlan, BuildTask

PLAN = BuildPlan(
    project_summary=(
        "A simple CRUD web app for food pantries. Staff can add food items they "
        "have available (name, quantity, category); anyone can view the current "
        "list. No auth, keep it minimal."
    ),
    tech_stack=["python", "fastapi", "uvicorn", "sqlite"],
    setup_notes=(
        "Single FastAPI app in main.py, SQLite via the sqlite3 stdlib module, no "
        "ORM. Install with: pip install fastapi uvicorn. Run with: uvicorn "
        "main:app --port 8000."
    ),
    tasks=[
        BuildTask(
            title="Scaffold app and storage",
            description=(
                "Create main.py with a FastAPI app and a SQLite 'items' table "
                "(id INTEGER PRIMARY KEY, name TEXT, quantity INTEGER, category "
                "TEXT) created on startup. Add a GET /health route returning "
                "{'status': 'ok'}."
            ),
            acceptance="main.py defines a FastAPI app and a GET /health route.",
        ),
        BuildTask(
            title="Add item creation",
            description=(
                "In main.py, add POST /items accepting JSON with name, quantity, "
                "and category, inserting a row into the items table. Return the "
                "created item including its id."
            ),
            acceptance="POST /items inserts a row and returns it as JSON.",
        ),
        BuildTask(
            title="Add item listing",
            description=(
                "In main.py, add GET /items returning every row in the items "
                "table as a JSON list."
            ),
            acceptance="GET /items returns all inserted items.",
        ),
        BuildTask(
            title="Verify the app runs",
            description=(
                "Create requirements.txt listing fastapi and uvicorn. Then start "
                "the app and confirm GET /health responds, POST /items creates an "
                "item, and GET /items lists it. Fix anything that fails."
            ),
            acceptance="The app starts and all three endpoints respond correctly.",
        ),
    ],
)


def _target() -> Path | None:
    raw = sys.argv[1] if len(sys.argv) > 1 else os.getenv("BUILD_OUTPUT_DIR", "")
    return Path(raw).expanduser().resolve() if raw.strip() else None


def test_build_crud_app(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if not (target / ".git").is_dir():
        nodes._git(str(target), "init", "-q")

    state = {
        "build_plan": PLAN,
        "build_tasks": PLAN.tasks,
        "build_dir": str(target),
    }
    result = asyncio.run(nodes.build_app_node(state))
    summary = result["build_result"]
    print(summary, "\n")

    tasks = result["build_tasks"]
    failed = [t.title for t in tasks if t.status != "ok"]
    assert not failed, f"tasks did not complete: {failed}\n\n{summary}"

    generated = [p for p in target.rglob("*.py") if ".git" not in p.parts]
    assert generated, f"no Python files generated in {target}"
    assert (target / "main.py").is_file(), "expected main.py at the project root"

    # Every generated file must at least parse; broken codegen fails loudly.
    for path in generated:
        subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)

    source = (target / "main.py").read_text()
    for route in ("/health", "/items"):
        assert route in source, f"{route} missing from main.py"

    commits = subprocess.run(
        ["git", "-C", str(target), "log", "--oneline"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert len(commits) >= len(
        PLAN.tasks
    ), f"expected one commit per task, got {len(commits)}:\n" + "\n".join(commits)

    print(f"ok: {len(generated)} Python files, {len(commits)} commits, all parse")
    print(f"\nBuilt in {target}. Try it with:")
    print(f"  cd {target} && pip install -r requirements.txt")
    print("  uvicorn main:app --port 8000")


if __name__ == "__main__":
    target = _target()
    if target is None:
        print(__doc__)
        print("skipped: no target directory given")
        sys.exit(0)
    test_build_crud_app(target)
    print("\nCRUD build check passed.")
