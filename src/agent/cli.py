"""Plain terminal driver for the whole pipeline, from DevPost URL to built app.

    uv run python -m agent.cli https://foo.devpost.com   # new run
    uv run python -m agent.cli ls                        # list past runs
    uv run python -m agent.cli --thread <id>             # resume a run

Runs the graph in-process with a SQLite checkpointer, so a run survives quitting
or crashing and can be resumed by thread id. The three interrupt nodes are
answered with `input()`.
"""

import argparse
import asyncio
import os
import subprocess
import tempfile
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agent.graph import build_graph

DB_PATH = ".runs.db"

# Graph node name -> printed label. Order is pipeline order.
PHASES: list[tuple[str, str]] = [
    ("fetch_html", "Fetch page"),
    ("extract_theme", "Extract theme"),
    ("extract_judges", "Extract judges"),
    ("research_one_judge", "Research judges"),
    ("compile_bias", "Compile bias"),
    ("review_bias", "Review bias"),
    ("generate_idea_candidates", "Brainstorm ideas"),
    ("rank_ideas", "Rank top 20"),
    ("research_one_idea", "Research ideas"),
    ("compile_ideas", "Flesh out ideas"),
    ("select_idea", "Select idea"),
    ("plan_build", "Plan build"),
    ("choose_build_dir", "Choose directory"),
    ("build_app", "Build app"),
]
LABELS = dict(PHASES)


# --------------------------------------------------------------------------
# Asking the user
# --------------------------------------------------------------------------


def ask(payload) -> str:
    """Answer an interrupt by asking on the terminal.

    Dispatches on payload *shape*, not node name, because choose_build_dir emits
    two different payloads (path prompt and non-empty confirmation) and can emit
    either repeatedly within one node run.
    """
    if not isinstance(payload, dict):
        return input(f"{payload}\n> ")
    if "draft_bias" in payload:
        return ask_bias(payload)
    if "ideas" in payload:
        print(f"\n{payload['ideas']}\n")
        print(payload.get("instructions", ""))
        return input("number or your own idea: ")
    if "warning" in payload:
        print(f"\n{payload['warning']}")
        return input("use it anyway? (yes/no): ")
    return ask_path(payload)


def ask_bias(payload: dict) -> str:
    draft = payload["draft_bias"]
    print(f"\n{draft}\n")
    choice = input("Enter to approve, or 'e' to edit in $EDITOR: ").strip().lower()
    if choice != "e":
        return draft
    return edit_in_editor(draft)


def edit_in_editor(text: str) -> str:
    """Open `text` in the user's editor and return what they saved."""
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        subprocess.run([os.environ.get("EDITOR", "vi"), path], check=True)
        with open(path) as f:
            return f.read()
    finally:
        os.remove(path)


def ask_path(payload: dict) -> str:
    print()
    if summary := payload.get("project_summary"):
        print(summary)
    if stack := payload.get("tech_stack"):
        print(f"Stack: {stack}")
    for task in payload.get("tasks", []):
        print(f"  {task}")
    if error := payload.get("error"):
        print(f"! {error}")
    print(payload.get("instructions", ""))
    return input("absolute path: ")


# --------------------------------------------------------------------------
# Running the graph
# --------------------------------------------------------------------------


async def drive(graph, config, payload, ask=ask) -> None:
    """Stream the graph, answering each interrupt, until it finishes."""
    while True:
        pending = None
        async for event in graph.astream(payload, config, stream_mode="updates"):
            if "__interrupt__" in event:
                pending = event["__interrupt__"][0].value
                continue
            for node in event:
                # Skip LangGraph's own bookkeeping keys (__metadata__, ...).
                if not node.startswith("__"):
                    print(f"=== {LABELS.get(node, node)} ===", flush=True)
        # Decided on the stream, never on state.next: a second interrupt inside
        # one node leaves state.next empty while the graph is still waiting.
        if pending is None:
            return
        payload = Command(resume=ask(pending))


async def load_checkpoint(graph, config):
    """Print what a resumed run already did. Returns None if it cannot resume."""
    thread_id = config["configurable"]["thread_id"]
    try:
        state = await graph.aget_state(config)
    except Exception as exc:
        print(f"Cannot read thread {thread_id}: {exc}")
        print("It was checkpointed by an older version of the graph. Start a new run.")
        return None

    if not state.values:
        print(f"No checkpoint found for thread {thread_id}.")
        print("Check the id, or start a new run with a DevPost URL.")
        return None

    print(f"=== Resumed thread {thread_id} ===")
    for line in resume_summary(state.values):
        print(line)
    print(f"  picking up at: {', '.join(sorted(set(state.next))) or 'none'}\n")
    return state


async def run(url, thread_id, ask=ask, db_path=DB_PATH) -> None:
    print(f"thread {thread_id}")
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            if url:
                payload = {"devpost_url": url}
            else:
                state = await load_checkpoint(graph, config)
                if state is None:
                    return
                # Answer the question it stopped on; with no pending question,
                # streaming None continues from the last checkpoint.
                pending = state.interrupts[0].value if state.interrupts else None
                payload = None if pending is None else Command(resume=ask(pending))
            await drive(graph, config, payload, ask)
        except (Exception, KeyboardInterrupt, EOFError) as exc:
            # The run is long and the backend is flaky, but the checkpoint is
            # still there, so always say how to pick it back up.
            print(f"\nStopped: {type(exc).__name__}: {exc}")
            print(f"Resume with: python -m agent.cli --thread {thread_id}")
            return

        state = await graph.aget_state(config)
        print()
        print(state.values.get("build_result", "Pipeline finished."))


# --------------------------------------------------------------------------
# Past runs
# --------------------------------------------------------------------------


def completed_phases(values: dict) -> set[str]:
    """Work out which phases already ran, from a checkpoint's state values.

    Each phase is inferred from the state it leaves behind. `review_bias` has no
    output of its own (it overwrites `judge_bias`), so it is inferred from the
    next phase's output instead.
    """
    judges = values.get("judges") or []
    ideas = values.get("ideas") or []
    present = {
        "fetch_html": values.get("html"),
        "extract_theme": values.get("hackathon_synopsis"),
        "extract_judges": judges,
        "research_one_judge": any(j.online_summary for j in judges),
        "compile_bias": values.get("judge_bias"),
        "review_bias": values.get("idea_candidates"),
        "generate_idea_candidates": values.get("idea_candidates"),
        "rank_ideas": ideas,
        "research_one_idea": any(i.research for i in ideas),
        "compile_ideas": values.get("final_ideas"),
        "select_idea": values.get("selected_idea"),
        "plan_build": values.get("build_plan"),
        "choose_build_dir": values.get("build_dir"),
        "build_app": values.get("build_result"),
    }
    return {name for name, value in present.items() if value}


def resume_summary(values: dict) -> list[str]:
    """Human-readable account of what was restored from the checkpoint."""
    lines = []
    if url := values.get("devpost_url"):
        lines.append(f"  page: {url}")
    if judges := values.get("judges"):
        researched = sum(1 for j in judges if j.online_summary)
        lines.append(f"  judges: {researched}/{len(judges)} researched")
    if values.get("judge_bias"):
        lines.append("  bias analysis: done")
    if ideas := values.get("ideas"):
        researched = sum(1 for i in ideas if i.research)
        lines.append(f"  ideas: {researched}/{len(ideas)} researched")
    if selected := values.get("selected_idea"):
        lines.append(f"  selected idea: {selected}")
    if build_dir := values.get("build_dir"):
        lines.append(f"  build dir: {build_dir}")
    return lines


async def list_threads(saver, graph) -> list[dict]:
    """Summarize every run in the checkpoint database, newest first.

    `alist(None)` walks checkpoints across all threads newest-first, so the first
    sighting of a thread id is its latest checkpoint. Values come from
    `aget_state` rather than the raw checkpoint, because a single checkpoint
    only holds the channels that step wrote.
    """
    latest: dict[str, str] = {}
    async for checkpoint in saver.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in latest:
            latest[thread_id] = str(checkpoint.checkpoint.get("ts", ""))[:19]

    rows = []
    for thread_id, timestamp in latest.items():
        row = {"thread_id": thread_id, "ts": timestamp.replace("T", " ")}
        try:
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        except Exception:
            # A run checkpointed by an older, broken graph can raise here.
            # One bad thread must not take out the whole listing.
            rows.append({**row, "url": "", "done": 0, "status": "unreadable"})
            continue
        values = state.values or {}
        if state.interrupts:
            status = "waiting for you"
        elif values.get("build_result"):
            status = "finished"
        else:
            status = "unfinished"
        rows.append(
            {
                **row,
                "url": values.get("devpost_url", ""),
                "done": len(completed_phases(values)),
                "status": status,
            }
        )
    return rows


async def print_threads(db_path=DB_PATH) -> None:
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        rows = await list_threads(saver, build_graph(saver))
    if not rows:
        print(f"No previous runs in {db_path}.")
        print("Start one with: python -m agent.cli <devpost-url>")
        return
    for row in rows:
        print(
            f"{row['ts']}  {row['done']:>2}/{len(PHASES)}  {row['status']:<15}  "
            f"{row['thread_id']}  {row['url']}"
        )
    print("\nResume one with: python -m agent.cli --thread <id>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hackathon pipeline")
    parser.add_argument("target", nargs="?", help="DevPost URL, or 'ls' to list runs")
    parser.add_argument("--thread", help="resume an existing run by thread id")
    args = parser.parse_args()

    if args.target == "ls":
        asyncio.run(print_threads())
    elif args.target or args.thread:
        thread_id = args.thread or str(uuid.uuid4())
        asyncio.run(run(args.target, thread_id))
    else:
        parser.error("give a DevPost URL, 'ls', or --thread <id>")


if __name__ == "__main__":
    main()
