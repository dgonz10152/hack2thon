"""Textual TUI that drives the whole pipeline, from DevPost URL to built app.

    uv run python -m agent.tui https://foo.devpost.com   # new run
    uv run python -m agent.tui ls                        # browse and resume past runs
    uv run python -m agent.tui --thread <id>             # resume a known thread id

Runs the graph in-process with a SQLite checkpointer, so a run survives quitting
or crashing. The three interrupt nodes surface as modal dialogs.
"""

from __future__ import annotations

import argparse
import queue
import time
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    Footer,
    RichLog,
    Static,
    TextArea,
)

from agent import nodes
from agent.graph import build_graph

DB_PATH = ".runs.db"

# Graph node name -> label in the sidebar. Order is pipeline order.
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

# These fan out via Send, so they emit one update per worker; show a count.
FAN_OUT = {"research_one_judge", "research_one_idea"}


def completed_phases(values: dict) -> tuple[set[str], dict[str, int]]:
    """Work out which phases already ran, from a checkpoint's state values.

    Needed on resume: the sidebar is otherwise driven only by events seen in the
    current process, so a resumed run looked like it had done nothing at all.

    Each phase is inferred from the state it leaves behind. `review_bias` has no
    output of its own (it overwrites `judge_bias`), so it is inferred from the
    next phase's output instead.
    """
    judges = values.get("judges") or []
    ideas = values.get("ideas") or []
    researched_judges = [j for j in judges if getattr(j, "online_summary", "")]
    researched_ideas = [i for i in ideas if getattr(i, "research", "")]

    present = {
        "fetch_html": bool(values.get("html")),
        "extract_theme": bool(values.get("hackathon_synposis")),
        "extract_judges": bool(judges),
        "research_one_judge": bool(researched_judges),
        "compile_bias": bool(values.get("judge_bias")),
        "review_bias": bool(values.get("idea_candidates")),
        "generate_idea_candidates": bool(values.get("idea_candidates")),
        "rank_ideas": bool(ideas),
        "research_one_idea": bool(researched_ideas),
        "compile_ideas": bool(values.get("final_ideas")),
        "select_idea": bool(values.get("selected_idea")),
        "plan_build": bool(values.get("build_plan")),
        "choose_build_dir": bool(values.get("build_dir")),
        "build_app": bool(values.get("build_result")),
    }
    counts = {
        "research_one_judge": len(researched_judges),
        "research_one_idea": len(researched_ideas),
    }
    return {name for name, done in present.items() if done}, counts


async def list_threads(saver, graph) -> list[dict]:
    """Summarize every run in the checkpoint database, newest first.

    `alist(None)` walks checkpoints across all threads newest-first, so the first
    sighting of a thread id is its latest checkpoint. Values come from
    `aget_state` rather than the raw checkpoint: an individual checkpoint holds
    only the channels that step wrote, so its `channel_values` is a delta, not
    the merged state.
    """
    order: list[tuple[str, str]] = []
    seen: set[str] = set()
    async for checkpoint in saver.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in seen:
            seen.add(thread_id)
            order.append((thread_id, str(checkpoint.checkpoint.get("ts", ""))[:19]))

    rows = []
    for thread_id, timestamp in order:
        try:
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        except Exception as exc:
            # Rebuilding a snapshot replays that step's pending writes, so a run
            # checkpointed by an older, broken graph can still raise here. One
            # bad thread must not take out the whole listing.
            rows.append(
                {
                    "thread_id": thread_id,
                    "ts": timestamp.replace("T", " "),
                    "url": "(unreadable)",
                    "done": 0,
                    "total": len(PHASES),
                    "status": "unreadable",
                    "next": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        values = state.values or {}
        done, _ = completed_phases(values)
        if state.interrupts:
            status = "waiting for you"
        elif values.get("build_result"):
            status = "finished"
        elif state.next:
            status = "unfinished"
        else:
            status = "idle"
        rows.append(
            {
                "thread_id": thread_id,
                "ts": timestamp.replace("T", " "),
                "url": values.get("devpost_url", "") or "(no page yet)",
                "done": len(done),
                "total": len(PHASES),
                "status": status,
                "next": ", ".join(sorted(set(state.next))),
                "error": "",
            }
        )
    return rows


class ThreadPicker(ModalScreen[str]):
    """`ls`: choose a previous run to resume."""

    def __init__(self, rows: list[dict]) -> None:
        super().__init__()
        self.rows = rows

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Previous runs", id="title")
            yield Label(
                "Enter to resume the highlighted run, escape to quit.",
                classes="hint",
            )
            yield ListView(
                *[ListItem(Label(self._describe(r))) for r in self.rows],
                id="threads",
            )

    @staticmethod
    def _describe(row: dict) -> str:
        page = row["url"].replace("https://", "").replace("http://", "").rstrip("/")
        return (
            f"{row['ts']}  {row['done']:>2}/{row['total']}  "
            f"{row['status']:<15}  {page}\n"
            f"    {row['thread_id']}"
            + (f"  (next: {row['next']})" if row["next"] else "")
        )

    def on_mount(self) -> None:
        self.query_one("#threads", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self.query_one("#threads", ListView).index or 0
        self.dismiss(self.rows[index]["thread_id"])

    def key_escape(self) -> None:
        self.dismiss("")


def resume_summary(values: dict) -> list[str]:
    """Human-readable account of what was restored from the checkpoint."""
    lines = []
    if url := values.get("devpost_url"):
        lines.append(f"  page: {url}")
    judges = values.get("judges") or []
    if judges:
        researched = sum(1 for j in judges if getattr(j, "online_summary", ""))
        lines.append(f"  judges: {researched}/{len(judges)} researched")
    if values.get("judge_bias"):
        lines.append("  bias analysis: done")
    ideas = values.get("ideas") or []
    if ideas:
        researched = sum(1 for i in ideas if getattr(i, "research", ""))
        lines.append(f"  ideas: {researched}/{len(ideas)} researched")
    if selected := values.get("selected_idea"):
        lines.append(f"  selected idea: {selected}")
    if build_dir := values.get("build_dir"):
        lines.append(f"  build dir: {build_dir}")
    return lines


# --------------------------------------------------------------------------
# Interrupt dialogs
# --------------------------------------------------------------------------


class BiasScreen(ModalScreen[str]):
    """review_bias: edit the draft analysis, or approve it unchanged."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Judge bias analysis", id="title")
            yield Label(self.payload.get("instructions", ""), classes="hint")
            yield TextArea(self.payload.get("draft_bias", ""), id="bias")
            yield Button("Approve", variant="success", id="ok")

    def on_mount(self) -> None:
        self.query_one("#bias", TextArea).focus()

    @on(Button.Pressed, "#ok")
    def approve(self) -> None:
        self.dismiss(self.query_one("#bias", TextArea).text)


class IdeaScreen(ModalScreen[str]):
    """select_idea: pick one of the ideas by number, or write your own."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload
        # select_idea_node passes a single newline-joined string, not a list.
        self.options = [
            line for line in str(payload.get("ideas", "")).splitlines() if line.strip()
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Pick an idea", id="title")
            yield Label(self.payload.get("instructions", ""), classes="hint")
            yield ListView(
                *[ListItem(Label(o)) for o in self.options],
                id="ideas",
            )
            yield Label("...or describe your own:", classes="hint")
            yield Input(placeholder="your own idea", id="own")
            yield Button("Use selected", variant="success", id="ok")

    def on_mount(self) -> None:
        self.query_one("#ideas", ListView).focus()

    @on(Input.Submitted, "#own")
    def submit_own(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.dismiss(event.value.strip())

    @on(Button.Pressed, "#ok")
    def confirm(self) -> None:
        own = self.query_one("#own", Input).value.strip()
        if own:
            self.dismiss(own)
            return
        index = self.query_one("#ideas", ListView).index
        # The node expects a 1-based number, matching how it rendered the list.
        self.dismiss(str((index or 0) + 1))


class ConfirmScreen(ModalScreen[str]):
    """choose_build_dir: the directory is not empty, confirm before writing."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Directory is not empty", id="title")
            yield Label(self.payload.get("warning", ""), classes="warn")
            yield Label(self.payload.get("instructions", ""), classes="hint")
            with Horizontal(id="buttons"):
                yield Button("Use it anyway", variant="error", id="yes")
                yield Button("Pick another", variant="primary", id="no")

    @on(Button.Pressed, "#yes")
    def yes(self) -> None:
        self.dismiss("yes")

    @on(Button.Pressed, "#no")
    def no(self) -> None:
        self.dismiss("no")


class PathScreen(ModalScreen[str]):
    """choose_build_dir: absolute path to build in, re-shown on a bad answer."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Where should this be built?", id="title")
            if summary := self.payload.get("project_summary"):
                yield Label(summary, classes="hint")
            if stack := self.payload.get("tech_stack"):
                yield Label(f"Stack: {stack}", classes="hint")
            for task in self.payload.get("tasks", []):
                yield Label(f"  {task}", classes="hint")
            if error := self.payload.get("error"):
                yield Label(error, classes="warn")
            yield Label(self.payload.get("instructions", ""), classes="hint")
            yield Input(placeholder="/absolute/path/to/build/in", id="path")
            yield Button("Build here", variant="success", id="ok")

    def on_mount(self) -> None:
        self.query_one("#path", Input).focus()

    @on(Input.Submitted, "#path")
    def submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    @on(Button.Pressed, "#ok")
    def confirm(self) -> None:
        self.dismiss(self.query_one("#path", Input).value)


def prompt_for(payload: dict) -> ModalScreen:
    """Pick the dialog for an interrupt payload.

    Dispatch is on payload shape, not node name, because choose_build_dir emits
    two different payloads (path prompt and non-empty confirmation) and can emit
    either of them repeatedly within a single node run.
    """
    if not isinstance(payload, dict):
        return PathScreen({"instructions": str(payload)})
    if "draft_bias" in payload:
        return BiasScreen(payload)
    if "ideas" in payload:
        return IdeaScreen(payload)
    if "warning" in payload:
        return ConfirmScreen(payload)
    return PathScreen(payload)


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------


class PhasePanel(Static):
    """Sidebar showing pipeline progress."""

    def __init__(self) -> None:
        super().__init__(id="phases")
        self.done: set[str] = set()
        self.counts: dict[str, int] = {}

    def mark(self, node: str) -> None:
        self.done.add(node)
        if node in FAN_OUT:
            self.counts[node] = self.counts.get(node, 0) + 1
        self.refresh_panel()

    def restore(self, done: set[str], counts: dict[str, int]) -> None:
        """Seed progress from a checkpoint when resuming an existing run."""
        self.done |= done
        self.counts.update({k: v for k, v in counts.items() if v})
        self.refresh_panel()

    def refresh_panel(self) -> None:
        running_marked = False
        lines = []
        for node, label in PHASES:
            if node in self.done:
                count = self.counts.get(node)
                suffix = f" ({count})" if count else ""
                lines.append(f"[green]✓[/] {label}{suffix}")
            elif not running_marked:
                lines.append(f"[yellow]▶[/] {label}")
                running_marked = True
            else:
                lines.append(f"[dim]·[/] {label}")
        self.update("\n".join(lines))


class HackathonApp(App):
    """Runs the pipeline start to finish."""

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #phases {
        width: 28;
        padding: 1 2;
        border-right: solid $panel;
    }
    #output { width: 1fr; padding: 0 1; }
    #status { height: 1; background: $panel; color: $text; padding: 0 1; }
    #dialog {
        width: 80%;
        max-height: 80%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }
    #title { text-style: bold; }
    .hint { color: $text-muted; }
    .warn { color: $warning; text-style: bold; }
    #bias { height: 20; }
    #ideas { height: 15; }
    #buttons { height: auto; }
    """

    BINDINGS = [
        ("s", "skip_step", "Skip this step"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, url: str | None, thread_id: str) -> None:
        super().__init__()
        self.url = url
        self.thread_id = thread_id
        self.started = time.monotonic()
        self.messages: queue.SimpleQueue = queue.SimpleQueue()

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            yield PhasePanel()
            yield RichLog(id="output", wrap=True, markup=False)
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        # SimpleQueue.put is thread-safe; LangGraph runs sync nodes in a thread
        # pool, so _say is not guaranteed to be on the event loop.
        nodes.set_say_sink(self.messages.put)
        self.set_interval(0.1, self.drain)
        self.set_interval(1.0, self.tick)
        self.tick()
        self.run_pipeline()

    def on_unmount(self) -> None:
        nodes.set_say_sink(None)

    def drain(self) -> None:
        log = self.query_one("#output", RichLog)
        while True:
            try:
                log.write(self.messages.get_nowait())
            except queue.Empty:
                return

    def tick(self) -> None:
        elapsed = int(time.monotonic() - self.started)
        if not self.thread_id:
            self.query_one("#status", Static).update("choosing a previous run...")
            return
        where = self.url or f"thread {self.thread_id}"
        self.query_one("#status", Static).update(
            f"{where}  |  {elapsed // 60}m{elapsed % 60:02d}s  |  thread {self.thread_id}"
        )

    def log_line(self, text: str) -> None:
        # Goes through the same queue as _say output rather than writing to the
        # widget directly. Two write paths meant build output (queued, drained
        # on a timer) could appear after the phase header or "Done" line that
        # logically preceded it.
        self.messages.put(text)

    def action_skip_step(self) -> None:
        """Abandon the research currently in flight and move to the next phase.

        Only the fan-out research phases can be skipped: they have per-item
        workers that degrade individually. Everything else has a single result
        the pipeline genuinely needs.
        """
        nodes.request_skip()
        self.log_line("  - skip requested; abandoning in-flight research")

    def record_update(self, event: dict) -> None:
        panel = self.query_one(PhasePanel)
        for node in event:
            # Skip LangGraph's own bookkeeping keys (__metadata__, __end__, ...);
            # they are not nodes and would show up as phantom phases.
            if node.startswith("__"):
                continue
            if node not in FAN_OUT:
                # The phase moved on, so re-arm skipping for the next one.
                nodes.clear_skip()
            panel.mark(node)
            label = dict(PHASES).get(node, node)
            self.log_line(f"=== {label} ===")

    @work(exclusive=True, exit_on_error=False)
    async def run_pipeline(self) -> None:
        async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
            graph = build_graph(saver)

            try:
                # `ls`: pick a previous run first. Inside the try because the
                # worker runs with exit_on_error=False, so anything raised out
                # here would vanish with no UI and no message at all.
                if not self.thread_id:
                    rows = await list_threads(saver, graph)
                    if not rows:
                        self.log_line(f"No previous runs in {DB_PATH}.")
                        self.log_line(
                            "Start one with: python -m agent.tui <devpost-url>"
                        )
                        self.log_line("Press q to quit.")
                        return
                    chosen = await self.push_screen_wait(ThreadPicker(rows))
                    if not chosen:
                        self.exit()
                        return
                    self.thread_id = chosen

                config = {"configurable": {"thread_id": self.thread_id}}
                payload = await self.first_payload(graph, config)
                while True:
                    pending = None
                    async for event in graph.astream(
                        payload, config, stream_mode="updates"
                    ):
                        if "__interrupt__" in event:
                            pending = event["__interrupt__"][0].value
                        else:
                            self.record_update(event)
                    # Decided on the stream, never on state.next: a second
                    # interrupt inside one node leaves state.next empty while
                    # the graph is still waiting.
                    if pending is None:
                        break
                    answer = await self.push_screen_wait(prompt_for(pending))
                    payload = Command(resume=answer)
            except LookupError:
                # Unknown thread id; first_payload already explained it, and
                # suggesting "resume with the same bad id" would be nonsense.
                self.log_line("Press q to quit.")
                return
            except Exception as exc:
                # The run is long and the backend is flaky, so a failure must
                # not tear down the UI: the checkpoint is still resumable.
                self.log_line("")
                self.log_line(f"Pipeline failed: {type(exc).__name__}: {exc}")
                self.log_line(
                    f"Resume with: python -m agent.tui --thread {self.thread_id}"
                )
                self.log_line("Press q to quit.")
                return

            state = await graph.aget_state(config)
            self.log_line("")
            self.log_line(state.values.get("build_result", "Pipeline finished."))
            self.log_line("")
            self.log_line("Done. Press q to quit.")

    async def first_payload(self, graph, config):
        """New run starts from the URL; a resumed run picks up where it paused."""
        if self.url:
            return {"devpost_url": self.url}

        try:
            state = await graph.aget_state(config)
        except Exception as exc:
            self.log_line(f"Cannot read thread {self.thread_id}: {exc}")
            self.log_line(
                "This run was checkpointed by an older version of the graph and "
                "cannot be resumed. Start a new run instead."
            )
            raise LookupError(f"unreadable thread {self.thread_id}") from exc

        if not state.values:
            self.log_line(f"No checkpoint found for thread {self.thread_id}.")
            self.log_line("Check the id, or start a new run with a DevPost URL.")
            raise LookupError(f"unknown thread {self.thread_id}")

        # Rebuild the visible progress, otherwise a resumed run shows an empty
        # sidebar and empty log and looks like it silently restarted.
        self.url = state.values.get("devpost_url") or self.url
        done, counts = completed_phases(state.values)
        self.query_one(PhasePanel).restore(done, counts)
        self.log_line(f"=== Resumed thread {self.thread_id} ===")
        for line in resume_summary(state.values):
            self.log_line(line)
        pending = ", ".join(sorted(set(state.next))) or "none"
        self.log_line(f"  picking up at: {pending}")
        self.log_line("")

        if state.interrupts:
            answer = await self.push_screen_wait(prompt_for(state.interrupts[0].value))
            return Command(resume=answer)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hackathon pipeline TUI",
        epilog=(
            "examples:\n"
            "  python -m agent.tui https://foo.devpost.com   start a new run\n"
            "  python -m agent.tui ls                        pick a previous run\n"
            "  python -m agent.tui --thread <id>             resume a known id"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="DevPost URL to start a new run, or 'ls' to browse previous runs",
    )
    parser.add_argument("--thread", help="resume an existing run by thread id")
    args = parser.parse_args()

    browsing = args.target == "ls"
    url = None if browsing else args.target
    if not url and not args.thread and not browsing:
        parser.error(
            "give a DevPost URL, 'ls' to browse previous runs, or --thread <id>"
        )

    if browsing:
        thread_id = ""  # chosen in the picker
    else:
        thread_id = args.thread or str(uuid.uuid4())

    HackathonApp(url=url, thread_id=thread_id).run()


if __name__ == "__main__":
    main()
