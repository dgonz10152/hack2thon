"""The Textual app that drives the whole pipeline, from DevPost URL to built app.

Runs the graph in-process with a SQLite checkpointer, so a run survives quitting
or crashing. The three interrupt nodes surface as the dialogs in `dialogs.py`.
"""

from __future__ import annotations

import argparse
import queue
import time
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, RichLog, Static

from agent import progress
from agent.graph import build_graph
from agent.tui.dialogs import ThreadPicker, prompt_for
from agent.tui.history import (
    FAN_OUT,
    PHASES,
    completed_phases,
    list_threads,
    resume_summary,
)

DB_PATH = ".runs.db"


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
        progress.set_say_sink(self.messages.put)
        self.set_interval(0.1, self.drain)
        self.set_interval(1.0, self.tick)
        self.tick()
        self.run_pipeline()

    def on_unmount(self) -> None:
        progress.set_say_sink(None)

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
        progress.request_skip()
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
                progress.clear_skip()
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
