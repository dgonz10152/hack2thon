"""The modal dialogs, one per human-in-the-loop moment.

`prompt_for` dispatches an interrupt payload to its dialog by payload *shape*,
not node name, because choose_build_dir emits two different payloads (path
prompt and non-empty confirmation) and can emit either repeatedly within one
node run.
"""

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, TextArea


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


def prompt_for(payload: dict) -> ModalScreen:
    """Pick the dialog for an interrupt payload, by its shape."""
    if not isinstance(payload, dict):
        return PathScreen({"instructions": str(payload)})
    if "draft_bias" in payload:
        return BiasScreen(payload)
    if "ideas" in payload:
        return IdeaScreen(payload)
    if "warning" in payload:
        return ConfirmScreen(payload)
    return PathScreen(payload)
