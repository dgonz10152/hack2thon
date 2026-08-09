"""Textual TUI for the pipeline.

    uv run python -m agent.tui https://foo.devpost.com   # new run
    uv run python -m agent.tui ls                        # browse and resume past runs
    uv run python -m agent.tui --thread <id>             # resume a known thread id

app.py     — the app itself: layout, driver loop, resume, skip
dialogs.py — one modal dialog per human-in-the-loop moment
history.py — inferring past-run progress from checkpointed state
"""

from agent.tui.app import DB_PATH, HackathonApp, PhasePanel, main
from agent.tui.dialogs import (
    BiasScreen,
    ConfirmScreen,
    IdeaScreen,
    PathScreen,
    ThreadPicker,
    prompt_for,
)
from agent.tui.history import (
    FAN_OUT,
    PHASES,
    completed_phases,
    list_threads,
    resume_summary,
)

__all__ = [
    "DB_PATH",
    "FAN_OUT",
    "PHASES",
    "BiasScreen",
    "ConfirmScreen",
    "HackathonApp",
    "IdeaScreen",
    "PathScreen",
    "PhasePanel",
    "ThreadPicker",
    "completed_phases",
    "list_threads",
    "main",
    "prompt_for",
    "resume_summary",
]
