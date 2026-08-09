"""Reading past runs out of the checkpoint database.

The graph only reports events for the current process, so everything shown
about a *previous* run - the sidebar on resume, the `ls` listing - has to be
inferred from checkpointed state values. That inference lives here.
"""

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
        "extract_theme": bool(values.get("hackathon_synopsis")),
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
