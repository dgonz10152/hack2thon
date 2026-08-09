# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

Multi-agent system built on LangGraph that analyzes a DevPost hackathon page end-to-end: fetches the page, extracts theme + judges, fans out a research agent per judge (DuckDuckGo via `search_web`), and synthesizes a panel-wide bias analysis. From the approved bias it then brainstorms a pool of app ideas, ranks the top 20, fans out a research agent per idea (feasibility, similar projects, past-winner "winning formula"), compiles 20 fully fleshed-out ideas, and interrupts for the user to pick one or write their own. Finally it decomposes the chosen idea into an ordered build plan and drives one `opencode` coding sub-agent per task to actually build the app in a directory the user names.

## Layout

All source lives under `src/agent/`:
- `graph.py` — `build_graph(checkpointer=None)` wires the LangGraph `StateGraph`. Entry point is the `graph` module-level export (no checkpointer, since `langgraph dev` supplies its own).
- `tui.py` — Textual app driving the whole pipeline in-process. Modal dialogs for the three interrupts, `AsyncSqliteSaver` in `.runs.db` for resume.
- `nodes.py` — node functions. The judge-research and idea-research steps are async and each gated by a module-level `asyncio.Semaphore`.
- `state.py` — `AgentState`, `Judge`/`Judges`/`JudgeResearchState`, and `Idea`/`IdeaCandidates`/`FinalIdea`/`FinalIdeas`/`IdeaResearchState`. The `judges` field uses a `merge_judges` reducer (upsert by name); the `ideas` field uses a `merge_ideas` reducer (upsert by title). Both are required for their fan-outs.
- `tools.py` — `@tool`-decorated functions (`fetch_html`, `search_web`).
- `prompts.py` — system prompts as module constants.
- `model.py` — `ChatOllama` builder; reads `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`.

## Architecture pattern

Orchestrator–worker via `langgraph.types.Send`, used twice (judge research and idea research):
```
fetch_html ─► extract_theme ─► extract_judges ─► research_one_judge (×N) ─► compile_bias ─► review_bias ─┐
                                                                                          ▼
  select_idea ◄─ compile_ideas ◄─ research_one_idea (×20) ◄─ rank_ideas ◄─ generate_idea_candidates
       │
       ▼
  plan_build ─► choose_build_dir ─► build_app ─► END
```
`research_judges_orchestrator` and `research_ideas_orchestrator` are conditional-edge dispatchers (not nodes) that return one `Send("research_one_judge"/"research_one_idea", ...)` per judge/idea. `review_bias` and `select_idea` are `interrupt()` (human-in-the-loop) nodes.

Idea flow: `generate_idea_candidates` brainstorms ~40 ideas from the approved bias + synopsis, `rank_ideas` selects the top 20 (one LLM call, no web search) and seeds them into the `ideas` fan-out field, each `research_one_idea` worker enriches one idea with web research, `compile_ideas` turns the 20 into fully fleshed-out `FinalIdea`s, and `select_idea` interrupts for the user to pick one or supply their own.

Build flow (**sequential, not a fan-out**): `plan_build` is the master agent, turning the selected idea into a `BuildPlan` of ordered `BuildTask`s. `choose_build_dir` interrupts for an absolute path, validates it, and `git init`s it. `build_app` then runs one `opencode` subprocess sub-agent per task, in order, committing after each. It is sequential because task N+1 reads the files task N wrote, so parallel agents in one directory would clobber each other.

## Conventions

- **Reducers required for fan-out fields.** Any state field updated by parallel `Send` branches needs an `Annotated[..., reducer]` declaration (see `judges`/`ideas` in `state.py`). Forgetting this raises `InvalidUpdateError`.
- **Async + semaphore for parallel LLM calls.** `research_one_judge_node` and `research_one_idea_node` are `async def` and use `async with _JUDGE_RESEARCH_SEMAPHORE` / `_IDEA_RESEARCH_SEMAPHORE` before `ainvoke`. Limits read `JUDGE_RESEARCH_CONCURRENCY` / `IDEA_RESEARCH_CONCURRENCY` (default 2 each). Lower if Ollama returns `'too many concurrent requests'`.
- **Tools are `@tool`-decorated** and invoked with `.invoke({"arg": value})` from nodes, or bound into a `create_react_agent` for agent loops.
- **Structured output** uses `model.with_structured_output(PydanticModel)` — see `judge_extractor`/`idea_generator`/`idea_compiler` in `nodes.py`. The `Judges`, `IdeaCandidates`, and `FinalIdeas` models each have a `@model_validator(mode="before")` that unwraps bare lists, since some Ollama models return `[...]` instead of `{"judges"/"ideas": [...]}`.
- **Prompts** live as `_SYSTEM` constants in `prompts.py` and are passed in as `SystemMessage(content=...)`.

## Commands

```bash
uv sync                    # install / update deps
uv run python -m agent.tui <devpost-url>   # TUI, full pipeline in-process
uv run python -m agent.tui ls              # browse past runs and resume one
uv run python -m agent.tui --thread <id>   # resume a checkpointed run
uv run langgraph dev       # local LangGraph dev server
uv run python -c "from agent.graph import graph; print(list(graph.get_graph().nodes.keys()))"  # quick smoke test
uv run python tests/test_build_app.py       # build-phase self-check (hermetic, no model calls)
uv run python tests/test_tui_driver.py      # TUI driver loop + dialogs (headless)
uv run python tests/test_graph_topology.py  # fan-in nodes run exactly once
uv run python tests/test_tui_e2e.py         # full pipeline through the TUI, model calls stubbed
uv run python tests/test_tools.py            # search failures degrade, not crash
uv run python tests/test_resilience.py       # retries + degraded workers
```

## Environment

`.env.example` documents the required vars. Important ones:
- `OLLAMA_MODEL` — e.g. `gpt-oss:120b-cloud`
- `JUDGE_RESEARCH_CONCURRENCY` — throttle for parallel judge research
- `IDEA_RESEARCH_CONCURRENCY` — throttle for parallel idea research (up to 20 ideas)
- `BUILD_MODEL` — `provider/model` for the coding sub-agents, default `ollama/minimax-m3`
- `BUILD_TASK_TIMEOUT` — per-task wall-clock limit in seconds, default 900
- `MODEL_RETRIES` — attempts per model call on transient network errors, default 3
- `RESEARCH_RECURSION_LIMIT` — supersteps a research ReAct agent may take, default 40
- `RESEARCH_TIMEOUT` — seconds before a single research worker is abandoned, default 300
- `LANGSMITH_*` — tracing

## Gotchas

- The `Judge.online_summary` and `Idea.research` fields default to `""`. Their reducers treat empty strings as "no update" so re-running enrichment is idempotent. `rank_ideas` seeds `ideas` with `research=""`, then each worker upserts the same title with a populated `research` — so `rank_ideas` must return idea titles **verbatim** (its prompt says so) or `merge_ideas` won't match the seed.
- **Multiple incoming edges are NOT a barrier join.** `compile_bias` used to be fed by both the judge fan-out and `extract_theme`; because those branches finish at different depths, it fired once per branch — writing `judge_bias` concurrently (`InvalidUpdateError`, since it has no reducer) and running once on judges that had not been researched yet. The same wiring also left `research_judges_orchestrator` reading an empty `hackathon_synposis`, so every judge was researched with no hackathon context, silently. Fixed by serializing `fetch_html → extract_theme → extract_judges`, giving `compile_bias` exactly one incoming edge. `defer=True` does **not** fix this. `tests/test_graph_topology.py` pins it.
- Any node that fans in from a `Send` fan-out should have exactly one incoming edge, or its state writes need a reducer. `compile_ideas` is already correct (only `research_one_idea` feeds it).
- **Model calls retry transient network failures.** Ollama Cloud drops connections mid-response (`httpx.ReadError`, usually with an empty message), which killed runs at `rank_ideas` and elsewhere. `_resilient()` in `nodes.py` wraps each bound object with `.with_retry(retry_if_exception_type=_TRANSIENT)`. **Wrap the bound objects, not `model`:** `model.with_retry()` returns a `RunnableRetry`, which no longer has `with_structured_output`. The two direct call sites use `chat`, not `model` — patch `chat` when stubbing.
- **Fan-out workers degrade; single nodes do not.** A failed `research_one_judge`/`research_one_idea` writes a non-empty "unavailable" placeholder and logs via `_say`, so one blip cannot cost a 20-minute run. The placeholder must stay non-empty: the reducers treat `""` as "no update", making an empty one indistinguishable from research never running. Single nodes like `rank_ideas` still raise, because there is no peer to fall back on and inventing a ranking would be worse than stopping.
- **Retries and degradation only fire on errors, never on silence.** A hung model call stalls a phase forever, so research workers run through `_bounded_research()` with `RESEARCH_TIMEOUT` (default 300s) and a skip watcher. The TUI's `s` binding calls `nodes.request_skip()` to abandon in-flight workers; `record_update` clears the flag when a non-fan-out node arrives, so it does not leak into the next phase. Skipping is deliberately distinguishable from failure in the placeholder text.
- **The skip flag is a plain bool, not an `asyncio.Event`.** An Event created at import binds its internal futures to the first loop that waits on it, which misfires across loops. Also: decide "was this skipped?" from the flag, never from which task landed in `asyncio.wait`'s `done` set - a watcher that errors or is cancelled lands there too and would be misreported as a deliberate skip.
- **A tool that never raises can become a tool that never stops.** `search_web` returning a failure string made the ReAct agents retry different queries until they blew LangGraph's 25-superstep recursion limit (`GraphRecursionError`, one per idea) whenever DuckDuckGo rate-limited. The message now bounds retries explicitly ("at most once ... then stop searching"), and both research agents run with `RESEARCH_RECURSION_LIMIT` (default 40) rather than the default 25.
- **`search_web` never raises.** It is bound into both ReAct agents, so it runs once per judge and once per idea; a single transient DNS error or rate-limit used to abort a run that was already 20 minutes in. It now returns a `SEARCH FAILED: ...` string as the tool result, letting the agent retry with a different query or finish without it. `fetch_html` deliberately still raises: with no page there is nothing to analyse, so failing fast is correct.
- Three `interrupt()` nodes exist, in order: `review_bias`, `select_idea`, then `choose_build_dir`. Resuming a thread must target the right one. `choose_build_dir` can interrupt repeatedly within a single node run (re-prompting on a bad path, or to confirm a non-empty directory), so do not assume one interrupt per node.
- **Never decide "the run finished" from `state.next`.** At a node's *second* `interrupt()`, LangGraph reports `state.next == ()` even though the graph is still waiting — so a driver keyed on it declares success mid-run. `choose_build_dir` hits this on any bad path. Key on `__interrupt__` appearing in the `astream(stream_mode="updates")` output instead; `tests/test_tui_driver.py` pins this behaviour.
- The TUI dispatches interrupt dialogs on **payload shape, not node name**, because `choose_build_dir` emits two different payloads (path prompt vs. non-empty-directory confirmation).
- **On resume the sidebar must be seeded from the checkpoint.** It is otherwise driven only by events seen in the current process, so a resumed run showed an empty sidebar and empty log and looked like it had silently restarted. `completed_phases()` in `tui.py` infers finished phases from state values; `review_bias` is special-cased because it has no output of its own (it overwrites `judge_bias`), so it is inferred from `idea_candidates`.
- **A checkpoint written by an older graph can make `aget_state` raise.** Rebuilding a snapshot replays that step's pending writes, so threads from before the `compile_bias` fix still blow up with the original `InvalidUpdateError` and cannot be resumed. `list_threads` catches this per thread and marks the row `unreadable`, so one bad run does not break `ls`; `first_payload` reports it and refuses rather than showing a bogus resume hint.
- The TUI log has **one write path**: everything goes through `self.messages` and is drained on a timer. Writing to the `RichLog` directly meant `_say` output (queued) could render after the phase header or "Done" line that logically preceded it. A dialog can still appear before its queued lines are drained, which matters when testing.
- `nodes.set_say_sink(fn)` redirects build output; `None` restores `print`. The TUI routes it through a `queue.SimpleQueue` because LangGraph runs sync nodes in a thread pool, so `_say` is not guaranteed to be on the event loop.
- **A coding sub-agent can exit 0 having done nothing.** Observed with `ollama/minimax-m3`, which intermittently returns an empty response and writes no files. `build_app` therefore verifies each task against `git status --porcelain` rather than trusting the exit code, retries once, and fails the build if a task changes nothing twice. Do not "simplify" this to an exit-code check.
- `opencode.json` at the repo root carries both the `ollama` provider and the `permission` block that lets sub-agents edit and run commands unattended. It is passed to sub-agents via `OPENCODE_CONFIG` because they run with `--dir` pointing at the user's build directory, where this repo's project config would not be discovered.
- `ddgs` is listed as a separate dep alongside `duckduckgo-search`; `langchain-community`'s `DuckDuckGoSearchRun` may pick up either depending on version.
