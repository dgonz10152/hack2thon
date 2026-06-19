# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

Multi-agent system built on LangGraph that analyzes a DevPost hackathon page end-to-end: fetches the page, extracts theme + judges, fans out a research agent per judge (DuckDuckGo via `search_web`), and synthesizes a panel-wide bias analysis. From the approved bias it then brainstorms a pool of app ideas, ranks the top 20, fans out a research agent per idea (feasibility, similar projects, past-winner "winning formula"), compiles 20 fully fleshed-out ideas, and interrupts for the user to pick one or write their own.

## Layout

All source lives under `src/agent/`:
- `graph.py` — `build_graph()` wires the LangGraph `StateGraph`. Entry point is the `graph` module-level export.
- `nodes.py` — node functions. The judge-research and idea-research steps are async and each gated by a module-level `asyncio.Semaphore`.
- `state.py` — `AgentState`, `Judge`/`Judges`/`JudgeResearchState`, and `Idea`/`IdeaCandidates`/`FinalIdea`/`FinalIdeas`/`IdeaResearchState`. The `judges` field uses a `merge_judges` reducer (upsert by name); the `ideas` field uses a `merge_ideas` reducer (upsert by title). Both are required for their fan-outs.
- `tools.py` — `@tool`-decorated functions (`fetch_html`, `search_web`).
- `prompts.py` — system prompts as module constants.
- `model.py` — `ChatOllama` builder; reads `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`.

## Architecture pattern

Orchestrator–worker via `langgraph.types.Send`, used twice (judge research and idea research):
```
fetch_html ─┬─► extract_theme ──────────────────────────────────────┐
            └─► extract_judges ─► research_one_judge (×N) ─► compile_bias ─► review_bias ─┐
                                                                                          ▼
  select_idea ◄─ compile_ideas ◄─ research_one_idea (×20) ◄─ rank_ideas ◄─ generate_idea_candidates
       │
       ▼  END
```
`research_judges_orchestrator` and `research_ideas_orchestrator` are conditional-edge dispatchers (not nodes) that return one `Send("research_one_judge"/"research_one_idea", ...)` per judge/idea. `review_bias` and `select_idea` are `interrupt()` (human-in-the-loop) nodes.

Idea flow: `generate_idea_candidates` brainstorms ~40 ideas from the approved bias + synopsis, `rank_ideas` selects the top 20 (one LLM call, no web search) and seeds them into the `ideas` fan-out field, each `research_one_idea` worker enriches one idea with web research, `compile_ideas` turns the 20 into fully fleshed-out `FinalIdea`s, and `select_idea` interrupts for the user to pick one or supply their own.

## Conventions

- **Reducers required for fan-out fields.** Any state field updated by parallel `Send` branches needs an `Annotated[..., reducer]` declaration (see `judges`/`ideas` in `state.py`). Forgetting this raises `InvalidUpdateError`.
- **Async + semaphore for parallel LLM calls.** `research_one_judge_node` and `research_one_idea_node` are `async def` and use `async with _JUDGE_RESEARCH_SEMAPHORE` / `_IDEA_RESEARCH_SEMAPHORE` before `ainvoke`. Limits read `JUDGE_RESEARCH_CONCURRENCY` / `IDEA_RESEARCH_CONCURRENCY` (default 2 each). Lower if Ollama returns `'too many concurrent requests'`.
- **Tools are `@tool`-decorated** and invoked with `.invoke({"arg": value})` from nodes, or bound into a `create_react_agent` for agent loops.
- **Structured output** uses `model.with_structured_output(PydanticModel)` — see `judge_extractor`/`idea_generator`/`idea_compiler` in `nodes.py`. The `Judges`, `IdeaCandidates`, and `FinalIdeas` models each have a `@model_validator(mode="before")` that unwraps bare lists, since some Ollama models return `[...]` instead of `{"judges"/"ideas": [...]}`.
- **Prompts** live as `_SYSTEM` constants in `prompts.py` and are passed in as `SystemMessage(content=...)`.

## Commands

```bash
uv sync                    # install / update deps
uv run langgraph dev       # local LangGraph dev server
uv run python -c "from agent.graph import graph; print(list(graph.get_graph().nodes.keys()))"  # quick smoke test
```

## Environment

`.env.example` documents the required vars. Important ones:
- `OLLAMA_MODEL` — e.g. `gpt-oss:120b-cloud`
- `JUDGE_RESEARCH_CONCURRENCY` — throttle for parallel judge research
- `IDEA_RESEARCH_CONCURRENCY` — throttle for parallel idea research (up to 20 ideas)
- `LANGSMITH_*` — tracing

## Gotchas

- The `Judge.online_summary` and `Idea.research` fields default to `""`. Their reducers treat empty strings as "no update" so re-running enrichment is idempotent. `rank_ideas` seeds `ideas` with `research=""`, then each worker upserts the same title with a populated `research` — so `rank_ideas` must return idea titles **verbatim** (its prompt says so) or `merge_ideas` won't match the seed.
- `compile_bias` joins both the judge-research branch and `extract_theme`. Both must complete before it fires — that's intentional (it needs the synopsis).
- Two `interrupt()` nodes exist: `review_bias` then `select_idea`. Resuming a thread must target the right one (the bias review fires first).
- `ddgs` is listed as a separate dep alongside `duckduckgo-search`; `langchain-community`'s `DuckDuckGoSearchRun` may pick up either depending on version.
