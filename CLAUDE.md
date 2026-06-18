# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

Multi-agent system built on LangGraph that analyzes a DevPost hackathon page end-to-end: fetches the page, extracts theme + judges, fans out a research agent per judge (DuckDuckGo via `search_web`), then synthesizes a panel-wide bias analysis.

## Layout

All source lives under `src/agent/`:
- `graph.py` — `build_graph()` wires the LangGraph `StateGraph`. Entry point is the `graph` module-level export.
- `nodes.py` — node functions. The judge-research step is async and gated by a module-level `asyncio.Semaphore`.
- `state.py` — `AgentState`, `Judge`, `Judges`, `JudgeResearchState`. `judges` field uses a `merge_judges` reducer to upsert by name (required for the fan-out).
- `tools.py` — `@tool`-decorated functions (`fetch_html`, `search_web`).
- `prompts.py` — system prompts as module constants.
- `model.py` — `ChatOllama` builder; reads `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`.

## Architecture pattern

Orchestrator–worker via `langgraph.types.Send`:
```
fetch_html ─┬─► extract_theme ─────────────────────┐
            └─► extract_judges ─► research_one_judge (×N) ─► compile_bias ─► END
```
`research_judges_orchestrator` is a conditional-edge dispatcher (not a node) that returns one `Send("research_one_judge", ...)` per judge.

## Conventions

- **Reducers required for fan-out fields.** Any state field updated by parallel `Send` branches needs an `Annotated[..., reducer]` declaration (see `judges` in `state.py`). Forgetting this raises `InvalidUpdateError`.
- **Async + semaphore for parallel LLM calls.** `research_one_judge_node` is `async def` and uses `async with _JUDGE_RESEARCH_SEMAPHORE` before `ainvoke`. Concurrency limit reads `JUDGE_RESEARCH_CONCURRENCY` env var (default 2). Lower if Ollama returns `'too many concurrent requests'`.
- **Tools are `@tool`-decorated** and invoked with `.invoke({"arg": value})` from nodes, or bound into a `create_react_agent` for agent loops.
- **Structured output** uses `model.with_structured_output(PydanticModel)` — see `judge_extractor` in `nodes.py`. The `Judges` model has a `@model_validator(mode="before")` that unwraps bare lists, since some Ollama models return `[...]` instead of `{"judges": [...]}`.
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
- `LANGSMITH_*` — tracing

## Gotchas

- The `Judge.online_summary` field defaults to `""`. The reducer treats empty strings as "no update" so re-running enrichment is idempotent.
- `compile_bias` joins both the judge-research branch and `extract_theme`. Both must complete before it fires — that's intentional (it needs the synopsis).
- `ddgs` is listed as a separate dep alongside `duckduckgo-search`; `langchain-community`'s `DuckDuckGoSearchRun` may pick up either depending on version.
