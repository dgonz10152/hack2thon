# multi-agent-research

A basic multi-agent system that uses AI agents to help you win hackathons end-to-end.

Point it at a DevPost URL and it scrapes the page, researches the judging panel, compiles a bias analysis, generates and researches app ideas against that bias, and then actually builds the idea you pick into a directory on your machine.

## What it does

1. **Fetch** - pulls the DevPost page and strips it down to clean text.
2. **Extract theme** - distills requirements, theme, judging criteria, and stated tech biases.
3. **Extract judges** - pulls each judge's name and blurb from the page.
4. **Research judges (parallel)** - dispatches one ReAct agent per judge, each using a DuckDuckGo `search_web` tool to build a short factual profile.
5. **Compile bias** - synthesizes the judge profiles plus the hackathon synopsis into a panel-wide analysis covering tech preferences, domain leanings, presentation style, and hard signals to hit.
6. **Review bias** (you) - the draft analysis pauses for you to edit or approve.
7. **Generate + rank ideas** - brainstorms a pool of app ideas from the approved bias and narrows it to the top 20.
8. **Research ideas (parallel)** - one agent per idea, covering feasibility, existing projects, and the past-winner "winning formula".
9. **Select idea** (you) - pick one of the 20 fleshed-out ideas by number, or write your own.
10. **Plan the build** - a master agent decomposes the chosen idea into an ordered list of atomic build tasks.
11. **Choose a directory** (you) - name an absolute path; it is created if missing and `git init`ed so you can diff and revert.
12. **Build it** - one `opencode` coding sub-agent per task, run in order in that directory, with a commit after each task.

## Architecture

Built on [LangGraph](https://github.com/langchain-ai/langgraph).
The research phases use an orchestrator-worker pattern via `Send` to fan out, each gated by an `asyncio.Semaphore` so the Ollama backend doesn't get hammered with concurrent requests.

```
fetch_html ─┬─► extract_theme ──────────────────────────────────────┐
            └─► extract_judges ─► research_one_judge (×N) ─► compile_bias ─► review_bias ─┐
                                                                                          ▼
  select_idea ◄─ compile_ideas ◄─ research_one_idea (×20) ◄─ rank_ideas ◄─ generate_idea_candidates
       │
       ▼
  plan_build ─► choose_build_dir ─► build_app ─► END
```

The build phase is deliberately **sequential**, unlike the two research fan-outs.
Task N+1 reads the files task N wrote, so parallel coding agents in one directory would clobber each other.

Coding sub-agents run through the [opencode](https://opencode.ai) CLI, configured in `opencode.json` to use the same Ollama credential as the rest of the pipeline.
A sub-agent that exits successfully without changing any files is treated as a failure, not a success, and retried once.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your Ollama + LangSmith config
```

The build phase also needs the [opencode](https://opencode.ai) CLI on your `PATH`.

Key env vars:
- `OLLAMA_MODEL` - model to use (cloud or local)
- `JUDGE_RESEARCH_CONCURRENCY` - max parallel judge-research agents (default 2; lower if Ollama returns "too many concurrent requests")
- `BUILD_MODEL` - `provider/model` for the coding sub-agents (default `ollama/minimax-m3`)
- `BUILD_TASK_TIMEOUT` - seconds a single coding sub-agent may run before it is killed (default 900)

## Run

```bash
uv run langgraph dev
```

Then invoke the graph with a DevPost URL as `devpost_url`.

The run pauses three times for you: to approve the bias analysis, to pick an idea, and to name a build directory.
Final state includes `hackathon_synposis`, enriched `judges`, `judge_bias`, the 20 `final_ideas`, the `build_plan`, and a `build_result` summarizing which tasks succeeded and where the app was written.

## Tests

```bash
uv run python tests/test_build_app.py   # build-phase self-check, no model calls or network
```
