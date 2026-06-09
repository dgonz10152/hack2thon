# multi-agent-research

A basic multi-agent system that uses AI agents to help you win hackathons end-to-end.

Point it at a DevPost URL and it scrapes the page, extracts the theme and judging panel, fans out a research agent per judge to gather public background, and then compiles a bias analysis you can use to tailor your submission.

## What it does

1. **Fetch** — pulls the DevPost page and strips it down to clean text.
2. **Extract theme** — distills requirements, theme, judging criteria, and stated tech biases.
3. **Extract judges** — pulls each judge's name and blurb from the page.
4. **Research judges (parallel)** — dispatches one ReAct agent per judge, each using a DuckDuckGo `search_web` tool to build a short factual profile.
5. **Compile bias** — synthesizes the judge profiles plus the hackathon synopsis into a panel-wide analysis covering tech preferences, domain leanings, presentation style, and hard signals to hit.

## Architecture

Built on [LangGraph](https://github.com/langchain-ai/langgraph) with an orchestrator–worker pattern. The judge-research step uses `Send` to fan out, gated by an `asyncio.Semaphore` so the Ollama backend doesn't get hammered with concurrent requests.

```
fetch_html ──┬─► extract_theme ─────────────────────┐
             └─► extract_judges ─► research_one_judge (×N) ─► compile_bias ─► END
```

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your Ollama + LangSmith config
```

Key env vars:
- `OLLAMA_MODEL` — model to use (cloud or local)
- `JUDGE_RESEARCH_CONCURRENCY` — max parallel judge-research agents (default 2; lower if Ollama returns "too many concurrent requests")

## Run

```bash
uv run langgraph dev
```

Then invoke the graph with a DevPost URL as `devpost_url`. Final state includes `hackathon_synposis`, enriched `judges` (each with `online_summary`), and `judge_bias`.
