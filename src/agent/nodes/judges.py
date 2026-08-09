"""Phase 1: read the DevPost page and figure out who is judging, and how.

fetch_html → extract_theme → extract_judges → research_one_judge (×N, parallel)
→ compile_bias → review_bias (human approval).
"""

import asyncio
import os

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send, interrupt

from agent.model import model, resilient
from agent.nodes.common import run_researcher
from agent.prompts import (
    BIAS_COMPILER_SYSTEM,
    JUDGE_EXTRACTOR_SYSTEM,
    JUDGE_RESEARCHER_SYSTEM,
    THEME_EXTRACTOR_SYSTEM,
)
from agent.state import AgentState, JudgeResearchState, Judges
from agent.tools import fetch_html, search_web

chat = resilient(model)  # for the plain-text calls (theme, bias)
judge_extractor = resilient(model.with_structured_output(Judges))
judge_researcher = resilient(create_react_agent(model, tools=[search_web]))

_SEMAPHORE = asyncio.Semaphore(int(os.getenv("JUDGE_RESEARCH_CONCURRENCY", "2")))


def fetch_html_node(state: AgentState) -> AgentState:
    html = fetch_html.invoke({"url": state["devpost_url"]})
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return {"html": text}


def extract_theme_node(state: AgentState) -> AgentState:
    system = SystemMessage(content=THEME_EXTRACTOR_SYSTEM)
    message = HumanMessage(content=state["html"])
    result = chat.invoke([system, message])
    return {"hackathon_synopsis": result.content}


def extract_judges_node(state: AgentState) -> AgentState:
    system = SystemMessage(content=JUDGE_EXTRACTOR_SYSTEM)
    user = HumanMessage(content=state["html"])
    result: Judges = judge_extractor.invoke([system, user])
    return {"judges": result.judges, "messages": [system, user]}


def research_judges_orchestrator(state: AgentState) -> list[Send]:
    """Fan out one research worker per judge."""
    synopsis = state.get("hackathon_synopsis", "")
    return [
        Send("research_one_judge", {"judge": j, "hackathon_synopsis": synopsis})
        for j in state["judges"]
    ]


async def research_one_judge_node(state: JudgeResearchState) -> AgentState:
    judge = state["judge"]
    user_content = (
        f"Judge name: {judge.name}\n"
        f"Page blurb: {judge.blurb or '(none)'}\n\n"
        f"Hackathon synopsis (for context):\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Research this judge and return the 4-8 sentence summary."
    )
    summary = await run_researcher(
        judge_researcher,
        JUDGE_RESEARCHER_SYSTEM,
        user_content,
        _SEMAPHORE,
        kind="Judge research",
        label=judge.name,
    )
    return {"judges": [judge.model_copy(update={"online_summary": summary})]}


def compile_bias_node(state: AgentState) -> AgentState:
    """Fan-in: one panel-wide bias analysis from all the judge profiles."""
    profiles = "\n\n".join(
        f"--- {j.name} ---\n"
        f"Page blurb: {j.blurb or '(none)'}\n"
        f"Research summary: {j.online_summary or '(no research)'}"
        for j in state.get("judges", [])
    )
    user_content = (
        f"Hackathon synopsis:\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Judge profiles:\n{profiles}"
    )
    result = chat.invoke(
        [
            SystemMessage(content=BIAS_COMPILER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"judge_bias": result.content}


def review_bias_node(state: AgentState) -> AgentState:
    """Human in the loop: edit the analysis or approve it unchanged."""
    approved = interrupt(
        {
            "draft_bias": state.get("judge_bias", ""),
            "instructions": "Edit the analysis or return it unchanged to approve.",
        }
    )
    return {"judge_bias": approved}
