import asyncio
import os

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send, interrupt

from agent.model import model
from agent.prompts import (
    BIAS_COMPILER_SYSTEM,
    JUDGE_EXTRACTOR_SYSTEM,
    JUDGE_RESEARCHER_SYSTEM,
    THEME_EXTRACTOR_SYSTEM,
)
from agent.state import (
    AgentState,
    AgentStateUpdate,
    Judge,
    JudgeResearchState,
    Judges,
)
from agent.tools import fetch_html, search_web

judge_extractor = model.with_structured_output(Judges)
judge_researcher = create_react_agent(model, tools=[search_web])

_JUDGE_RESEARCH_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("JUDGE_RESEARCH_CONCURRENCY", "2"))
)


def fetch_html_node(state: AgentState) -> AgentStateUpdate:
    html = fetch_html.invoke({"url": state["devpost_url"]})
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return {"html": text}


def extract_theme_node(state: AgentState) -> AgentStateUpdate:
    system = SystemMessage(content=THEME_EXTRACTOR_SYSTEM)
    message = HumanMessage(content=state["html"])
    result = model.invoke([system, message])
    return {"hackathon_synposis": result.content}


def extract_judges_node(state: AgentState) -> AgentStateUpdate:
    system = SystemMessage(content=JUDGE_EXTRACTOR_SYSTEM)
    user = HumanMessage(content=state["html"])
    result: Judges = judge_extractor.invoke([system, user])
    return {"judges": result.judges, "messages": [system, user]}


def research_judges_orchestrator(state: AgentState) -> list[Send]:
    synopsis = state.get("hackathon_synposis", "")
    return [
        Send("research_one_judge", {"judge": j, "hackathon_synposis": synopsis})
        for j in state["judges"]
    ]


async def research_one_judge_node(state: JudgeResearchState) -> AgentStateUpdate:
    judge = state["judge"]
    synopsis = state.get("hackathon_synposis", "")
    user_content = (
        f"Judge name: {judge.name}\n"
        f"Page blurb: {judge.blurb or '(none)'}\n\n"
        f"Hackathon synopsis (for context):\n{synopsis}\n\n"
        f"Research this judge and return the 4–8 sentence summary."
    )
    async with _JUDGE_RESEARCH_SEMAPHORE:
        result = await judge_researcher.ainvoke(
            {
                "messages": [
                    SystemMessage(content=JUDGE_RESEARCHER_SYSTEM),
                    HumanMessage(content=user_content),
                ]
            }
        )
    summary = result["messages"][-1].content
    enriched = judge.model_copy(update={"online_summary": summary})
    return {"judges": [enriched]}


def compile_bias_node(state: AgentState) -> AgentStateUpdate:
    judges = state.get("judges", [])
    synopsis = state.get("hackathon_synposis", "")
    profiles = "\n\n".join(
        f"--- {j.name} ---\nPage blurb: {j.blurb or '(none)'}\nResearch summary: {j.online_summary or '(no research)'}"
        for j in judges
    )
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge profiles:\n{profiles}"
    )
    result = model.invoke(
        [
            SystemMessage(content=BIAS_COMPILER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"judge_bias": result.content}


def review_bias_node(state: AgentState) -> AgentStateUpdate:
    draft = state.get("judge_bias", "")
    approved = interrupt(
        {
            "draft_bias": draft,
            "instructions": "Edit the analysis or return it unchanged to approve.",
        }
    )
    return {"judge_bias": approved}
