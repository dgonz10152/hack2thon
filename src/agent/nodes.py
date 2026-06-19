import asyncio
import os

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send, interrupt

from agent.model import model
from agent.prompts import (
    BIAS_COMPILER_SYSTEM,
    IDEA_COMPILER_SYSTEM,
    IDEA_GENERATOR_SYSTEM,
    IDEA_RANKER_SYSTEM,
    IDEA_RESEARCHER_SYSTEM,
    JUDGE_EXTRACTOR_SYSTEM,
    JUDGE_RESEARCHER_SYSTEM,
    THEME_EXTRACTOR_SYSTEM,
)
from agent.state import (
    AgentState,
    AgentStateUpdate,
    FinalIdeas,
    IdeaCandidates,
    IdeaResearchState,
    Judge,
    JudgeResearchState,
    Judges,
)
from agent.tools import fetch_html, search_web

judge_extractor = model.with_structured_output(Judges)
judge_researcher = create_react_agent(model, tools=[search_web])

idea_generator = model.with_structured_output(IdeaCandidates)
idea_ranker = model.with_structured_output(IdeaCandidates)
idea_compiler = model.with_structured_output(FinalIdeas)
deep_researcher = create_react_agent(model, tools=[search_web])

_JUDGE_RESEARCH_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("JUDGE_RESEARCH_CONCURRENCY", "2"))
)
_IDEA_RESEARCH_SEMAPHORE = asyncio.Semaphore(
    int(os.getenv("IDEA_RESEARCH_CONCURRENCY", "2"))
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


def generate_idea_candidates_node(state: AgentState) -> AgentStateUpdate:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge bias analysis:\n{bias}\n\n"
        f"Generate the diverse pool of candidate app ideas."
    )
    result: IdeaCandidates = idea_generator.invoke(
        [
            SystemMessage(content=IDEA_GENERATOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"idea_candidates": result.ideas}


def rank_ideas_node(state: AgentState) -> AgentStateUpdate:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    candidates = state.get("idea_candidates", [])
    listing = "\n".join(
        f"- {i.title}: {i.pitch} (rationale: {i.rationale or 'n/a'})"
        for i in candidates
    )
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge bias analysis:\n{bias}\n\n"
        f"Candidate ideas:\n{listing}\n\n"
        f"Select the best 20, returning them unchanged."
    )
    result: IdeaCandidates = idea_ranker.invoke(
        [
            SystemMessage(content=IDEA_RANKER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"ideas": result.ideas}


def research_ideas_orchestrator(state: AgentState) -> list[Send]:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    return [
        Send(
            "research_one_idea",
            {"idea": i, "hackathon_synposis": synopsis, "judge_bias": bias},
        )
        for i in state["ideas"]
    ]


async def research_one_idea_node(state: IdeaResearchState) -> AgentStateUpdate:
    idea = state["idea"]
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    user_content = (
        f"Idea title: {idea.title}\n"
        f"Pitch: {idea.pitch}\n\n"
        f"Hackathon synopsis (for context):\n{synopsis}\n\n"
        f"Judge bias analysis (for context):\n{bias}\n\n"
        f"Research this idea and return the Feasibility / Existing Projects / "
        f"Winning Formula summary."
    )
    async with _IDEA_RESEARCH_SEMAPHORE:
        result = await deep_researcher.ainvoke(
            {
                "messages": [
                    SystemMessage(content=IDEA_RESEARCHER_SYSTEM),
                    HumanMessage(content=user_content),
                ]
            }
        )
    summary = result["messages"][-1].content
    enriched = idea.model_copy(update={"research": summary})
    return {"ideas": [enriched]}


def compile_ideas_node(state: AgentState) -> AgentStateUpdate:
    synopsis = state.get("hackathon_synposis", "")
    bias = state.get("judge_bias", "")
    ideas = state.get("ideas", [])
    researched = "\n\n".join(
        f"--- {i.title} ---\nPitch: {i.pitch}\nResearch:\n{i.research or '(no research)'}"
        for i in ideas
    )
    user_content = (
        f"Hackathon synopsis:\n{synopsis}\n\n"
        f"Judge bias analysis:\n{bias}\n\n"
        f"Researched ideas:\n{researched}"
    )
    result: FinalIdeas = idea_compiler.invoke(
        [
            SystemMessage(content=IDEA_COMPILER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"final_ideas": result.ideas}


def select_idea_node(state: AgentState) -> AgentStateUpdate:
    ideas = state.get("final_ideas", [])
    options = "\n".join(
        f"{n}. {i.title} — {i.problem}" for n, i in enumerate(ideas, start=1)
    )
    selected = interrupt(
        {
            "ideas": options,
            "instructions": (
                "Pick an idea by its number, or write your own idea description "
                "to use instead."
            ),
        }
    )
    return {"selected_idea": selected}
