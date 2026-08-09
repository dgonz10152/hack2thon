"""Phase 2: turn the approved bias analysis into researched app ideas.

generate_idea_candidates → rank_ideas (top 20) → research_one_idea (×20,
parallel) → compile_ideas → select_idea (human picks one or writes their own).
"""

import asyncio
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send, interrupt

from agent.model import model, resilient
from agent.nodes.common import run_researcher
from agent.prompts import (
    IDEA_COMPILER_SYSTEM,
    IDEA_GENERATOR_SYSTEM,
    IDEA_RANKER_SYSTEM,
    IDEA_RESEARCHER_SYSTEM,
)
from agent.state import AgentState, FinalIdeas, IdeaCandidates, IdeaResearchState
from agent.tools import search_web

idea_generator = resilient(model.with_structured_output(IdeaCandidates))
idea_ranker = resilient(model.with_structured_output(IdeaCandidates))
idea_compiler = resilient(model.with_structured_output(FinalIdeas))
idea_researcher = resilient(create_react_agent(model, tools=[search_web]))

_SEMAPHORE = asyncio.Semaphore(int(os.getenv("IDEA_RESEARCH_CONCURRENCY", "2")))


def generate_idea_candidates_node(state: AgentState) -> AgentState:
    user_content = (
        f"Hackathon synopsis:\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Judge bias analysis:\n{state.get('judge_bias', '')}\n\n"
        f"Generate the diverse pool of candidate app ideas."
    )
    result: IdeaCandidates = idea_generator.invoke(
        [
            SystemMessage(content=IDEA_GENERATOR_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"idea_candidates": result.ideas}


def rank_ideas_node(state: AgentState) -> AgentState:
    """Pick the best 20 candidates. Titles must come back verbatim: they seed
    the fan-out, and merge_ideas matches research onto them by title."""
    listing = "\n".join(
        f"- {i.title}: {i.pitch} (rationale: {i.rationale or 'n/a'})"
        for i in state.get("idea_candidates", [])
    )
    user_content = (
        f"Hackathon synopsis:\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Judge bias analysis:\n{state.get('judge_bias', '')}\n\n"
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
    """Fan out one research worker per ranked idea."""
    synopsis = state.get("hackathon_synopsis", "")
    bias = state.get("judge_bias", "")
    return [
        Send(
            "research_one_idea",
            {"idea": i, "hackathon_synopsis": synopsis, "judge_bias": bias},
        )
        for i in state["ideas"]
    ]


async def research_one_idea_node(state: IdeaResearchState) -> AgentState:
    idea = state["idea"]
    user_content = (
        f"Idea title: {idea.title}\n"
        f"Pitch: {idea.pitch}\n\n"
        f"Hackathon synopsis (for context):\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Judge bias analysis (for context):\n{state.get('judge_bias', '')}\n\n"
        f"Research this idea and return the Feasibility / Existing Projects / "
        f"Winning Formula summary."
    )
    summary = await run_researcher(
        idea_researcher,
        IDEA_RESEARCHER_SYSTEM,
        user_content,
        _SEMAPHORE,
        kind="Idea research",
        label=idea.title,
    )
    return {"ideas": [idea.model_copy(update={"research": summary})]}


def compile_ideas_node(state: AgentState) -> AgentState:
    """Fan-in: flesh out every researched idea into a full FinalIdea."""
    researched = "\n\n".join(
        f"--- {i.title} ---\nPitch: {i.pitch}\nResearch:\n{i.research or '(no research)'}"
        for i in state.get("ideas", [])
    )
    user_content = (
        f"Hackathon synopsis:\n{state.get('hackathon_synopsis', '')}\n\n"
        f"Judge bias analysis:\n{state.get('judge_bias', '')}\n\n"
        f"Researched ideas:\n{researched}"
    )
    result: FinalIdeas = idea_compiler.invoke(
        [
            SystemMessage(content=IDEA_COMPILER_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"final_ideas": result.ideas}


def select_idea_node(state: AgentState) -> AgentState:
    """Human in the loop: pick an idea by number, or describe your own."""
    options = "\n".join(
        f"{n}. {i.title} — {i.problem}"
        for n, i in enumerate(state.get("final_ideas", []), start=1)
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
