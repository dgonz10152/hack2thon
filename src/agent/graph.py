from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    compile_bias_node,
    extract_judges_node,
    extract_theme_node,
    fetch_html_node,
    research_judges_orchestrator,
    research_one_judge_node,
    review_bias_node,
)
from agent.state import AgentState, InputState


def build_graph() -> CompiledStateGraph:
    g = StateGraph(AgentState, input=InputState)
    g.add_node("fetch_html", fetch_html_node)
    g.add_node("extract_judges", extract_judges_node)
    g.add_node("extract_theme", extract_theme_node)
    g.add_node("research_one_judge", research_one_judge_node)
    g.add_node("compile_bias", compile_bias_node)
    g.add_node("review_bias", review_bias_node)

    g.add_edge(START, "fetch_html")
    g.add_edge("fetch_html", "extract_judges")
    g.add_edge("fetch_html", "extract_theme")
    g.add_conditional_edges(
        "extract_judges", research_judges_orchestrator, ["research_one_judge"]
    )
    g.add_edge("research_one_judge", "compile_bias")
    g.add_edge("extract_theme", "compile_bias")
    g.add_edge("compile_bias", "review_bias")
    g.add_edge("review_bias", END)
    return g.compile(checkpointer=MemorySaver())


graph = build_graph()
