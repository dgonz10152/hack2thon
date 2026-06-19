from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    compile_bias_node,
    compile_ideas_node,
    extract_judges_node,
    extract_theme_node,
    fetch_html_node,
    generate_idea_candidates_node,
    rank_ideas_node,
    research_ideas_orchestrator,
    research_judges_orchestrator,
    research_one_idea_node,
    research_one_judge_node,
    review_bias_node,
    select_idea_node,
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
    g.add_node("generate_idea_candidates", generate_idea_candidates_node)
    g.add_node("rank_ideas", rank_ideas_node)
    g.add_node("research_one_idea", research_one_idea_node)
    g.add_node("compile_ideas", compile_ideas_node)
    g.add_node("select_idea", select_idea_node)

    g.add_edge(START, "fetch_html")
    g.add_edge("fetch_html", "extract_judges")
    g.add_edge("fetch_html", "extract_theme")
    g.add_conditional_edges(
        "extract_judges", research_judges_orchestrator, ["research_one_judge"]
    )
    g.add_edge("research_one_judge", "compile_bias")
    g.add_edge("extract_theme", "compile_bias")
    g.add_edge("compile_bias", "review_bias")
    g.add_edge("review_bias", "generate_idea_candidates")
    g.add_edge("generate_idea_candidates", "rank_ideas")
    g.add_conditional_edges(
        "rank_ideas", research_ideas_orchestrator, ["research_one_idea"]
    )
    g.add_edge("research_one_idea", "compile_ideas")
    g.add_edge("compile_ideas", "select_idea")
    g.add_edge("select_idea", END)
    return g.compile()


graph = build_graph()
