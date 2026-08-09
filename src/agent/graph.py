from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from agent.nodes import (
    build_app_node,
    choose_build_dir_node,
    compile_bias_node,
    compile_ideas_node,
    extract_judges_node,
    extract_theme_node,
    fetch_html_node,
    generate_idea_candidates_node,
    plan_build_node,
    rank_ideas_node,
    research_ideas_orchestrator,
    research_judges_orchestrator,
    research_one_idea_node,
    research_one_judge_node,
    review_bias_node,
    select_idea_node,
)
from agent.state import AgentState, InputState


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """Wire the graph. Pass a checkpointer to run it outside `langgraph dev`,
    which supplies its own; without one, `interrupt()` cannot resume."""
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
    g.add_node("plan_build", plan_build_node)
    g.add_node("choose_build_dir", choose_build_dir_node)
    g.add_node("build_app", build_app_node)

    g.add_edge(START, "fetch_html")
    # extract_theme runs before extract_judges rather than beside it. Two
    # branches into compile_bias is NOT a barrier join: they finish at different
    # depths, so compile_bias fired once per branch, writing judge_bias
    # concurrently (InvalidUpdateError) and running once on judges that had not
    # been researched yet. One incoming edge means it fires exactly once, after
    # every Send worker. It also guarantees the orchestrator below has the
    # synopsis it hands to each worker.
    g.add_edge("fetch_html", "extract_theme")
    g.add_edge("extract_theme", "extract_judges")
    g.add_conditional_edges(
        "extract_judges", research_judges_orchestrator, ["research_one_judge"]
    )
    g.add_edge("research_one_judge", "compile_bias")
    g.add_edge("compile_bias", "review_bias")
    g.add_edge("review_bias", "generate_idea_candidates")
    g.add_edge("generate_idea_candidates", "rank_ideas")
    g.add_conditional_edges(
        "rank_ideas", research_ideas_orchestrator, ["research_one_idea"]
    )
    g.add_edge("research_one_idea", "compile_ideas")
    g.add_edge("compile_ideas", "select_idea")
    g.add_edge("select_idea", "plan_build")
    g.add_edge("plan_build", "choose_build_dir")
    g.add_edge("choose_build_dir", "build_app")
    g.add_edge("build_app", END)
    return g.compile(checkpointer=checkpointer)


graph = build_graph()
