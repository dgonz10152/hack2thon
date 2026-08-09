"""The pipeline's node functions, one module per phase.

judges.py — read the page, research the panel, compile the bias analysis
ideas.py  — brainstorm, rank, research, and flesh out app ideas
build.py  — plan the build and drive the opencode coding sub-agents
common.py — the shared research-worker policy (timeout, skip, degrade)
"""

from agent.nodes.build import (
    build_app_node,
    choose_build_dir_node,
    plan_build_node,
)
from agent.nodes.ideas import (
    compile_ideas_node,
    generate_idea_candidates_node,
    rank_ideas_node,
    research_ideas_orchestrator,
    research_one_idea_node,
    select_idea_node,
)
from agent.nodes.judges import (
    compile_bias_node,
    extract_judges_node,
    extract_theme_node,
    fetch_html_node,
    research_judges_orchestrator,
    research_one_judge_node,
    review_bias_node,
)

__all__ = [
    "build_app_node",
    "choose_build_dir_node",
    "compile_bias_node",
    "compile_ideas_node",
    "extract_judges_node",
    "extract_theme_node",
    "fetch_html_node",
    "generate_idea_candidates_node",
    "plan_build_node",
    "rank_ideas_node",
    "research_ideas_orchestrator",
    "research_judges_orchestrator",
    "research_one_idea_node",
    "research_one_judge_node",
    "review_bias_node",
    "select_idea_node",
]
