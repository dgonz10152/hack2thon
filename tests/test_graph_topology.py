"""Self-check for graph wiring: the fan-in nodes must run exactly once.

Hermetic - the real graph is built, but every node function is replaced with a
stub, so no model calls and no network.

Regression guard for a real failure: compile_bias originally had two incoming
edges (the judge fan-out and extract_theme). That is not a barrier join. The two
branches finish at different depths, so compile_bias ran once per branch, which
    (a) wrote judge_bias twice concurrently -> InvalidUpdateError, and
    (b) ran once on judges whose research had not happened yet.

Run:  uv run python tests/test_graph_topology.py
"""

import asyncio
import importlib

from langgraph.checkpoint.memory import MemorySaver

from agent import nodes
from agent.state import Idea, Judge

# agent/__init__.py re-exports the compiled graph, so `from agent import graph`
# would bind the graph object rather than the module we need to reload.
graph_module = importlib.import_module("agent.graph")


def _stub_nodes(calls: dict, judge_count: int, idea_count: int):
    """Replace every node function with a fast stub that records its calls."""

    def record(name, result=None):
        def fn(state):
            calls.setdefault(name, []).append(state)
            return result(state) if callable(result) else (result or {})

        return fn

    nodes.fetch_html_node = record("fetch_html", {"html": "page"})
    nodes.extract_theme_node = record("extract_theme", {"hackathon_synopsis": "SYN"})
    nodes.extract_judges_node = record(
        "extract_judges",
        {"judges": [Judge(name=f"J{i}", blurb="b") for i in range(judge_count)]},
    )

    async def research_one_judge(state):
        calls.setdefault("research_one_judge", []).append(state)
        judge = state["judge"]
        assert state.get("hackathon_synopsis"), (
            "worker received an empty synopsis; extract_theme must run before "
            "the orchestrator that hands the synopsis to each Send"
        )
        return {"judges": [judge.model_copy(update={"online_summary": "done"})]}

    nodes.research_one_judge_node = research_one_judge
    nodes.compile_bias_node = record("compile_bias", {"judge_bias": "BIAS"})
    nodes.review_bias_node = record("review_bias", {"judge_bias": "APPROVED"})
    nodes.generate_idea_candidates_node = record(
        "generate_idea_candidates",
        {
            "idea_candidates": [
                Idea(title=f"I{i}", pitch="p") for i in range(idea_count)
            ]
        },
    )
    nodes.rank_ideas_node = record(
        "rank_ideas",
        {"ideas": [Idea(title=f"I{i}", pitch="p") for i in range(idea_count)]},
    )

    async def research_one_idea(state):
        calls.setdefault("research_one_idea", []).append(state)
        return {"ideas": [state["idea"].model_copy(update={"research": "done"})]}

    nodes.research_one_idea_node = research_one_idea
    nodes.compile_ideas_node = record("compile_ideas", {"final_ideas": []})
    nodes.select_idea_node = record("select_idea", {"selected_idea": "1"})
    nodes.plan_build_node = record(
        "plan_build", {"build_plan": None, "build_tasks": []}
    )
    nodes.choose_build_dir_node = record("choose_build_dir", {"build_dir": "/tmp/x"})

    async def build_app(state):
        calls.setdefault("build_app", []).append(state)
        return {"build_result": "ok"}

    nodes.build_app_node = build_app


def test_fan_in_nodes_run_exactly_once():
    originals = {k: getattr(nodes, k) for k in dir(nodes) if k.endswith("_node")}
    calls: dict = {}
    try:
        _stub_nodes(calls, judge_count=3, idea_count=4)
        importlib.reload(graph_module)  # rebind the graph to the stubbed nodes
        graph = graph_module.build_graph(MemorySaver())

        async def run():
            config = {"configurable": {"thread_id": "topology"}}
            async for _ in graph.astream(
                {"devpost_url": "https://x.devpost.com"}, config, stream_mode="updates"
            ):
                pass

        asyncio.run(run())
    finally:
        for name, fn in originals.items():
            setattr(nodes, name, fn)
        importlib.reload(graph_module)

    assert len(calls.get("research_one_judge", [])) == 3, calls.keys()
    assert len(calls.get("research_one_idea", [])) == 4, calls.keys()

    # The bug: these fired once per incoming branch instead of once overall.
    assert len(calls.get("compile_bias", [])) == 1, (
        f"compile_bias ran {len(calls.get('compile_bias', []))}x; more than once "
        "means concurrent judge_bias writes (InvalidUpdateError)"
    )
    assert (
        len(calls.get("compile_ideas", [])) == 1
    ), f"compile_ideas ran {len(calls.get('compile_ideas', []))}x"

    # And it must see fully researched judges, not a half-finished fan-out.
    seen = calls["compile_bias"][0]["judges"]
    assert all(
        j.online_summary for j in seen
    ), f"compile_bias ran before judge research finished: {seen}"
    print("ok: compile_bias and compile_ideas each run once, on complete data")


if __name__ == "__main__":
    test_fan_in_nodes_run_exactly_once()
    print("\nAll graph topology checks passed.")
