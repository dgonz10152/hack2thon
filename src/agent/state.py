from typing import TypedDict, Annotated

from pydantic import BaseModel, model_validator
from langgraph.graph.message import add_messages


class Judge(BaseModel):
    name: str
    blurb: str
    online_summary: str = ""


class Judges(BaseModel):
    judges: list[Judge]

    @model_validator(mode="before")
    @classmethod
    def _unwrap_bare_list(cls, data):
        if isinstance(data, list):
            return {"judges": data}
        return data


def merge_judges(left: list[Judge] | None, right: list[Judge] | None) -> list[Judge]:
    """Upsert judges by name so parallel branches can enrich them independently."""
    by_name: dict[str, Judge] = {}
    for j in (left or []) + (right or []):
        existing = by_name.get(j.name)
        if existing is None:
            by_name[j.name] = j
        else:
            merged = existing.model_copy(
                update={
                    "blurb": j.blurb or existing.blurb,
                    "online_summary": j.online_summary or existing.online_summary,
                }
            )
            by_name[j.name] = merged
    return list(by_name.values())


class Idea(BaseModel):
    title: str
    pitch: str
    rationale: str = ""
    research: str = ""


class IdeaCandidates(BaseModel):
    ideas: list[Idea]

    @model_validator(mode="before")
    @classmethod
    def _unwrap_bare_list(cls, data):
        if isinstance(data, list):
            return {"ideas": data}
        return data


class FinalIdea(BaseModel):
    title: str
    problem: str
    target_users: str
    core_features: list[str]
    tech_stack: list[str]
    differentiation: str
    bias_alignment: str
    winning_formula_notes: str
    feasibility_notes: str


class FinalIdeas(BaseModel):
    ideas: list[FinalIdea]

    @model_validator(mode="before")
    @classmethod
    def _unwrap_bare_list(cls, data):
        if isinstance(data, list):
            return {"ideas": data}
        return data


class BuildTask(BaseModel):
    title: str
    description: str  # self-contained instructions for one coding-agent step
    acceptance: str = ""  # short, checkable definition of done
    status: str = ""  # "" | "ok" | "failed", filled in by build_app_node


class BuildPlan(BaseModel):
    project_summary: str
    tech_stack: list[str]
    setup_notes: str = ""  # scaffold/dir layout/run commands
    tasks: list[BuildTask]

    @model_validator(mode="before")
    @classmethod
    def _unwrap_bare_list(cls, data):
        # Some Ollama models return a bare task list instead of the full object.
        if isinstance(data, list):
            return {"project_summary": "", "tech_stack": [], "tasks": data}
        return data


def merge_ideas(left: list[Idea] | None, right: list[Idea] | None) -> list[Idea]:
    """Upsert ideas by title so parallel research branches can enrich them independently."""
    by_title: dict[str, Idea] = {}
    for i in (left or []) + (right or []):
        existing = by_title.get(i.title)
        if existing is None:
            by_title[i.title] = i
        else:
            merged = existing.model_copy(
                update={
                    "pitch": i.pitch or existing.pitch,
                    "rationale": i.rationale or existing.rationale,
                    "research": i.research or existing.research,
                }
            )
            by_title[i.title] = merged
    return list(by_title.values())


class InputState(TypedDict):
    devpost_url: str


class JudgeResearchState(TypedDict):
    judge: Judge
    hackathon_synopsis: str


class IdeaResearchState(TypedDict):
    idea: Idea
    hackathon_synopsis: str
    judge_bias: str


class AgentState(InputState, total=False):
    messages: Annotated[list, add_messages]
    html: str
    judges: Annotated[list[Judge], merge_judges]
    hackathon_synopsis: str
    judge_bias: str
    idea_candidates: list[Idea]
    ideas: Annotated[list[Idea], merge_ideas]
    final_ideas: list[FinalIdea]
    selected_idea: str
    build_plan: BuildPlan
    build_tasks: list[BuildTask]
    build_dir: str
    build_result: str
