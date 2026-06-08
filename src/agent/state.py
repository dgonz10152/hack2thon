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


class InputState(TypedDict):
    devpost_url: str


class OpenCodeState(TypedDict):
    prompts: str
    responses: str


class JudgeResearchState(TypedDict):
    judge: Judge
    hackathon_synposis: str


class AgentState(InputState, total=False):
    messages: Annotated[list, add_messages]
    html: str
    judges: Annotated[list[Judge], merge_judges]
    hackathon_synposis: str
    judge_bias: str


class AgentStateUpdate(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    html: str
    judges: list[Judge]
    hackathon_synposis: str
    judge_bias: str
    devpost_url: str
