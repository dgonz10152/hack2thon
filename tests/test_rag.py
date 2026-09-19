"""Judge notes store: isolation by run and judge, idempotent saves, reranking.

Hermetic - a real Chroma store in a temp dir, with fake embeddings and a fake
reranker, so no Ollama and no model download.

Run:  uv run python tests/test_rag.py
"""

import tempfile

from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

from agent import rag
from agent.state import Judge

ADA = Judge(name="Ada", blurb="analytical engine")


def temp_store(directory: str) -> Chroma:
    return Chroma(
        collection_name="judge_notes",
        persist_directory=directory,
        embedding_function=DeterministicFakeEmbedding(size=32),
    )


def reverse_alphabetical(_query: str, passages: list[str]) -> list[str]:
    return sorted(passages, reverse=True)


def test_evidence_is_scoped_to_one_run_and_one_judge():
    with tempfile.TemporaryDirectory() as tmp:
        db = temp_store(tmp)
        rag.save_judge_notes(db, "run-1", "Ada", ["ada a", "ada b"])
        rag.save_judge_notes(db, "run-1", "Alan", ["alan a"])
        rag.save_judge_notes(db, "run-2", "Ada", ["ada from another run"])

        found = rag.find_judge_evidence(db, reverse_alphabetical, "run-1", ADA)
        assert found == ["ada b", "ada a"], found


def test_saving_twice_does_not_duplicate():
    with tempfile.TemporaryDirectory() as tmp:
        db = temp_store(tmp)
        rag.save_judge_notes(db, "run-1", "Ada", ["same snippet"])
        rag.save_judge_notes(db, "run-1", "Ada", ["same snippet"])
        assert len(db.get()["ids"]) == 1


def test_keeps_only_the_top_reranked_snippets():
    with tempfile.TemporaryDirectory() as tmp:
        db = temp_store(tmp)
        rag.save_judge_notes(db, "run-1", "Ada", [f"snippet {n}" for n in range(9)])
        found = rag.find_judge_evidence(
            db, reverse_alphabetical, "run-1", ADA, top_n=3
        )
        assert found == ["snippet 8", "snippet 7", "snippet 6"], found


def test_empty_store_returns_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        found = rag.find_judge_evidence(
            temp_store(tmp), reverse_alphabetical, "run-1", ADA
        )
        assert found == [], found


if __name__ == "__main__":
    test_evidence_is_scoped_to_one_run_and_one_judge()
    test_saving_twice_does_not_duplicate()
    test_keeps_only_the_top_reranked_snippets()
    test_empty_store_returns_nothing()
    print("ok")
