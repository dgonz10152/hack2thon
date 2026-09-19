"""Judge notes: keep the raw search results behind each judge summary.

The judge researchers see far more than their 4-8 sentence summaries keep.
Their raw search snippets are stored in Chroma, tagged by run and judge, so
compile_bias can pull the most relevant evidence back out, reranked.
"""

import hashlib
import os
from functools import cache

from flashrank import Ranker, RerankRequest
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from agent.state import Judge

# What compile_bias needs to know about each judge; mirrors its four sections.
BIAS_TOPICS = (
    "technology preferences, domain and industry leanings, "
    "presentation and style preferences, strong opinions and red flags"
)


@cache
def store() -> Chroma:
    """The on-disk notes store. Built on first use, so importing the graph
    never touches Ollama or the disk."""
    # Not OLLAMA_BASE_URL: that may point at Ollama Cloud, which does not
    # serve the embedding model.
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b"),
        base_url=os.getenv("EMBED_BASE_URL"),
    )
    return Chroma(
        collection_name="judge_notes",
        persist_directory=".chroma",
        embedding_function=embeddings,
    )


@cache
def _ranker() -> Ranker:
    return Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir=os.path.expanduser("~/.cache/flashrank"),
    )


def rerank(query: str, passages: list[str]) -> list[str]:
    """Order passages best-first for the query with a FlashRank cross-encoder."""
    request = RerankRequest(
        query=query, passages=[{"id": i, "text": p} for i, p in enumerate(passages)]
    )
    return [hit["text"] for hit in _ranker().rerank(request)]


def save_judge_notes(db: Chroma, thread_id: str, judge: str, snippets: list[str]):
    # ponytail: snippets are stored whole; add a text splitter if we ever store
    # full pages, since the reranker only reads the first 512 tokens.
    # Ids are content hashes, so a retried or resumed worker overwrites its
    # earlier notes instead of duplicating them.
    ids = [
        hashlib.sha256(f"{thread_id}|{judge}|{s}".encode()).hexdigest()
        for s in snippets
    ]
    metadata = {"thread_id": thread_id, "judge": judge}
    db.add_texts(snippets, metadatas=[metadata] * len(snippets), ids=ids)


def find_judge_evidence(
    db: Chroma, rerank, thread_id: str, judge: Judge, top_n: int = 5
) -> list[str]:
    """The judge's stored snippets most relevant to a bias analysis, best first.

    Over-fetches by vector similarity, then lets the cross-encoder pick the
    best few. The blurb is in the query so snippets about the right person
    (company, role) outrank a same-name stranger.
    """
    query = f"{judge.name} ({judge.blurb}): {BIAS_TOPICS}"
    only_this_judge = {"$and": [{"thread_id": thread_id}, {"judge": judge.name}]}
    candidates = db.similarity_search(query, k=20, filter=only_this_judge)
    if not candidates:
        return []
    return rerank(query, [doc.page_content for doc in candidates])[:top_n]
