import urllib.request

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


@tool
def fetch_html(url: str) -> str:
    """Fetches the HTML content of a webpage, used to read its content"""
    with urllib.request.urlopen(url) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8")


@tool
def search_web(query: str) -> str:
    """Searches the web using DuckDuckGo and returns a summary of results."""
    try:
        return DuckDuckGoSearchRun().run(query)
    except Exception as exc:
        # Search runs inside a ReAct loop for every judge and every idea, so a
        # single transient DNS blip or rate-limit would otherwise abort a
        # pipeline that has already spent many minutes. Handing the failure
        # back as a tool result keeps the agent in control: it can retry with a
        # different query or finish and say nothing was found. Deliberately
        # broad, because the failure modes span DDGS, httpx and DNS errors.
        # Deliberately discourages repeated retries. An earlier version invited
        # the agent to "try a different query", which under DuckDuckGo rate
        # limiting turned every failure into an endless retry loop that burned
        # through the graph recursion limit instead of finishing.
        return (
            f"SEARCH FAILED ({type(exc).__name__}): {exc}\n"
            "Retry at most once with a different query. If it fails again, stop "
            "searching, work with what you already have, and state plainly that "
            "web results were unavailable. Do not invent results."
        )
