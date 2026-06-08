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
    return DuckDuckGoSearchRun().run(query)
