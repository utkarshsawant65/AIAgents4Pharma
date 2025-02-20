#!/usr/bin/env python3

"""
This tool is used to search for academic papers on Semantic Scholar,
including metadata such as title, abstract, year, citation count, URL, and arXiv ID.
"""

import logging
from typing import Annotated, Any, Dict, Optional
import hydra
import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Input schema for the search papers tool."""

    query: str = Field(
        description="Search query string to find academic papers."
        " Be specific and include relevant academic terms."
    )
    limit: int = Field(
        default=2, description="Maximum number of results to return", ge=1, le=100
    )
    year: Optional[str] = Field(
        default=None,
        description="Year range in format: YYYY for specific year, "
        "YYYY- for papers after year, -YYYY for papers before year, or YYYY:YYYY for range",
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Load Hydra configuration
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(config_name="config", overrides=["tools/search=default"])
    cfg = cfg.tools.search


@tool(args_schema=SearchInput)
def search_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    limit: int = 2,
    year: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for academic papers on Semantic Scholar.

    Args:
        query (str): The search query string to find academic papers.
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        limit (int, optional): The maximum number of results to return. Defaults to 2.
        year (str, optional): Year range for papers.
        Supports formats like "2024-", "-2024", "2024:2025". Defaults to None.

    Returns:
        Dict[str, Any]: The search results including title, abstract, citation count, URL, and arXiv ID.
    """
    logger.info("Starting paper search...")
    endpoint = cfg.api_endpoint
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "paperId,title,abstract,year,citationCount,url,externalIds",
    }

    # Add year parameter if provided
    if year:
        params["year"] = year

    response = requests.get(endpoint, params=params, timeout=10)
    data = response.json()
    papers = data.get("data", [])

    logger.info("Received %d papers", len(papers))

    # Create a dictionary to store the papers
    filtered_papers = {
        paper["paperId"]: {
            "Title": paper.get("title", "N/A"),
            "Abstract": paper.get("abstract", "N/A"),
            "Year": paper.get("year", "N/A"),
            "Citation Count": paper.get("citationCount", "N/A"),
            "URL": paper.get("url", "N/A"),
            "arXiv ID": paper.get("externalIds", {}).get("ArXiv", "N/A"),  # Extract arXiv ID
        }
        for paper in papers
        if paper.get("title") and paper.get("authors")
    }

    return Command(
        update={
            "papers": filtered_papers,  # Now sending the dictionary directly
            "messages": [
                ToolMessage(content=f"Search Successful: {filtered_papers}",
                tool_call_id=tool_call_id)
            ],
        }
    )
