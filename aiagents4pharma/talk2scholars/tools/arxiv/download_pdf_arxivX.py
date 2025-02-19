#!/usr/bin/env python3
"""
arxiv_paper_fetch: Tool for fetching an arXiv paper by its arXiv ID and returning the PDF as an object.

This tool interacts with the arXiv API to fetch metadata and download research papers
as PDFs based on their unique arXiv ID.
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import Annotated, Any, Dict
import hydra
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.tools import tool
from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FetchArxivPaperInput(BaseModel):
    """Input schema for the arXiv paper fetching tool."""
    arxiv_id: str = Field(
        description="The arXiv ID to fetch. Example: '1905.02244' or '2109.12345v2'."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


# Use an absolute config path relative to this file's location.
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(config_name="config", overrides=["+tools/download_pdf_arxiv=default"])
    API_URL = cfg.tools.download_pdf_arxiv.api_url
    REQUEST_TIMEOUT = cfg.tools.download_pdf_arxiv.request_timeout



@tool(args_schema=FetchArxivPaperInput)
def fetch_arxiv_paper(arxiv_id: str, tool_call_id: str) -> Dict[str, Any]:
    """
    Fetch an arXiv paper's metadata and download its PDF, returning the PDF as an object.

    Args:
        arxiv_id (str): The arXiv ID of the paper.
        tool_call_id (str): Unique tool call identifier.

    Returns:
        Dict[str, Any]: A Command update containing the PDF object and a ToolMessage.
    """
    print("Starting arXiv paper fetch...")

    # Construct the API URL using the arXiv ID and fetch metadata.
    api_url = f"{API_URL}?search_query=id:{arxiv_id}&start=0&max_results=1"
    logger.info("Fetching metadata from: %s", api_url)
    response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    # Parse the XML response.
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    pdf_url = next(
        (
            link.attrib.get("href")
            for entry in root.findall("atom:entry", ns)
            for link in entry.findall("atom:link", ns)
            if link.attrib.get("title") == "pdf"
        ),
        None,
    )

    if not pdf_url:
        raise ValueError(f"Could not find a PDF link for arXiv ID {arxiv_id}.")

    logger.info("Downloading PDF from: %s", pdf_url)
    pdf_response = requests.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT)
    pdf_response.raise_for_status()

    # Read the PDF data as binary chunks and join them.
    pdf_data = b"".join(chunk for chunk in pdf_response.iter_content(chunk_size=1024) if chunk)
    message = f"Successfully downloaded PDF for arXiv ID {arxiv_id} as an object."
    logger.info(message)

    return Command(
        update={
            "pdf_object": pdf_data,
            "messages": [
                ToolMessage(content=message, tool_call_id=tool_call_id)
            ],
        }
    )


if __name__ == "__main__":
    # Simulated state: a dictionary of filtered papers.
    # Each paper is stored with details including an "arXiv ID".
    filtered_papers = {
        "paper1": {
            "Title": "Example Title",
            "Abstract": "Example Abstract",
            "Year": "2020",
            "Citation Count": "15",
            "URL": "http://example.com",
            "arXiv ID": "1905.02244"
        },
        # Additional papers could be added here.
    }

    # Extract the arXiv ID from the state. For instance, we use the paper with key "paper1".
    arxiv_id = filtered_papers["paper1"]["arXiv ID"]

    # Prepare the input for the tool.
    test_input = {
        "arxiv_id": arxiv_id,
        "tool_call_id": "test123"
    }
    
    # Run the tool using the invoke method.
    result = fetch_arxiv_paper.invoke(test_input)
    
    # Display the results.
    print("PDF Object (first 100 bytes):", result.update["pdf_object"][:100])
    for msg in result.update["messages"]:
        print("Tool Message:", msg.content)
