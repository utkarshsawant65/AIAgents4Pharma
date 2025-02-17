#!/usr/bin/env python3

"""
arxiv_paper_fetch: Tool for fetching and downloading an arXiv paper by its ID.

This tool interacts with the arXiv API to fetch metadata and download research papers
as PDFs based on their unique arXiv ID.
"""

import logging
import requests
import xml.etree.ElementTree as ET
from typing import Annotated, Any, Dict
import hydra
from langchain_core.tools import tool
#from langgraph.types import Command
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FetchArxivPaperInput(BaseModel):
    """Input schema for the arXiv paper fetching tool."""
    
    paper_id: str = Field(
        description="The arXiv paper ID to fetch. Example: '1905.02244' or '2109.12345v2'."
    )
    tool_call_id: Annotated[str, str]  # Required for LangChain tool call tracking

    model_config = {"arbitrary_types_allowed": True}

# Load Hydra configuration for the arXiv paper fetch tool.
with hydra.initialize(version_base=None, config_path="../../configs"):
    cfg = hydra.compose(
        config_name="config", overrides=["+tools/download_pdf_arxiv=default"]
    )
    API_URL = cfg.tools.download_pdf_arxiv.api_url
    REQUEST_TIMEOUT = cfg.tools.download_pdf_arxiv.request_timeout


@tool(args_schema=FetchArxivPaperInput)
def fetch_arxiv_paper(paper_id: str, tool_call_id: str) -> Dict[str, Any]:
    """
    Fetches an arXiv paper by its ID, retrieves its metadata, and downloads the PDF.

    This function interacts with the arXiv API to fetch metadata based on a paper ID.
    If a PDF link is found, the paper is downloaded and stored locally.

    Args:
        paper_id (str): The arXiv paper ID (e.g., "1905.02244" or "2109.12345v2").
        tool_call_id (str): Unique identifier for the tool invocation.

    Returns:
        Dict[str, Any]: A dictionary containing the file path if the download is successful,
                        or an error message if the paper is not found.
    """
    logger.info("Starting arXiv paper fetch for paper ID: %s", paper_id)

    # Construct API URL
    api_url = f"{API_URL}?search_query=id:{paper_id}&start=0&max_results=1"
    logger.info("Fetching metadata from: %s", api_url)

    try:
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        error_message = f"Exception occurred while fetching metadata: {str(e)}"
        logger.error(error_message)
        return {"error": error_message}

    if response.status_code != 200:
        return {"error": f"Failed to fetch metadata for paper {paper_id}. Status code: {response.status_code}"}

    # Parse the XML response from arXiv
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as pe:
        return {"error": f"Error parsing XML for paper {paper_id}: {str(pe)}"}

    # Extract PDF link
    pdf_url = None
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

    if not pdf_url:
        return {"error": f"Error: Could not find a PDF link for paper {paper_id}."}

    logger.info("Downloading PDF from: %s", pdf_url)
    local_filename = f"arxiv_{paper_id.replace('/', '_')}.pdf"

    try:
        pdf_response = requests.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT)
        if pdf_response.status_code == 200:
            with open(local_filename, "wb") as f:
                for chunk in pdf_response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return {
                "arxiv_pdf": local_filename,
                "download_info": f"Successfully downloaded {paper_id} and saved as {local_filename}"
            }
        else:
            return {"error": f"Failed to download paper {paper_id}; HTTP status code: {pdf_response.status_code}"}
    except Exception as e:
        return {"error": f"Exception occurred while downloading PDF: {str(e)}"}


# Uncomment this block to manually test the function
"""if __name__ == "__main__":
     sample_paper_id = "2502.07400"  # Replace with a valid paper ID
     sample_tool_call_id = "test123"  # Dummy tool call ID for testing
     input_data = {"paper_id": sample_paper_id, "tool_call_id": sample_tool_call_id}
     
     # Test tool function
     result = fetch_arxiv_paper.invoke(input_data)
     print(result)"""
