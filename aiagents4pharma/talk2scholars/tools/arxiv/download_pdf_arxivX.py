#!/usr/bin/env python3
"""
arxiv_paper_fetch: Tool for fetching and downloading an arXiv paper by its ID.
"""

import os
import logging
import requests
import yaml  
import xml.etree.ElementTree as ET
from typing import Annotated, Any, Dict

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

# Ensure logging is configured to show info messages
logging.basicConfig(level=logging.INFO)


def load_config() -> dict:
    """
    Load configuration from the YAML file.
    The path is computed relative to this file's location.
    """
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../../configs/tools/download_pdf_arxiv/default.yaml"
    )
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Config file is empty.")
            return config
    except Exception as e:
        logging.error("Failed to load configuration: %s", str(e))
        # Fallback to defaults if necessary
        return {
            "api_url": "http://export.arxiv.org/api/query",
            "request_timeout": 10,
        }

# Load configuration values
CONFIG = load_config()
API_URL = CONFIG.get("api_url", "http://export.arxiv.org/api/query")
REQUEST_TIMEOUT = CONFIG.get("request_timeout", 10)


class FetchArxivPaperInput(BaseModel):
    """Input schema for the arXiv paper fetching tool."""
    paper_id: str = Field(
        description="The arXiv paper ID to fetch. Example: '1905.02244' or '2109.12345v2'."
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

    # For Pydantic v2 use model_config; if using v1, define an inner Config class:
    model_config = {"arbitrary_types_allowed": True}
    # Alternatively, for Pydantic v1:
    # class Config:
    #     arbitrary_types_allowed = True


@tool(args_schema=FetchArxivPaperInput)
def fetch_arxiv_paper(
    paper_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Dict[str, Any]:
    """
    Fetch an arXiv paper by its ID, download the PDF locally,
    and return metadata and download information.

    Args:
        paper_id (str): The arXiv paper ID (e.g., '1905.02244' or '2109.12345v2').
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.

    Returns:
        Dict[str, Any]: A dictionary containing the download information and file path,
                        along with a message formatted for a state update.
    """
    logging.info("Starting arXiv paper fetch for paper ID: %s", paper_id)

    # Construct the API URL using the value loaded from config
    api_url = f"{API_URL}?search_query=id:{paper_id}&start=0&max_results=1"
    logging.info("Fetching metadata from: %s", api_url)

    try:
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        error_message = f"Exception occurred while fetching metadata for paper {paper_id}: {str(e)}"
        logging.error(error_message)
        return Command(update={"download_info": error_message})

    if response.status_code != 200:
        error_message = (
            f"Error: Failed to fetch metadata for paper {paper_id}. "
            f"Status code: {response.status_code}"
        )
        logging.error(error_message)
        return Command(update={"download_info": error_message})

    # Parse the XML response from the arXiv API
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as pe:
        error_message = f"Error parsing XML for paper {paper_id}: {str(pe)}"
        logging.error(error_message)
        return Command(update={"download_info": error_message})

    pdf_url = None
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break
        if pdf_url:
            break

    if not pdf_url:
        error_message = f"Error: Could not find a PDF link for paper {paper_id}."
        logging.error(error_message)
        return Command(update={"download_info": error_message})

    logging.info("Downloading PDF from: %s", pdf_url)
    local_filename = f"arxiv_{paper_id.replace('/', '_')}.pdf"

    try:
        pdf_response = requests.get(pdf_url, stream=True, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        error_message = f"Exception occurred while downloading PDF for paper {paper_id}: {str(e)}"
        logging.error(error_message)
        return Command(update={"download_info": error_message})

    if pdf_response.status_code == 200:
        with open(local_filename, "wb") as f:
            for chunk in pdf_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        success_message = f"Successfully downloaded {paper_id} and saved as {local_filename}"
        logging.info(success_message)
        return Command(
            update={
                "arxiv_pdf": local_filename,
                "download_info": success_message,
                "tool_message": success_message,  # Key for downstream needs.
            }
        )
    else:
        error_message = (
            f"Failed to download paper {paper_id}; HTTP status code: {pdf_response.status_code}"
        )
        logging.error(error_message)
        return Command(update={"download_info": error_message})


if __name__ == "__main__":
    sample_paper_id = "2502.07400"   # Replace with a valid paper ID
    sample_tool_call_id = "test123"   # Dummy tool call id for testing
    input_data = {"paper_id": sample_paper_id, "tool_call_id": sample_tool_call_id}
    
    # Use .invoke() instead of calling the function directly
    result = fetch_arxiv_paper.invoke(input_data)
    print(result)
