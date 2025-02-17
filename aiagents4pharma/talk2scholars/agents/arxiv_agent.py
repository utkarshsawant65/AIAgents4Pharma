#!/usr/bin/env python3

import logging
import hydra
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.arxiv.download_pdf_arxivX import fetch_arxiv_paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def agent_arxiv_node(state: Talk2Scholars):
    """
    Fetches the arXiv paper using the paper ID extracted from state.
    """
    logger.info("Agent Arxiv received state: %s", state)

    paper_id = state.get("paper_id", None)
    if not paper_id:
        error_message = "No valid paper ID found in state."
        logger.error(error_message)
        return Command(update={"error": error_message})

    logger.info("Fetching arXiv paper with ID: %s", paper_id)

    response = fetch_arxiv_paper.invoke({"paper_id": paper_id, "tool_call_id": "arxiv_fetch_1"})

    if "error" in response:
        logger.error("Failed to fetch arXiv paper: %s", response["error"])
        return Command(update={"error": response["error"], "suggestion": "Try another paper ID."})

    return Command(update=response)


def get_app(thread_id, llm_model="gpt-4o-mini"):
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/arxiv_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.arxiv_agent

    tools = ToolNode([fetch_arxiv_paper])
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_arxiv", agent_arxiv_node)
    workflow.add_edge(START, "agent_arxiv")

    app = workflow.compile(checkpointer=MemorySaver())
    return app
