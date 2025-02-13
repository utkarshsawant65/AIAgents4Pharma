#!/usr/bin/env python3
"""
Agent for interacting with arXiv papers.
"""

import os
import logging
import yaml  # Ensure pyyaml is installed: pip install pyyaml
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.arxiv.download_pdf_arxivX import fetch_arxiv_paper

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> dict:
    """
    Loads the agent configuration from the YAML file located at:
    configs/agents/talk2scholars/arxiv_agent/default.yaml
    """
    # Compute the path relative to this file's location.
    config_path = os.path.join(
        os.path.dirname(__file__),
        "../configs/agents/talk2scholars/arxiv_agent/default.yaml"
    )
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("Config file is empty.")
            return config
    except Exception as e:
        logger.error("Failed to load configuration: %s", str(e))
        # Fallback default configuration
        return {
            "temperature": 0.7,
            "arxiv_agent": {
                "system_prompt": "Default system prompt for arxiv agent."
            }
        }

# Load configuration from the YAML file.
CONFIG = load_config()

def get_app(uniq_id, llm_model="gpt-4o-mini"):
    """
    Returns the LangGraph app for agent_arxiv.
    This app is integrated as a sub-agent of the main agent (talk2scholars).
    """

    def agent_arxiv_node(state: Talk2Scholars):
        """
        Node function for agent_arxiv that invokes the model with the current state.
        """
        logger.info("Creating agent_arxiv node with thread_id %s", uniq_id)
        # Pass the unique thread id and other configuration if needed.
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return response

    # Define the tool node with the fetch_arxiv_paper tool.
    tools = ToolNode([fetch_arxiv_paper])

    # Initialize the LLM using ChatOpenAI with the temperature from the configuration.
    logger.info("Using OpenAI model %s", llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=CONFIG.get("temperature", 0.7))

    # Create the agent using create_react_agent.
    # The state_modifier now comes from the configuration loaded from YAML,
    # which includes your system prompt.
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Scholars,
        state_modifier=CONFIG.get("arxiv_agent", {}),
        checkpointer=MemorySaver(),
    )

    # Build the state graph that defines the workflow.
    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_arxiv", agent_arxiv_node)
    workflow.add_edge(START, "agent_arxiv")

    # Initialize memory to persist state between graph runs.
    checkpointer = MemorySaver()

    # Compile the graph into a runnable LangChain app.
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("Compiled the graph for agent_arxiv.")
    return app
