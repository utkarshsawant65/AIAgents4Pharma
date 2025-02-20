#!/usr/bin/env python3
"""
This module contains the arxiv_agent used for interacting with arXiv API
to fetch paper metadata and PDFs for the Talk2Scholars project.
"""
import logging
import hydra
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.arxiv.download_pdf_arxivx import fetch_arxiv_paper

#initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model="gpt-4o-mini"):
    """""
    This function returns the langgraph app.
    """""
    def agent_arxiv_node(state: Talk2Scholars):
        """
        Fetches the arXiv paper using the paper ID extracted from state.
        """
        logger.log(logger.info("Creating Agent Arxiv with thread_id: %s",uniq_id))
        response = model.invoke(state, {"configurable": {"thread_id": uniq_id}})
        return response

    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars arxiv agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/arxiv_agent=default"]
            )
        cfg = cfg.agents.talk2scholars.arxiv_agent

#define the tools
    tools = ToolNode([fetch_arxiv_paper])

#define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)
    llm = ChatOpenAI(model=llm_model, temperature=cfg.temperature)

#create the agent
    model = create_react_agent(
        llm,
        tools=tools,
        state_schema=Talk2Scholars,
        # prompt=cfg.arxiv_agent,
        state_modifier=cfg.arxiv_agent,
        checkpointer=MemorySaver(),
    )
#define new graph
    workflow = StateGraph(Talk2Scholars)

#defining 2 cycle nodes
    workflow.add_node("agent_arxiv", agent_arxiv_node)

#entering into the agent
    workflow.add_edge(START, "agent_arxiv")

#starting memory of states between graph runs
    checkpointer = MemorySaver()

#compiling the graph
    app = workflow.compile(checkpointer=checkpointer)

#logging the information and returning the app
    logger.log(logging.INFO, "Compiled the graph")
    return app
