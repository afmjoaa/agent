"""
Simple ReAct Agent with MCP integration and session memory.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from langgraph.constants import START
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# Simple storage for active agents
_contexts: Dict[str, Any] = {}

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RESET = '\033[0m'

# System message and quit indicators
SYSTEM_MESSAGE = """You are a helpful assistant. Be concise.
Say goodbye/bye/farewell when user wants to end the conversation."""

QUIT_WORDS = ["goodbye", "bye", "farewell"]

# Agent state schema
class AgentState(MessagesState):
    quit: bool

def create_agent(llm: BaseChatModel, tools: list[BaseTool]) -> Any:
    """Create a ReAct agent using LangGraph."""
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages

        response = llm_with_tools.invoke(messages)
        should_quit = any(word in (response.content or "").lower() for word in QUIT_WORDS)
        return {"messages": [response], "quit": should_quit}

    # Build graph
    graph = StateGraph(AgentState)  # type: ignore
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools=tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()

@dataclass
class AgentContext:
    """Container for agent session."""
    agent: Any
    config: Dict[str, Any]
    agent_id: str

async def initialize(agent_id: str, thread_id: str, mcp_config: Dict, llm_config: Dict) -> AgentContext:
    """Initialize a new agent."""
    print(f"Initializing agent {agent_id}...")

    # Get tools from MCP
    mcp_client = MultiServerMCPClient(mcp_config)
    tools = await mcp_client.get_tools()
    print(f"Retrieved {len(tools)} tools")

    # Initialize LLM
    llm = init_chat_model(
        model=llm_config["model"],
        api_key=llm_config.get("api_key"),
        temperature=llm_config.get("temperature", 0),
    )

    # Create agent
    agent = create_agent(llm, tools)

    return AgentContext(
        agent=agent,
        config={"configurable": {"thread_id": thread_id}},
        agent_id=agent_id,
    )

async def process(message: str, context: AgentContext) -> Dict[str, Any]:
    """Process a message through the agent."""
    response = await context.agent.ainvoke(
        {"messages": [{"role": "user", "content": message}], "quit": False},
        context.config,
    )

    return {
        "response": response["messages"][-1].content,
        "quit": response.get("quit", False),
    }

async def chat(message: str, agent_id: str, thread_id: str,
               mcp_config: Optional[Dict] = None, llm_config: Optional[Dict] = None) -> Dict:
    """
    Main chat function with session memory.
    First call: needs mcp_config and llm_config
    Subsequent calls: reuses stored context
    """
    context = _contexts.get(agent_id)

    if context is None:
        if not mcp_config or not llm_config:
            raise ValueError("First call requires mcp_config and llm_config")
        context = await initialize(agent_id, thread_id, mcp_config, llm_config)
        _contexts[agent_id] = context

    result = await process(message, context)

    # Cleanup on quit
    if result["quit"]:
        _contexts.pop(agent_id, None)

    return result

async def demo():
    """Demo chat loop."""
    mcp_config = {
        "whatsapp": {
            "command": "/Users/afmjoaa/.local/bin/uv",
            "args": [
                "--directory",
                "/Users/afmjoaa/PycharmProjects/agents/whatsapp-mcp/whatsapp-mcp-server",
                "run",
                "main.py",
            ],
            "transport": "stdio",
        },
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "transport": "streamable_http",
            "headers": {"Authorization": f"Bearer {os.environ.get('GITHUB_PAT', '')}"},
        },
    }

    llm_config = {
        "model": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0,
    }

    print("=== Agent Chat ===")
    print("Type 'quit' or 'bye' to end\n")

    first_msg = True

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        start = time.time()
        try:
            if first_msg:
                result = await chat(user_input, "agent-001", "thread-001", mcp_config, llm_config)
                first_msg = False
            else:
                result = await chat(user_input, "agent-001", "thread-001")

            elapsed = time.time() - start
            print(f"{Colors.GREEN}Agent: {result['response']}{Colors.RESET}")
            print(f"({elapsed:.2f}s)\n")

            if result["quit"]:
                print("Goodbye!")
                break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    asyncio.run(demo())