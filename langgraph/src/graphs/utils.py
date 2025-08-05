#%%
from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
from langchain_mcp_adapters.client import MultiServerMCPClient
import logging
from typing import List
import json
from pathlib import Path

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """
    Wrap a tool to support human-in-the-loop review.
    Ref: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/add-human-in-the-loop/#review-tool-calls
    """ 
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(  
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call."
        }
        response = interrupt([request])[0]  
        # response = validate_response(response, tool)

        # approve the tool call
        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt

async def get_mcp_tools() -> List[BaseTool]:
    """
    Load MCP tools from configuration file.
    Ref: https://langchain-ai.github.io/langgraph/agents/mcp/#use-mcp-tools
    
    Returns:
        List[BaseTool]: The list of MCP tools, or an empty list if errors occur.
    """
    mcp_tools: List[BaseTool] = []
    try:
        # Resolve path to config
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "mcps" / "mcp_config.json"

        if not config_path.exists():
            logging.warning(f"Configuration file not found: {config_path}")
            return mcp_tools

        # Load config JSON
        try:
            with open(config_path, "r") as f:
                mcp_servers = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return mcp_tools

        # Validate expected structure
        if 'mcpServers' not in mcp_servers or not isinstance(mcp_servers['mcpServers'], dict):
            logging.error("Missing or invalid 'mcpServers' section in config.")
            return mcp_tools

        connections = mcp_servers['mcpServers']
        logging.info(f"Found MCP servers: {list(connections.keys())}")

        # Attempt to connect and fetch tools
        client = MultiServerMCPClient(connections=connections)
        mcp_tools = await client.get_tools()

        logging.info(f"Found {len(mcp_tools)} MCP tools: {[tool.name for tool in mcp_tools]}")

    except Exception as e:
        logging.error(f"Unexpected error while loading MCP tools: {e}")

    return mcp_tools