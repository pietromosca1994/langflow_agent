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

async def get_mcp_tools(
    config_path: Path = Path(__file__).parent.parent / "mcps" / "mcp_config.json"
) -> List[BaseTool]:
    """
    Load MCP tools from configuration file.
    Ref: https://langchain-ai.github.io/langgraph/agents/mcp/#use-mcp-tools
    """
    mcp_tools: List[BaseTool] = []

    try:
        if not config_path.exists():
            logging.warning(f"Configuration file not found: {config_path}")
            return mcp_tools

        # Load config JSON
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return mcp_tools

        # Validate expected structure
        mcp_servers = config_data.get("mcpServers")
        if not isinstance(mcp_servers, dict):
            logging.error("Missing or invalid 'mcpServers' section in config.")
            return mcp_tools

        logging.info(f"Found MCP servers: {list(mcp_servers.keys())}")

        # Attempt to connect and fetch tools
        for mcp_server, connection_config in mcp_servers.items():
            try:
                client = MultiServerMCPClient(connections={mcp_server: connection_config})
                server_tools = await client.get_tools()
                mcp_tools.extend(server_tools)
                logging.info(
                    f"Successfully loaded MCP server '{mcp_server}' "
                    f"with tools: {[tool.name for tool in server_tools]}"
                )
            except Exception as e:
                logging.error(f"Failed to load MCP server '{mcp_server}': {e}")

    except Exception as e:
        logging.error(f"Unexpected error while loading MCP tools: {e}")

    return mcp_tools