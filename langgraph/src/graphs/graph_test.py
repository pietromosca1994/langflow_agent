#%%
import os 
import dotenv
from typing import Callable
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages.system import SystemMessage

dotenv.load_dotenv(override=True)

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

def validate_response(response, tool):
    """Validate the response from the agent."""
    if not isinstance(response, dict):
        raise ValueError("Response must be a dictionary.")
    if "messages" not in response:
        raise ValueError("Response must contain 'messages' key.")
    if not isinstance(response["messages"], list):
        raise ValueError("'messages' must be a list.")
    return response


def book_hotel(hotel_name: str):
    """
    Book a hotel reservation. Use this immediately when user specifies a hotel name.
    
    This tool should be called whenever a user mentions booking, reserving, or staying at a specific hotel.
    Do not ask for clarification if the hotel name is provided in the user's message.
    
    Args:
        hotel_name: The name of the hotel mentioned by the user
    """
    return f"Successfully booked a stay at {hotel_name}."

def get_graph()->CompiledStateGraph:
    checkpointer = InMemorySaver()
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
#     system_prompt = """You are a helpful assistant. 

#     CRITICAL RULE: Before calling any tool, you MUST first explain what you're about to do in natural language.

#     Format:
#     1. First: Write an explanation of the action you'll take
#     2. Then: Call the appropriate tool

# Never call a tool without first providing a clear explanation."""

    # prompt=SystemMessage(system_prompt)
    graph = create_react_agent(
        model=model,
        tools=[
            add_human_in_the_loop(book_hotel), 
        ],
        checkpointer=checkpointer,
        # prompt=prompt,
    )
    return graph

# #%%
# graph = get_graph()
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# #%%
# graph = get_graph()
# config = {"configurable": {"thread_id": "1"}}

# # Run the agent
# response=graph.invoke({"messages": [{"role": "user", "content": "book a stay at McKittrick hotel"}]}, config)
# print(response)
# # %%
# from langgraph.types import Command 
# response=graph.invoke(Command(resume=[{"type": "accept"}]), config)
# print(response)
# %%
