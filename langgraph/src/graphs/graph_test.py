#%%
import dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
import logging

from .utils import add_human_in_the_loop, get_mcp_tools

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

dotenv.load_dotenv(override=True)

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

async def get_graph()->CompiledStateGraph:
    checkpointer = InMemorySaver()
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                                   temperature=0.5)
#     system_prompt = """You are a helpful assistant. 

#     CRITICAL RULE: Before calling any tool, you MUST first explain what you're about to do in natural language.

#     Format:
#     1. First: Write an explanation of the action you'll take
#     2. Then: Call the appropriate tool

# Never call a tool without first providing a clear explanation."""
    mcp_tools = await get_mcp_tools()

    tools=[
            add_human_in_the_loop(book_hotel), 
        ]+mcp_tools
    
    # prompt=SystemMessage(system_prompt)
    graph = create_react_agent(
        model=model,
        tools=tools,
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
