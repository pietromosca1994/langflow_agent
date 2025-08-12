import sys
from fastapi import BackgroundTasks, FastAPI, status, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
import datetime
import os
import logging
import threading
import importlib
from langgraph.graph.state import CompiledStateGraph 
from langgraph.types import Command 
from typing import Dict
from models import InvokeModel
from dataclasses import dataclass 
import logging

VERBOSE=logging.INFO

logging.basicConfig(
    level=VERBOSE,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

GRAPH_ID = "test_graph"
app = FastAPI(title='LangGraph', description='LangGraph Microservice', version='0.1.0')

@dataclass
class Graph:
    graph: CompiledStateGraph
    interrupt: bool = False

graph_cache: Dict[str, Graph] = {}  # Cache for loaded graphs
graph_cache_lock = threading.Lock()  # to protect cache updates

def get_graph_names(graphs_dir="graphs"):
    """
    Extract graph names from files in the format graph_<graph_name>.py
    """
    graph_names = []
    
    try:
        files = os.listdir(graphs_dir)
        
        for file in files:
            # Check if file matches pattern graph_*.py
            if file.startswith("graph_") and file.endswith(".py"):
                # Extract the graph name by removing "graph_" prefix and ".py" suffix
                graph_name = file[6:-3]  # Remove first 6 chars ("graph_") and last 3 chars (".py")
                graph_names.append(graph_name)

        logging.info(f'Available graphs {graph_names}')      
    except FileNotFoundError:
        logging.error(f"Directory '{graphs_dir}' not found")
    except Exception as e:
        logging.error(f"Error reading directory: {e}") 

@app.get("/")
async def ping():
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder({
            'service': 'LangGraph Microservice',
            'version': '0.1.0',
            'date_time': datetime.datetime.now().isoformat()
        })
    )

@app.post("/api/v1/run/{graph_id}")
async def invoke(graph_id: str, body: InvokeModel):
    logging.debug(f'Calling graph {graph_id}')
    
    # Load graph if not cached
    if graph_id not in graph_cache:
        with graph_cache_lock:
            if graph_id not in graph_cache:  # double-checked locking
                try:
                    module_name = f"graphs.graph_{graph_id}"
                    # logging.debug(f'Looking for module {module_name}')
                    # logging.debug(f'sys.path: {sys.path[:3]}')  # Show first 3 entries
                    # logging.debug(f'Current working dir: {os.getcwd()}')
                    graph_module = importlib.import_module(module_name)
                    logging.debug(f'Successfully imported {module_name}')
                    logging.debug(f'Module attributes: {dir(graph_module)}')

                    # Check if get_graph exists
                    if not hasattr(graph_module, 'get_graph'):
                        raise AttributeError(f"Module {module_name} has no get_graph function")
                    
                    compiled_graph = await graph_module.get_graph()
                    graph_cache[graph_id] = Graph(graph=compiled_graph)
                    logging.info(f"Graph {graph_id} loaded successfully.")
                
                except ImportError as e:
                    logging.error(f'ImportError: {e}')
                    raise HTTPException(status_code=404, detail=f"Failed to load graph module '{module_name}': {str(e)}")
                except AttributeError as e:
                    logging.error(f'AttributeError: {e}')
                    raise HTTPException(status_code=500, detail=f"Graph module '{module_name}' missing get_graph function: {str(e)}")
                except Exception as e:
                    logging.error(f'Unexpected error loading graph: {e}')
                    raise HTTPException(status_code=500, detail=f"Error loading graph '{graph_id}': {str(e)}")
    
    graph_data = graph_cache[graph_id]
    logging.debug(f"Graph {graph_id} found in cache: {graph_data}")

    if graph_data.interrupt==True:
        input_data = Command(resume=[{"type": body.content}])
    else:
        input_data = {
            'messages': [
                {
                    'role': 'user',
                    'content': body.content
                }
            ]
        }
    config = {"configurable": {"thread_id": body.session_id}}

    try:
        # ainvoke ensures that the graph is invoked asynchronously (for MCP tools)
        response = await graph_data.graph.ainvoke(input_data, config)
        
        if '__interrupt__' in response.keys():
            graph_data.interrupt = True
            # logging.debug(f"Interrupt True detected for graph {graph_id}.")
        else:
            graph_data.interrupt = False
            # logging.debug(f"Interrupt False detected for graph {graph_id}.")

    except Exception as e:
        logging.error(f"Graph invocation error: {e}")
        raise HTTPException(status_code=500, detail="Error during graph invocation")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=jsonable_encoder(response)
    )

def listen():
    host = "127.0.0.1" if "HOST" not in os.environ else os.environ["HOST"]
    if host == "127.0.0.1":
        logging.warning("Using default host")

    port = 7861 if "PORT" not in os.environ else int(os.environ["PORT"])
    logging.info(f"Running {app.title} @ {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False)

if __name__=='__main__':    
    get_graph_names()
    listen()
