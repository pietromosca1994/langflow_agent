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

logging.basicConfig(
    level=logging.DEBUG,
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

@app.get("/ping")
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
    # Load graph if not cached
    if graph_id not in graph_cache:
        with graph_cache_lock:
            if graph_id not in graph_cache:  # double-checked locking
                try:
                    module_name = f"graphs.graph_{graph_id}"
                    graph_module = importlib.import_module(module_name)
                    compiled_graph = graph_module.get_graph()
                    graph_cache[graph_id] = Graph(graph=compiled_graph)
                    logging.info(f"Graph {graph_id} loaded successfully.")
                except ModuleNotFoundError:
                    raise HTTPException(status_code=404, detail=f"Graph module '{module_name}' not found")
                except AttributeError:
                    raise HTTPException(status_code=500, detail=f"Graph module '{module_name}' missing get_graph function")

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
        # Use the actual input content instead of hardcoded strin
        response = graph_data.graph.invoke(input_data, config)
        
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

    port = 5000 if "PORT" not in os.environ else int(os.environ["PORT"])
    logging.info(f"Running {app.title} @ {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=False)

if __name__=='__main__':
    listen()
