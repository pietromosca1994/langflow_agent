from abc import ABC, abstractmethod
import aiohttp
import logging 
import os 
from urllib.parse import urljoin
from pydantic import BaseModel
from typing import Literal, Union
import requests 
from requests.exceptions import RequestException

LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")
LANGFLOW_PORT = 7860
LANGGRAPH_FLOW_ID = 'test'
LANGGRAPH_PORT = 7861
ENV: Literal['prod', 'dev'] = os.getenv('ENV', 'prod')

class AIMessage(BaseModel):
    text: str
    session_id: str
    interrupt: bool = False

class GraphRunner(ABC):
    def __init__(self, 
                 base_url: str, 
                 verbose: int = logging.INFO):
        self.init_logger(verbose)
        self.base_url = base_url
        self._ping_server(base_url)
        pass

    def _ping_server(self, url: str) -> bool:
        """Ping the server with a HEAD request."""
        try:
            response = requests.head(url, timeout=3)
            self.logger.info(f'Successfully connected to graph server @ {url}')
            return response.status_code < 400
        except RequestException as e:
            self.logger.warning(f"Ping to graph server @ {url} failed: {e}")
            return False
    
    @abstractmethod
    def run(self, graph_id: str, config: dict = None):
        """
        Run the graph with the given ID and configuration.
        
        :param graph_id: The ID of the graph to run.
        :param config: Optional configuration for the graph.
        """
        pass 
    
    def init_logger(self, verbose=logging.INFO):
        """Initialize the logger for the Telegram agent."""
        self.logger = logging.getLogger('api')  # Get a logger unique to the class
        self.logger.setLevel(verbose)  # Set the logging level
        
        # Check if handlers are already added (to prevent duplicate logs)
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # Console handler (logs to terminal)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Optional: File logging (uncomment if needed)
            # file_handler = logging.FileHandler(f"{self.__class__.__name__}.log")
            # file_handler.setFormatter(formatter)
            # self.logger.addHandler(file_handler)

            # Prevent logs from propagating to the root logger
            self.logger.propagate = False

    async def _post(self, url: str, payload: dict, headers: dict)-> Union[dict, None]:
        """
        Internal method to post data to the graph.
        
        :param graph_id: The ID of the graph.
        :param body: The data to post to the graph.
        """
        try:
            # Use aiohttp for async HTTP requests
            timeout = aiohttp.ClientTimeout(total=1200)  # 20 minutes timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    
                    response_json = await response.json()

                    return response_json

        except aiohttp.ClientError as e:
            self.logger.error(f"Error making async API request: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
        pass

    @abstractmethod
    def parse_response(self, response: dict) -> AIMessage:
        """
        Parse the response from the graph.
        
        :param response: The JSON response from the graph.
        :return: An AIMessage object containing the parsed response.
        """
        pass

    @abstractmethod
    async def run(self, message: str) -> AIMessage:
        """
        Run the specified graph with the given message.
        
        :param message: The message to process.
        :return: An AIMessage object containing the response.
        """
        pass

class LangflowRunner(GraphRunner):
    """
    LangflowRunner is a concrete implementation of GraphRunner that runs graphs using the Langflow API.
    """
    def __init__(self, verbose: int = logging.INFO):
        match ENV:
            case 'dev':
                host='localhost'
                port=5000
            case 'prod':
                host='langgraph'
                port=LANGFLOW_PORT

        base_url=f'http://{host}:{port}/api/v1/run'
        super().__init__(base_url, verbose)
        pass

    def parse_response(self, response: dict=None) -> AIMessage:
        if response: 
            text = response['outputs'][0]['outputs'][0]["results"]["message"]["text"]
            text = text.strip()

            session_id = response.get("session_id", "default_session")
            
            self.logger.info(f'Session ID {session_id}')
            message=AIMessage(text=text, 
                              session_id=response["session_id"],
                              interrupt=False) #TODO: handle interrupt in response
        else: 
            text='The graph  is unavailable at this time'
            message=AIMessage(text=text,
                              session_id='None',
                              interrupt=False
            )                   
        return message
    
    async def run(self, message: str)->AIMessage:
        """
        Run the specified graph with the given configuration.
        """
        
        payload = {
            "input_value": message,
            "output_type": "chat",
            "input_type": "chat"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LANGFLOW_TOKEN}"
        }

        url = urljoin(self.base_url + '/', LANGFLOW_FLOW_ID)

        response=await self._post(url, payload, headers)
        message = self.parse_response(response)

        return message
    
class LanggraphRunner(GraphRunner):
    def __init__(self, verbose: int = logging.INFO):
        match ENV:
            case 'dev':
                host='localhost'
                port=5000
            case 'prod':
                host='langgraph'
                port=LANGGRAPH_PORT

        base_url=f'http://{host}:{port}/api/v1/run'
        super().__init__(base_url, verbose)
        pass

    def parse_response(self, response: dict) -> AIMessage:
        """
        Parse the response from the Langflow API.
        
        :param response: The JSON response from the API.
        :return: The text content of the AI message.
        """
        if response: 
            content = response['messages'][-1]['content'].strip()
            if "__interrupt__" in response.keys(): 
                description = response['__interrupt__'][-1]['value'][-1]['description']
                text = content + f"\n\nPlease review the tool call: {description}"
                interrupt = True
            else:
                text = content
                interrupt = False

            message=AIMessage(text=text,
                            session_id=response.get("session_id", "default_session"),
                            interrupt=interrupt)
        else: 
            text='The graph  is unavailable at this time'
            message=AIMessage(text=text,
                              session_id='None',
                              interrupt=False
            )    
        return message
    
    async def run(self, message: str, session_id: str = 'default')->AIMessage:
        """
        Run the specified graph with the given configuration.
        """
    
        payload = {
            "content": message,
            "session_id": session_id
        }
        headers = {
            "Content-Type": "application/json"
        }

        url = urljoin(self.base_url + '/', LANGGRAPH_FLOW_ID)

        response=await self._post(url, payload, headers)
        message = self.parse_response(response)

        return message