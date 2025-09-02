from abc import ABC, abstractmethod
import aiohttp
import logging 
import os 
from urllib.parse import urljoin
from pydantic import BaseModel
from typing import Literal, Union
import requests 
import time
from requests.exceptions import RequestException

LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")
LANGFLOW_PORT = 7860
LANGGRAPH_FLOW_ID = 'test'
LANGGRAPH_PORT = 7861

class AIMessage(BaseModel):
    text: str
    session_id: str
    interrupt: bool = False

class BaseGraphAdapter(ABC):
    def __init__(self, 
                 host: str, 
                 port: str,
                 verbose: int = logging.INFO):
        self.host=host
        self.port=port
        self.init_logger(verbose)
        self._ping_server()
        pass

    def _ping_server(self, retries: int = 3, delay: float = 1.0) -> bool:
        """Ping the server with a HEAD request, retrying on failure."""
        attempt = 0
        url=f'http://{self.host}:{self.port}/'

        while attempt < retries:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code < 400:
                    self.logger.info(f'Successfully connected to graph server @ {url}')
                    return True
                else:
                    self.logger.warning(f"Graph server @ {url} returned status {response.status_code}")
            except RequestException as e:
                self.logger.warning(f"Ping to graph server @ {url} failed (attempt {attempt + 1}/{retries}): {e}")
            
            attempt += 1
            if attempt < retries:
                time.sleep(delay)

        self.logger.error(f"Failed to connect to graph server @ {url} after {retries} attempts")
        return False
    
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

    async def _post(self, url: str, payload: dict, headers: dict) -> Union[dict, None]:
        """
        Internal method to post data to the graph.
        """
        try:
            timeout = aiohttp.ClientTimeout(total=1200)  # 20 minutes timeout

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        # Log both the status code and response body for debugging
                        error_text = await response.text()
                        self.logger.error(
                            f"HTTP {response.status} from {url}. Response body: {error_text}"
                        )
                    response.raise_for_status()

                    return await response.json()

        except aiohttp.ClientResponseError as e:
            self.logger.error(
                f"HTTP error: {e.status} {e.message}, URL: {e.request_info.url}"
            )
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"Client error during API request: {e}")
            return None
        except (KeyError, ValueError) as e:
            self.logger.error(f"Error parsing response: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None

    @abstractmethod
    def parse_response(self, response: dict) -> AIMessage:
        """
        Parse the response from the graph.
        
        :param response: The JSON response from the graph.
        :return: An AIMessage object containing the parsed response.
        """
        pass

    @abstractmethod
    async def run(self, message: str,  session_id: str = 'default') -> AIMessage:
        """
        Run the specified graph with the given message.
        
        :param message: The message to process.
        :return: An AIMessage object containing the response.
        """
        pass

class LangflowAdapter(BaseGraphAdapter):
    """
    LangflowAdapter is a concrete implementation of BaseGraphAdapter that runs graphs using the Langflow API.
    """
    def __init__(self, 
                 host: str = 'langgraph', 
                 port: int = LANGGRAPH_PORT,
                 verbose: int = logging.INFO):
        
        super().__init__(host, port , verbose)
        self.base_url=f'http://{host}:{port}/api/v1/run'
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
    
    async def run(self, 
                  message: str,  
                  graph_id: str, 
                  session_id: str = 'default')->AIMessage:
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

        url = urljoin(self.base_url + '/', graph_id)

        response=await self._post(url, payload, headers)
        message = self.parse_response(response)

        return message
    
class LanggraphAdapter(BaseGraphAdapter):
    def __init__(self, 
                 host: str = 'langgraph', 
                 port: int = LANGGRAPH_PORT,
                 verbose: int = logging.INFO):
        super().__init__(host, port, verbose)
        self.base_url=f'http://{self.host}:{self.port}/api/v1/run'
        pass

    def parse_response(self, response: dict) -> AIMessage:
        """
        Parse the response from the Langflow API.
        
        :param response: The JSON response from the API.
        :return: The text content of the AI message.
        """
        if response: 
            content = response['messages'][-1]['content'].strip()

            # interrupt handling
            if "__interrupt__" in response.keys(): 
                description = response['__interrupt__'][-1]['value'][-1]['description']
                action_request = response['__interrupt__'][-1]['value'][-1]['action_request']
                text = content + f"\n\nPlease review the tool call: {action_request}"
                interrupt = True
            # normal response
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
    
    async def run(self, 
                  message: str, 
                  graph_id: str = LANGGRAPH_FLOW_ID, 
                  session_id: str = 'default')->AIMessage:
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

        url = urljoin(self.base_url + '/', graph_id)

        response=await self._post(url, payload, headers)
        message = self.parse_response(response)

        return message