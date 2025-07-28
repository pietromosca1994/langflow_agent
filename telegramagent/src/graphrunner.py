from abc import ABC, abstractmethod
import aiohttp
import logging 
import os 
from urllib.parse import urljoin

LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")

class GraphRunner(ABC):
    def __init__(self, 
                 base_url: str, 
                 verbose: int = logging.INFO):
        self.init_logger(verbose)
        self.base_url = base_url
        pass
    
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

    async def _post(self, url: str, payload: dict, headers: dict):
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
                    
                    response_data = await response.json()
                    text = response_data['outputs'][0]['outputs'][0]["results"]["message"]["text"]
                    text = text.strip()
                    
                    self.logger.info(f'Session ID {response_data["session_id"]}')
                    return text

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

class LangflowRunner(GraphRunner):
    """
    LangflowRunner is a concrete implementation of GraphRunner that runs graphs using the Langflow API.
    """
    def __init__(self, verbose: int = logging.INFO):
        # base_url='http://langflow:7860/api/v1/run'
        base_url='http://localhost:7860/api/v1/run'
        super().__init__(base_url, verbose)
        pass

    async def run(self, message: str):
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

        return await self._post(url, payload, headers)
    
# def LangflowRunner(GraphRunner):