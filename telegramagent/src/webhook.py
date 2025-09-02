import os
import logging
import threading
import datetime
from typing import Callable, Optional, Any, Dict, Union
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
from pydantic import BaseModel
from abc import ABC
from models import WebhookBody
import inspect

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000


class BaseWebhook(ABC):
    """
    A modular webhook server that can be integrated with any agent or service.
    Handles incoming HTTP requests and delegates message processing to configurable handlers.
    """
    
    def __init__(self,
                 verbose: int = logging.INFO):
        """
        Initialize the webhook server.
        
        Args:
            config: WebhookConfig instance with server settings
            logger: Logger instance (will create one if not provided)
            message_handler: Async function to handle incoming messages
        """
        self.init_logger(verbose)
        self.init_server()

        self.callback: Callable
    
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

    def init_server(self):
        """Initialize FastAPI webhook server"""
        self.webhook_app = FastAPI(title="Webhook Server")
        
        @self.webhook_app.post("/webhook/message")
        async def handle_webhook_message(
            request: Request,
            background_tasks: BackgroundTasks
        ):
            """Handle incoming webhook messages"""
            try:
                # # Optional: Verify webhook secret
                # auth_header = request.headers.get("Authorization")
                # if WEBHOOK_SECRET and auth_header != f"Bearer {WEBHOOK_SECRET}":
                #     raise HTTPException(status_code=401, detail="Invalid authorization")
                
                # Log the incoming webhook message
                await self._log_message(request)
                
                body= await self._preprocess_request(request)

                # Process message in background
                background_tasks.add_task(
                    self._handle_request,
                    body
                )
                
                response = await self._postprocess_request(request)
                
                return response
                
            except Exception as e:
                self.logger.error(f"Error handling webhook message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.webhook_app.get("/webhook/ping")
        async def ping():
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=jsonable_encoder({
                    'service': 'Webhook Server',
                    'version': '0.1.0',
                    'date_time': datetime.datetime.now().isoformat()
                })
            )
    
    async def _log_message(self, request: Request):
        headers=request.headers
        body=await request.json()
        self.logger.info(
            f"\n📨 Webhook message\n"
            f"   👤 Headers: {headers}\n"
            f"   👤 Body:    {body}\n"
        )

    def register_callback(self, callback: Callable):
        self.callback=callback

    @classmethod
    async def _preprocess_request(self, request: Request)->Union[None, dict]: 
        body=await request.json()
        return body

    @classmethod
    async def _postprocess_request(self, request: Request):
        return JSONResponse({
                    "status": "success",
                    "message": "Message received and queued for processing"
                })
         
    async def _handle_request(self, body: Union[dict, None]):  
        if body:
            if hasattr(self, "callback") and callable(self.callback):
                try:
                    if inspect.iscoroutinefunction(self.callback):
                        await self.callback(body)
                    else:
                        self.callback(body)
                except Exception as e:
                    self.logger.error(f"Error executing callback: {e}")
            else:
                self.logger.warning('Callback is not registered. Use register_callback methood to register a callback')

    def add_custom_route(self, method: str, path: str, handler: Callable):
        """Add custom routes to the webhook server"""
        if method.upper() == "GET":
            self.webhook_app.get(path)(handler)
        elif method.upper() == "POST":
            self.webhook_app.post(path)(handler)
        elif method.upper() == "PUT":
            self.webhook_app.put(path)(handler)
        elif method.upper() == "DELETE":
            self.webhook_app.delete(path)(handler)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        self.logger.info(f"Added custom route: {method.upper()} {path}")

    def start_server(self):
            """Start the webhook server in a separate thread"""

            host = DEFAULT_HOST if "HOST" not in os.environ else os.environ["HOST"]
            if host == DEFAULT_HOST:
                self.logger.warning("Using default host")

            port = DEFAULT_PORT if "PORT" not in os.environ else int(os.environ["PORT"])  
            
            def run_server():
                uvicorn.run(
                    self.webhook_app,
                    host=host,
                    port=port,
                    log_level="info"
                )
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            self.logger.info(f"Webhook server started @ {host}:{port}")
            return server_thread

class DIMOWebhook(BaseWebhook):

    async def _preprocess_request(self, request: Request):
        return super(request)
    
    async def _postprocess_request(self, request: Request):
        return super(request)
