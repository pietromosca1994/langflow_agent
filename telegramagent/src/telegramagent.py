import os 
from telegram import User, Update, Message, Chat, ReplyKeyboardMarkup, InlineKeyboardButton, InlineKeyboardMarkup, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from telegram.constants import ChatAction
import logging
import re
import asyncio
import textwrap
from typing import Union, Literal
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import uvicorn
import threading
from pydantic import BaseModel
import datetime

from graphrunner import LangflowRunner, LanggraphRunner
from models import WebhookBody

MAX_LENGTH = 4096  # Telegram message limit
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")
WHISPER=False
LLM_FRAMEWORK='langgraph'
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT=5000

async def keep_typing(bot, chat_id, stop_event):
    """Keep sending typing indicator every 4 seconds until stopped"""
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)  # Typing indicator lasts ~5 seconds
        except Exception as e:
            logging.error(f"Error in keep_typing: {e}")
            break
class TelegramAgent:
    def __init__(self, 
                 verbose=logging.INFO):
        
        self.init_logger(verbose)
        self.init_app()
        self.init_llm(LLM_FRAMEWORK)
        self.init_text_to_speech()
        self.init_webhook_server()

    def listen(self):
        """Start the bot with proper event loop handling"""
        self.start_webhook_server()
        self.logger.info("Bot is running and waiting for messages...")
        
        # Check if we're in an environment with an existing event loop
        try:
            loop = asyncio.get_running_loop()
            self.logger.info("Existing event loop detected. Using async method.")
            # If there's already a running loop, use the async version
            return self.listen_async()
        except RuntimeError:
            # No running loop, safe to use run_polling
            self.logger.info("No existing event loop. Using run_polling.")
            self.app.run_polling()

    async def listen_async(self):
        """Async version for environments with existing event loops"""
        # Initialize the application
        self.start_webhook_server()
        await self.app.initialize()
        await self.app.start()
        
        # Start polling
        await self.app.updater.start_polling()
        self.logger.info("Bot is running and waiting for messages...")
        
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal. Shutting down...")
        finally:
            # Clean shutdown
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    def listen_in_thread(self):
        """Alternative: Run the bot in a separate thread"""
        import threading
        self.start_webhook_server()
        def run_bot():
            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            
            try:
                self.logger.info("Bot is running in separate thread...")
                self.app.run_polling()
            finally:
                new_loop.close()
        
        # Start the bot in a separate thread
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        self.logger.info("Bot started in background thread.")
        return bot_thread
        
    def init_app(self):
        self.app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
        self.app.add_handler(CallbackQueryHandler(self.handle_button_choice))

    def init_text_to_speech(self): 
        if WHISPER==True:
            import whisper
            self.whisper_model=whisper.load_model("small")
            self.tmp_folder=os.path.join(os.getcwd(), "temp")
            os.makedirs(self.tmp_folder, exist_ok=True)  # Ensure temp folder exists

    def init_llm(self, framework: Literal['langflow', 'langgraph']):
        if framework == 'langflow':
            self.llm=LangflowRunner(verbose=self.logger.level)
        elif framework == 'langgraph':
            self.llm=LanggraphRunner(verbose=self.logger.level)

    def init_webhook_server(self):
        """Initialize FastAPI webhook server"""
        self.webhook_app = FastAPI(title="Telegram Agent Webhook Server")
        
        # Store reference to self for use in route handlers
        self.webhook_app.state.agent = self
        
    #     # Add webhook routes
    #     self.setup_webhook_routes()

    # def setup_webhook_routes(self):
    #     """Setup webhook routes"""
        
        @self.webhook_app.post("/webhook/message")
        async def handle_webhook_message(
            request: Request,
            body: WebhookBody,
            background_tasks: BackgroundTasks
        ):
            """Handle incoming webhook messages"""
            try:
                # # Optional: Verify webhook secret
                # auth_header = request.headers.get("Authorization")
                # if WEBHOOK_SECRET and auth_header != f"Bearer {WEBHOOK_SECRET}":
                #     raise HTTPException(status_code=401, detail="Invalid authorization")
                
                # Log the incoming webhook message
                self.logger.info(f"Webhook message from {body.source}: {body.text}")
                
                # Process message in background
                background_tasks.add_task(
                    self._handle_webhook_response,
                    body
                )
                
                return JSONResponse({
                    "status": "success",
                    "message": "Message received and queued for processing"
                })
                
            except Exception as e:
                self.logger.error(f"Error handling webhook message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.webhook_app.get("/webhook/ping")
        async def ping():
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=jsonable_encoder({
                    'service': 'TelegramAgent Microservice',
                    'version': '0.1.0',
                    'date_time': datetime.datetime.now().isoformat()
                })
            )
    
    async def _handle_webhook_response(self, body: WebhookBody):
        text=body.text
        chat_id=body.chat_id
        chat=await self.app.bot.get_chat(chat_id)
        update=Update(update_id=1,
                      message=Message(message_id=1,
                                      text=text,
                                      date=datetime.datetime.now(),
                                      from_user=User(id=chat_id, 
                                                     first_name=chat.first_name,
                                                     last_name=chat.last_name,
                                                     username=chat.username, 
                                                     is_bot=True),
                                      chat=Chat(id= chat_id,
                                                type=chat.type)
                                     ),
                      )
        await self._handle_text_message(update)
        pass
    
    def start_webhook_server(self):
        """Start the webhook server in a separate thread"""

        host = DEFAULT_HOST if "HOST" not in os.environ else os.environ["HOST"]
        if host == DEFAULT_HOST:
            logging.warning("Using default host")

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
    
        # @self.webhook_app.post("/webhook/custom/{source}")
        # async def handle_custom_webhook(
        #     source: str,
        #     request: Request,
        #     background_tasks: BackgroundTasks
        # ):
        #     """Handle custom webhook formats from different sources"""
        #     try:
        #         # Get raw body
        #         body = await request.body()
                
        #         # Try to parse as JSON
        #         try:
        #             data = json.loads(body.decode('utf-8'))
        #         except json.JSONDecodeError:
        #             data = {"raw_body": body.decode('utf-8')}
                
        #         # Convert to standard format
        #         message = self.convert_custom_webhook(source, data)
                
        #         if message:
        #             self.logger.info(f"Custom webhook from {source}: {message.text}")
                    
        #             # Process message in background
        #             background_tasks.add_task(
        #                 self.process_webhook_message,
        #                 message
        #             )
                
        #         return JSONResponse({
        #             "status": "success",
        #             "source": source,
        #             "message": "Webhook processed"
        #         })
                
        #     except Exception as e:
        #         self.logger.error(f"Error handling custom webhook from {source}: {e}")
        #         raise HTTPException(status_code=500, detail=str(e))
    async def send_text_message(self, text: str, chat_id: int, reply_markup: Union[InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove] = None):
        '''
        chat_id: Refers to the unique ID of the chat (group, channel, or 1-on-1).
        user_id: Refers to the unique ID of a user.
        In private chats (where the bot and user communicate directly), chat_id is equal to user.id.
        The telegram bot cannot initiate a conversation with a user unless the user has already started the chat by sending a message or clicking “Start” first. This is a Telegram platform restriction.
        '''
        try:
            bot = self.app.bot
            for chunk in self._split_message(text):
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode="Markdown",
                    reply_markup=reply_markup
                )
        except Exception as e:
            self.logger.error(f"Failed to send message to {chat_id}: {e}")

    async def send_reply(
        self, 
        reply: str, 
        update: Update, 
        reply_markup: Union[InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove] = None
    ):
        try:
            # Determine the correct message object
            message = update.message or (
                update.callback_query.message if update.callback_query else None
            )

            if message and reply:
                for chunk in self._split_message(reply):
                    await message.reply_text(chunk, parse_mode="Markdown", reply_markup=reply_markup)
            else:
                await message.reply_text("Sorry, I couldn't process your message right now.")

        except Exception as e:
            self.logger.error(f'Failed to send reply: {e}')

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

    async def reply_with_llm(self, incoming: str, chat_id: int, *args, **kwargs):

        stop_typing = asyncio.Event()
    
        # Start continuous typing indicator
        typing_task = asyncio.create_task(
            keep_typing(self.app.bot, chat_id, stop_typing)
        )
        
        try:
            # Process with LLM if needed
            if incoming:
                message = await self.llm.run(incoming)

                # handle human-in-the-loop interaction 
                if message.interrupt==True:
                    keyboard = [
                        [InlineKeyboardButton("✅ Accept", callback_data="accept")],
                        [InlineKeyboardButton("❌ Deny", callback_data="deny")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await self.send_text_message(message.text, chat_id, reply_markup=reply_markup)
                else:
                    await self.send_text_message(message.text, chat_id)
            else:
                await self.send_text_message(None, chat_id)
            # await asyncio.sleep(10)
        finally:
            # Stop the typing indicator
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    def _log_message(self, update: Update):
        self.logger.info(
            f"\n📨 Incoming message\n"
            f"   👤 From:    {update.message.from_user.username}\n"
            f"   🆔 Chat ID: {update.effective_chat.id}\n"
            f"   💬 Message: {update.message.text}"
        )

    async def _handle_text_message(self, 
                                   update: Update):
        incoming = update.message.text
        chat_id = update.effective_chat.id
        self._log_message(update)

        await self.reply_with_llm(incoming, chat_id)

        pass
    
    async def _handle_voice_message(self,
                                    update: Update):
        voice = update.message.voice
        chat_id = update.effective_chat.id
        file = await self.app.bot.get_file(voice.file_id)
        
        # download the voice message to a temporary file
        file_path = os.path.join(self.tmp_folder, f"voice.ogg")
        await file.download_to_drive(file_path)
        
        # text to speech conversion
        incoming = self.text_to_speech(file_path)
        self._log_message(update)

        await self.reply_with_llm(incoming, chat_id)
        
        pass

    async def handle_button_choice(self, update: Update):
        chat_id = update.effective_chat.id
        query = update.callback_query
        await query.answer()

        choice = query.data  # 'accept' or 'deny'

        # You could send this back to your LangGraph as:
        # Command(resume=[{"type": choice}])
        await query.edit_message_text(f"You selected *{choice.upper()}*.", parse_mode="Markdown")
        self.logger.info(f"Selected: {choice}")

        await self.reply_with_llm(choice, chat_id)

    @staticmethod
    def _split_message(text: str, max_length: int = MAX_LENGTH) -> list[str]:
        """
        Split a text message into chunks that respect Telegram's max_length limit
        while preserving valid Markdown formatting, especially code blocks.

        Args:
            text (str): The input text to split.
            max_length (int): Maximum length of each chunk (default: 4096).

        Returns:
            list[str]: List of valid Markdown chunks.
        """
        if not text:
            return []

        chunks = []
        current_chunk = []
        current_length = 0
        in_code_block = False
        
        # Split text into lines to process them individually
        lines = text.split('\n')

        for line in lines:
            line_len = len(line) + 1  # +1 for the newline character

            # Check for code block delimiters
            if '```' in line:
                # If there are multiple ``` on one line, handle them sequentially
                parts = line.split('```')
                for i, part in enumerate(parts):
                    if i % 2 == 0:  # Text outside of potential '```'
                        if in_code_block: # If already in a code block, this `part` is plain text within it
                            if current_length + len(part) + 3 + 1 > max_length: # +3 for ```, +1 for newline
                                if current_chunk:
                                    chunks.append('\n'.join(current_chunk) + '\n```') # Close current code block
                                current_chunk = ['```'] # Start new code block
                                current_length = 3
                            current_chunk.append(part)
                            current_length += len(part) + 1
                        else: # Not in code block, regular text part
                            if current_length + len(part) + 1 > max_length:
                                if current_chunk:
                                    chunks.append('\n'.join(current_chunk))
                                current_chunk = []
                                current_length = 0
                            current_chunk.append(part)
                            current_length += len(part) + 1
                    else: # This 'part' is between '```' markers or is empty
                        if in_code_block: # Closing a code block
                            if current_chunk:
                                chunks.append('\n'.join(current_chunk) + '```')
                                current_chunk = []
                                current_length = 0
                            in_code_block = False
                        else: # Opening a code block
                            if current_length > 0: # Add any accumulated text before the code block
                                chunks.append('\n'.join(current_chunk))
                                current_chunk = []
                                current_length = 0
                            current_chunk.append('```' + part)
                            current_length += 3 + len(part) # Account for ``` and content
                            in_code_block = True
                            
                # Special handling for lines ending or starting with ```
                if line.endswith('```') and not in_code_block: # A closing ``` just ended a block
                    pass # Already handled by the logic above
                elif line.startswith('```') and in_code_block: # An opening ``` just started a block
                    pass # Already handled
                elif line.count('```') % 2 != 0: # Unclosed code block on this line
                    in_code_block = not in_code_block
                    if in_code_block: # Just entered a code block
                        if current_length > 0:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = []
                            current_length = 0
                        current_chunk.append(line)
                        current_length += line_len
                    else: # Just exited a code block
                        current_chunk.append(line)
                        current_length += line_len
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                else: # No ` ` ` on this line or balanced ` ` `
                    if current_length + line_len > max_length:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                        current_length = line_len
                    else:
                        current_chunk.append(line)
                        current_length += line_len
            else: # Not a code block delimiter line
                if in_code_block:
                    if current_length + line_len > max_length - 3: # Need room for closing ```
                        chunks.append('\n'.join(current_chunk) + '\n```')
                        current_chunk = ['```'] # Start new code block chunk
                        current_length = 3 + line_len
                        current_chunk.append(line)
                    else:
                        current_chunk.append(line)
                        current_length += line_len
                else:
                    if current_length + line_len > max_length:
                        # If current accumulated chunk plus new line exceeds max_length
                        # Try to split the accumulated chunk by sentences/paragraphs if possible
                        temp_chunk_content = '\n'.join(current_chunk)
                        
                        # Split by paragraphs
                        paragraphs = re.split(r'(\n{2,})', temp_chunk_content)
                        temp_buffer = []
                        temp_buffer_len = 0
                        
                        for p_idx, p in enumerate(paragraphs):
                            if temp_buffer_len + len(p) + 1 <= max_length:
                                temp_buffer.append(p)
                                temp_buffer_len += len(p) + 1
                            else:
                                if temp_buffer:
                                    chunks.append('\n'.join(temp_buffer))
                                temp_buffer = [p]
                                temp_buffer_len = len(p) + 1
                        
                        if temp_buffer:
                            chunks.append('\n'.join(temp_buffer))
                        
                        # Start a new chunk with the current line
                        current_chunk = [line]
                        current_length = line_len
                    else:
                        current_chunk.append(line)
                        current_length += line_len

        # Add any remaining content in current_chunk to chunks
        if current_chunk:
            final_chunk = '\n'.join(current_chunk)
            if in_code_block:
                final_chunk += '\n```' # Ensure code block is closed
            chunks.append(final_chunk)

        # Final pass to ensure all chunks are within max_length and properly closed
        # This handles cases where a single line or segment was longer than MAX_LENGTH
        # and ensures all Markdown elements are balanced.
        final_safe_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                # For chunks still too long, perform a hard wrap, trying to be smart about markdown
                # This is a fallback and might break complex markdown if a single line is too long.
                sub_chunks = textwrap.wrap(chunk, width=max_length, break_long_words=False, 
                                            replace_whitespace=False, break_on_hyphens=False)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    # Attempt to re-balance common markdown if broken by textwrap
                    if sub_chunk.count('**') % 2 != 0:
                        sub_chunk += '**'
                    if sub_chunk.count('*') % 2 != 0 and sub_chunk.count('**') != sub_chunk.count('*'):
                        sub_chunk += '*'
                    if sub_chunk.count('`') % 2 != 0:
                        sub_chunk += '`'
                    
                    final_safe_chunks.append(sub_chunk.strip()) # strip to remove extra newlines from wrapping
            else:
                # Ensure all markdown delimiters are balanced
                # This part is crucial for making sure each chunk is valid Markdown.
                balanced_chunk = chunk
                
                # Balance triple backticks
                if balanced_chunk.count('```') % 2 != 0:
                    if balanced_chunk.strip().startswith('```'):
                        balanced_chunk += '\n```'
                    elif balanced_chunk.strip().endswith('```'):
                        balanced_chunk = '```\n' + balanced_chunk

                # Balance single backticks (inline code)
                if balanced_chunk.count('`') % 2 != 0 and '```' not in balanced_chunk:
                    balanced_chunk += '`'

                # Balance bold/italic. This is tricky without a full Markdown parser.
                # A simple check for unmatched pairs, assuming they don't span across ```
                if '```' not in balanced_chunk:
                    if balanced_chunk.count('**') % 2 != 0:
                        balanced_chunk += '**'
                    if balanced_chunk.count('*') % 2 != 0 and balanced_chunk.count('**')*2 != balanced_chunk.count('*'):
                        balanced_chunk += '*'
                
                final_safe_chunks.append(balanced_chunk.strip()) # strip to remove extra newlines

        # Remove any empty strings that might have resulted from splitting
        return [chunk for chunk in final_safe_chunks if chunk]
        
    def text_to_speech(self, audio)->Union[str, None]:
        try:
            if WHISPER == True:
                result = self.whisper_model.transcribe(audio,
                                                        language="en",
                                                        beam_size=5,             # Better decoding (default is 5)
                                                        best_of=5,               # Tries multiple candidates
                                                        fp16=False 
                                                        )
                text=result["text"]
            else: 
                text=None
            return text
        
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None
        