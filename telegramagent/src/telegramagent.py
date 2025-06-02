import os 
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction
import logging 
import requests
import whisper
from typing import Union
import re
import asyncio
import aiohttp
import textwrap

MAX_LENGTH = 4096  # Telegram message limit
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LANGFLOW_TOKEN = os.getenv("LANGFLOW_TOKEN")
LANGFLOW_FLOW_ID = os.getenv("LANGFLOW_FLOW_ID")

async def keep_typing(context, chat_id, stop_event):
    """Keep sending typing indicator every 4 seconds until stopped"""
    while not stop_event.is_set():
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)  # Typing indicator lasts ~5 seconds
        except Exception as e:
            logging.error(f"Error in keep_typing: {e}")
            break

def with_langflow_api(func):
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        stop_typing = asyncio.Event()
        
        # Start continuous typing indicator
        typing_task = asyncio.create_task(
            keep_typing(context, update.effective_chat.id, stop_typing)
        )
        
        try:
            # Call the original handler
            incoming = await func(self, update, context, *args, **kwargs)
            
            # Process with Langflow if needed
            if incoming:
                reply = self.call_langflow_api(incoming)
                if reply:
                    for chunk in self._split_message(reply):
                        await update.message.reply_text(chunk, parse_mode="Markdown")

                else:
                    await update.message.reply_text("Sorry, I couldn't process your message right now.")
        finally:
            # Stop the typing indicator
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
    
    return wrapper

class TelegramAgent:
    def __init__(self, 
                 verbose=logging.INFO):
        
        self.init_logger(verbose)
        self.init_app()
        self.whisper_model=whisper.load_model("small")
        self.temp_folder=os.path.join(os.getcwd(), "temp")
        os.makedirs(self.temp_folder, exist_ok=True)  # Ensure temp folder exists
    
    def run(self):
        self.logger.info("Bot is running and waiting for messages...")
        self.app.run_polling()
        
    def init_app(self):
        self.app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        self.app.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
       
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

    async def reply_with_langflow_api(self, incoming: str, update: Update, context: ContextTypes.DEFAULT_TYPE):

        stop_typing = asyncio.Event()
    
        # Start continuous typing indicator
        typing_task = asyncio.create_task(
            keep_typing(context, update.effective_chat.id, stop_typing)
        )
        
        try:
            # Process with Langflow if needed
            if incoming:
                reply = await self.call_langflow_api(incoming)
                if reply:
                    for chunk in self._split_message(reply):
                        await update.message.reply_text(chunk, parse_mode="Markdown")

                else:
                    await update.message.reply_text("Sorry, I couldn't process your message right now.")
            # await asyncio.sleep(10)
        finally:
            # Stop the typing indicator
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

    async def _handle_text_message(self, 
                                   update: Update, 
                                   context: ContextTypes.DEFAULT_TYPE):
        incoming = update.message.text
        chat_id = update.effective_chat.id
        from_user__username=update.message.from_user.username
        self.logger.info(f"Message from {from_user__username}: {incoming}")

        await self.reply_with_langflow_api(incoming, update, context)

        pass
    
    async def _handle_voice_message(self,
                                    update: Update, 
                                    context: ContextTypes.DEFAULT_TYPE):
        voice = update.message.voice
        chat_id = update.effective_chat.id
        from_user__username=update.message.from_user.username
        file = await context.bot.get_file(voice.file_id)
        
        # download the voice message to a temporary file
        file_path = os.path.join(self.temp_folder, f"voice.ogg")
        await file.download_to_drive(file_path)
        
        # text to speech conversion using Whisper
        incoming = self.text_to_speech_whisper(file_path)
        self.logger.info(f"Message from {from_user__username}: {incoming}")

        await self.reply_with_langflow_api(incoming, update, context)
        
        pass

    @staticmethod
    def _split_message(text, max_length=MAX_LENGTH):
        """
        Splits a Markdown-formatted text into chunks under max_length,
        ensuring Markdown formatting stays valid (balanced * _ `).
        """
        def is_balanced(s):
            return (
                s.count("*") % 2 == 0 and
                s.count("_") % 2 == 0 and
                s.count("`") % 2 == 0
            )

        def safe_append(buffer, chunks):
            if buffer.strip():
                chunks.append(buffer.strip())

        paragraphs = re.split(r"(\n\s*\n)", text)  # Keep paragraph breaks
        chunks = []
        buffer = ""

        for para in paragraphs:
            if len(buffer) + len(para) <= max_length:
                buffer += para
            else:
                # Break paragraph into sentences
                sentences = re.split(r'(?<=[.!?]) +', para)
                for sentence in sentences:
                    if len(buffer) + len(sentence) <= max_length:
                        buffer += sentence
                    else:
                        if is_balanced(buffer):
                            safe_append(buffer, chunks)
                            buffer = sentence
                        else:
                            # Try to fix imbalance by dropping last token (fallback)
                            buffer = buffer.rstrip("*_`")
                            safe_append(buffer, chunks)
                            buffer = sentence

        safe_append(buffer, chunks)

        # Fallback: final pass to ensure no chunk exceeds max_length
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_length:
                wrapped = textwrap.wrap(chunk, width=max_length, break_long_words=False, replace_whitespace=False)
                final_chunks.extend(wrapped)
            else:
                final_chunks.append(chunk)

        return final_chunks

    async def call_langflow_api(self, input_value: str) -> Union[str, None]:
        """
        Async version using aiohttp - RECOMMENDED APPROACH
        """
        self.logger.info(f"Calling Langflow...")
        # flow_id = '51efbc8d-c17f-4503-b694-8c5adb7578d5/api/v1/run/ba452ad5-8cc7-41cc-a73d-7f42e3dddcc6'
        # url = f"https://api.langflow.astra.datastax.com/lf/{flow_id}"

        url=f"http://langflow:7860/api/v1/run/{LANGFLOW_FLOW_ID}"
 

        payload = {
            "input_value": input_value,
            "output_type": "chat",
            "input_type": "chat"
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LANGFLOW_TOKEN}"
        }

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
        
    def text_to_speech_whisper(self, audio)->Union[str, None]:
        try:
            result = self.whisper_model.transcribe(audio,
                                                    language="en",
                                                    beam_size=5,             # Better decoding (default is 5)
                                                    best_of=5,               # Tries multiple candidates
                                                    fp16=False 
                                                    )
            return result["text"]
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return None
        