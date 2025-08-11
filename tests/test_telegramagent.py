#%%
# import modules 
import sys
import os 
PROJECT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_PATH, 'telegramagent/src'))
import dotenv

# from telegramagent.src.telegramagent import TelegramAgent
from telegramagent import TelegramAgent
from webhook import BaseWebhook

# set environment variables
dotenv.load_dotenv(override=True)
os.environ["PORT"] = "5001"
os.environ["ENV"] = "dev"

#%%
# run the telegram agent
webhook=BaseWebhook() 
# webhook=None
telegram_agent=TelegramAgent(webhook=webhook)
await telegram_agent.listen_async()

# %%
