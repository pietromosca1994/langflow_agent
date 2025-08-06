#%%
# import modules 
import sys
import os 
PROJECT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_PATH, 'telegramagent/src'))
import dotenv

# from telegramagent.src.telegramagent import TelegramAgent
from telegramagent import TelegramAgent
dotenv.load_dotenv(override=True)
os.environ["PORT"] = "5001"
os.environ["ENV"] = "dev"

#%%
# set environment variable 
telegram_agent=TelegramAgent()
await telegram_agent.listen_async()

# %%
