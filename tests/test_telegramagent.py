#%%
# import modules 
import sys
import os 
PROJECT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_PATH)
import dotenv

from telegramagent.src.telegramagent import TelegramAgent

dotenv.load_dotenv(override=True)

#%%
telegram_agent=TelegramAgent()
await telegram_agent.listen_async()
# %%
