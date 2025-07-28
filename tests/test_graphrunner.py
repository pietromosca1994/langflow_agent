#%%
# import modules 
import sys
import os 
PROJECT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_PATH)
import dotenv

from telegramagent.src.graphrunner import LangflowRunner

dotenv.load_dotenv(override=True)

#%%
graph_runner = LangflowRunner()
response= await graph_runner.run("hello",)
# %%
