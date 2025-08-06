#%%
# import modules 
import sys
import os 
PROJECT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_PATH)
import dotenv

from telegramagent.src.graphrunner import LangflowRunner, LanggraphRunner

dotenv.load_dotenv(override=True)
os.environ["ENV"] = "dev"

#%%
# test the LangflowRunner
graph_runner = LangflowRunner()
response= await graph_runner.run("hello",)

# %%
# test the LanggraphRunner
graph_runner = LanggraphRunner()
response= await graph_runner.run("hello",)
# %%
graph_runner = LanggraphRunner()
response= await graph_runner.run("book a stay at the McKormik hotel",)
# %%
