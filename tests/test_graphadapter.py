#%%
# import modules 
import sys
import os 
PROJECT_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_PATH)
import dotenv

from telegramagent.src.graphadapter import LangflowAdapter, LanggraphAdapter

dotenv.load_dotenv(override=True)
os.environ["ENV"] = "dev"

#%%
# test the LangflowAdapter
graph_runner = LangflowAdapter()
response= await graph_runner.run("hello",)

# %%
# test the LanggraphAdapter
graph_runner = LanggraphAdapter(host='localhost')
response= await graph_runner.run("hello",)
# %%
graph_runner = LanggraphAdapter(host='localhost')
response= await graph_runner.run("book a stay at the McKormik hotel",)
# %%
