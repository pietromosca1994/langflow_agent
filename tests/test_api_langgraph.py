#%%
import requests

BASE_URL = "http://langgraph:7861"  # Change if your server is running elsewhere

def test_ping():
    url = f"{BASE_URL}/ping"
    response = requests.get(url)
    print("🔁 Testing /ping")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
    print("-" * 40)
    return response

def test_run(graph_id="test_graph", content="Hello from test script", session_id="test_session"):
    url = f"{BASE_URL}/api/v1/run/{graph_id}"
    payload = {
        "content": content,
        "session_id": session_id
    }
    response = requests.post(url, json=payload)
    print(f"🔁 Testing /api/v1/run/{graph_id}")
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw Response:", response.text)
    print("-" * 40)
    return response

#%%
# test ping 
test_ping()

#%%
# test run
response=test_run('test',
                  content="book a stay at McKittrick hotel")
print(response.json())
# %%
