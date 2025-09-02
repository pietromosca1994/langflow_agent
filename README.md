# Langflow Agent
[Langflow UI](http://localhost:7860/flows)

## Configuration 
### MCP
Ref: MCP [Tools](https://langchain-ai.github.io/langgraph/agents/mcp/#use-mcp-tools)  
``` json
client = MultiServerMCPClient(
    {
        "math": {
            "command": "python",
            # Replace with absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        }
    }
)
```

# References
[OpenAI-Whisper](https://github.com/openai/whisper)  
[Langflow Docker](https://docs.langflow.org/deployment-docker)  
[Langflow Environment Variables](https://docs.langflow.org/environment-variables)  
[Ngrok](https://ngrok.com)
[Pinggy](https://pinggy.io)

# MCPs
[Brave Search](https://github.com/brave/brave-search-mcp-server)
[Alpaca](https://github.com/brave/brave-search-mcp-server)