from ast import main
import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

#load_dotenv()
load_dotenv(override=True)
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

client = MultiServerMCPClient(
    {
        "aviationstack": {
            "transport": "stdio",
            "command": r"C:\Users\karti\OneDrive\Desktop\travel-plan-agent-mcp\aviationstack-mcp\.venv\Scripts\python.exe",
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
        }
    }
)

async def main():
    tools = await client.get_tools()
    print("Available MCP Tools:\n")
    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())