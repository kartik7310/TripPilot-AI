
from groq import __name
import os
import sys
from typing import TypedDict, Annotated
from langchain_mcp_adapters.client import MultiServerMCPClient
import operator
import asyncio
from dotenv import load_dotenv
load_dotenv(override=True)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "stdio",
            "command": r"C:\Users\karti\OneDrive\Desktop\travel-plan-agent-mcp\.venv\Scripts\python.exe",
            "args": [
                r"C:\Users\karti\OneDrive\Desktop\travel-plan-agent-mcp\weather-mcp-server.py"
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        }
    }
)

async def main():

    print("Loading tools...")

    tools = await client.get_tools()

    print("Tools loaded!")

    for tool in tools:
        print(tool.name)

if __name__ == "__main__":
    asyncio.run(main())