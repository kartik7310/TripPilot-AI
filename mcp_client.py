import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
client = MultiServerMCPClient(
     { 
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
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
        },
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

search_tool = None
aviation_tool = {}

async def initlize_mcp():
    
    global search_tool
    global aviation_tool
    if search_tool is not None and aviation_tool is not None:
        return
    tools = await client.get_tools()
    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)

    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    aviation_tool = next(
        tool
        for tool in tools
        if tool.name != "tavily_search"
    )


async def tevely_mcp_search(query:str):
    await initlize_mcp()
    result = await search_tool.ainvoke({
        "query":query
    })
    return result

tavily_mcp_search = tevely_mcp_search


async def aviation_mcp_call(tool_name:str,tool_args:dict=None):
    tools = await client.get_tools()
    tool = next(
        t for t in tools
        if t.name == tool_name
    )

    result = await tool.ainvoke(
        tool_args or {}
    )

    return result


async def get_airports():
    return await aviation_mcp_call("list_airports")


async def get_airlines():
    return await aviation_mcp_call("list_airlines")
    
weather_tool = None
forecast_tool = None

async def initialize_weather_tools():

    global weather_tool, forecast_tool

    if weather_tool is not None:
        return

    tools = await client.get_tools()

    weather_tool = next(
        t for t in tools
        if t.name == "get_current_weather"
    )

    forecast_tool = next(
        t for t in tools
        if t.name == "get_forecast"
    )


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tool.ainvoke(
        {
            "city": city
        }
    )
    


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await forecast_tool.ainvoke(
        {
            "city": city
        }
    )





if __name__ == "__main__":
    async def _test():
        print("Testing MCP tools...")
        tools = await client.get_tools()
        print("Tools:", [t.name for t in tools])

    asyncio.run(_test())

