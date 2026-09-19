from langchain_core.utils import print_text
import os
import sys
from typing import TypedDict, Annotated
import operator
import asyncio
# Set UTF-8 encoding for Windows terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

# from tools.tavily_tool import tavily_search
from mcp_client import tevely_mcp_search,aviation_mcp_call,extract_destination,forecast_mcp_search,weather_mcp_search  
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

# create the state

class travelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    

# flight node
# def search_flight_agent(state: travelState):
#     user_query = state["user_query"]
#     flight_data = search_flights(user_query)
#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(content="Flight information fetched")
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }

# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""

def search_flight_agent(state: travelState):
     print("\nINSIDE FLIGHT AGENT\n")
     query = state["user_query"]
     
     try:
        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )
        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )
        
        
        response=llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content
     except Exception as e:
         flight_data = f"Flight information unavailable: {str(e)}"
    
     return {
            "flight_results": flight_data,
            "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
     }

# hotel node
def hotel_agent(state: travelState):
    query = f"Best hotels for {state['user_query']}"
    # hotel_results = tavily_search(query)
    hotel_results = asyncio.run(tevely_mcp_search(query))

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def weather_agent(state: travelState):

    city = extract_destination(state["user_query"])

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }




# Itinerary node
def itinerary_agent(state: travelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# creating the graph
graph = StateGraph(travelState)

graph.add_node("flight_agent", search_flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", END)

# memory checkpointer with postgreSql
_conn = psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

# compile the graph

app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_kartik"
        }
    }

    print("Travel Agent started. Type 'exit' to stop.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "stop"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        for chunk in app.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ],
                "user_query": user_input,
                "flight_results": "",
                "hotel_results": "",
                "itinerary": "",
                "llm_calls": 0
            },
            config=config,
            stream_mode="messages"
        ):
            message, metadata = chunk

            if message.content:
                print(message.content, end="", flush=True)

        print("\n")

    
