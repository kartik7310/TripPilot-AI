import os
import sys
from typing import TypedDict, Annotated
import operator

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

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
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
def search_flight_agent(state: travelState):
    user_query = state["user_query"]
    flight_data = search_flights(user_query)
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# hotel node
def hotel_agent(state: travelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
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


# Final Response Agent
def final_agent(state: travelState):

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# creating the graph
graph = StateGraph(travelState)

graph.add_node("flight_agent", search_flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

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

    
