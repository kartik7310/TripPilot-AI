from mcp_client import weather_mcp_search
from mcp_client import extract_destination
from mcp_client import forecast_mcp_search
from mcp_client import tevely_mcp_search
import asyncio
import json
from typing import Any
from langchain_core.messages import (AnyMessage,HumanMessage,AIMessage,SystemMessage,)
from langgraph.types import interrupt
from config import get_llm
from mcp_client import current_weather ,forecast_tool,list_airlines,list_airports,tavily_search
from state import travelState
from prompts import supervisor_agent_prompt
llm = get_llm()


def _llm_text(system:str,prompt:str)->str:
    response = llm.invoke([SystemMessage(content=system),HumanMessage(content=prompt)])
    return response.content

def _json_from_llm(text:str) -> str | dict[str,Any]:
  
        start = text.index("{")
        end= text.rindex("}")+1
        json_text =text[start:end]
        return json.loads(json_text)


def supervisor_agent(state:travelState):
    query = state["user_query"]
    prompt =supervisor_agent_prompt(query)
    raw = _llm_text("you route work to specilaist agents,Return strict JSON only ",prompt)

    parsed = _json_from_llm(raw)
    selected= parsed["selected_agents"]
    return {
        "selected_agents":selected,
        "trip_constraints":parsed["trip_constraints"],
        "supervisor_reasoning":parsed["reasoning"],
        "messages":[AIMessage(content="Supervisor created the agent plan")],
        "llm_calls  ":state.get("llm_calls", 0) + 1,
    }  


def flight_agent(state: travelState):
    query = state["user_query"]
    constraints = state["trip_constraints"]
    destination = constraints["destination"]
    airports = asyncio.run(list_airports(destination, limit=10))
    airlines = asyncio.run(list_airlines("", limit=10))
    prompt = f"""
Create flight guidance for this trip.

User request:
{query}

Trip constraints:
{constraints}

Airport MCP data:
{str(airports)[:3000]}

Airline MCP data:
{str(airlines)[:3000]}

Include likely departure/arrival airports, relevant airlines,
estimated duration, fare range, peak season warning,
and booking advice.
"""

    result = _llm_text(
        "You are a flight planning specialist.",
        prompt,
    )

    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



   
# hotel agent
def hotel_agent(state: travelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = asyncio.run(tevely_mcp_search(query))

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }



# Destination Extractor

def extract_destination(query: str):

    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()


# weather agent
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



def budget_agent(state: travelState):
    prompt = f"""
Analyze whether this trip plan is realistic for the user's budget.

User request:
{state['user_query']}

Constraints:
{state.get('trip_constraints', {})}

Flight results:
{state.get('flight_results', '')}

Hotel results:
{state.get('hotel_results', '')}

Weather results:
{state.get('weather_results', '')}

Return a concise budget assessment with:
1. estimated cost categories
2. risk areas
3. money-saving suggestions
4. whether the plan seems feasible
"""

    result = _llm_text(
        "You are a practical travel budget analyst.",
        prompt,
    )
    return {
        "budget_results": result,
        "messages": [AIMessage(content="Budget agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: travelState):
    prompt = f"""
Create a clear draft travel itinerary.

User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight results:
{state.get('flight_results', '')}

Hotel results:
{state.get('hotel_results', '')}

Weather results:
{state.get('weather_results', '')}

Budget results:
{state.get('budget_results', '')}

Make the output structured, practical, and ready for human review.
"""

    result = _llm_text(
        "You are an expert itinerary planner.",
        prompt,
    )


    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }



def human_approval_agent(state: travelState):
    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    approved = feedback["approved"]
    human_feedback = feedback["feedback"]

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


def final_response_agent(state:travelState):
    if state["approved"]:
        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

Draft itinerary:
{state['itinerary']}

Budget notes:
{state['budget_results']}
"""
    else:
        prompt = f"""
The human did not approve the draft.

Original user request:
{state['user_query']}

Draft itinerary:
{state['itinerary']}

Human feedback:
{state['human_feedback']}

Budget notes:
{state['budget_results']}
"""

    result = _llm_text(
        "You produce final user-ready travel plans.",
        prompt,
    )

    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }