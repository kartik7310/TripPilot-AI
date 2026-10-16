import asyncio
import json
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt
from config import get_llm
from mcp_client import (
    weather_mcp_search,
    forecast_mcp_search,
    tevely_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,
)
from state import travelState
from prompts import supervisor_agent_prompt

llm = get_llm()


def _llm_text(system: str, prompt: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return response.content


def _json_from_llm(text: str) -> str | dict[str, Any]:
    start = text.index("{")
    end = text.rindex("}") + 1
    json_text = text[start:end]
    return json.loads(json_text)


def supervisor_agent(state: travelState):
    query = state["user_query"]
    prompt = supervisor_agent_prompt(query)
    raw = _llm_text("you route work to specilaist agents,Return strict JSON only ", prompt)

    parsed = _json_from_llm(raw)
    selected = parsed["selected_agents"]
    return {
        "selected_agents": selected,
        "trip_constraints": parsed["trip_constraints"],
        "supervisor_reasoning": parsed["reasoning"],
        "messages": [AIMessage(content="Supervisor created the agent plan")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def flight_agent(state: travelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {})
    try:
        airports = asyncio.run(get_airports())
        airlines = asyncio.run(get_airlines())
    except Exception as e:
        airports = f"Airport data unavailable: {e}"
        airlines = f"Airline data unavailable: {e}"
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


    approval_request = f"""### 📋 Draft Travel Itinerary

{result}

---
### ✋ Human in the Loop: Review & Approval Required
Please review the draft travel plan above.
- **To approve:** Reply with **"yes"** or **"approve"** to confirm and finalize your itinerary.
- **To modify:** Reply with your specific feedback or changes (e.g., budget adjustments, activities, or dates).
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content=approval_request)],
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

    if isinstance(feedback, dict):
        approved = bool(feedback.get("approved", True))
        human_feedback = str(feedback.get("feedback", "") or "")
    elif isinstance(feedback, bool):
        approved = feedback
        human_feedback = ""
    elif isinstance(feedback, str):
        cleaned = feedback.strip().lower()
        if cleaned in ["yes", "y", "approve", "approved", "ok", "looks good", "proceed", "confirmed"]:
            approved = True
            human_feedback = ""
        elif cleaned.startswith("yes") or cleaned.startswith("approve"):
            approved = True
            human_feedback = feedback
        else:
            approved = False
            human_feedback = feedback
    else:
        approved = True
        human_feedback = str(feedback) if feedback else ""

    status_msg = "✅ Itinerary approved! Finalizing your complete travel plan..." if approved else f"✏️ Feedback received: '{human_feedback}'. Revising travel plan..."

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content=status_msg)],
    }


def final_response_agent(state: travelState):
    if state.get("approved", True):
        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

Draft itinerary:
{state.get('itinerary', '')}

Budget notes:
{state.get('budget_results', '')}
"""
    else:
        prompt = f"""
The human requested changes to the draft.

Original user request:
{state.get('user_query', '')}

Draft itinerary:
{state.get('itinerary', '')}

Human feedback:
{state.get('human_feedback', '')}

Budget notes:
{state.get('budget_results', '')}
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