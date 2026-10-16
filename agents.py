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
    query = state.get("user_query", "").strip()

    # Build context summary from state
    trip_context_parts = []
    constraints = state.get("trip_constraints") or {}
    if constraints.get("destination"):
        trip_context_parts.append(f"Destination: {constraints.get('destination')}")
    if constraints.get("duration"):
        trip_context_parts.append(f"Duration: {constraints.get('duration')}")
    if constraints.get("budget"):
        trip_context_parts.append(f"Budget: {constraints.get('budget')}")
    if state.get("itinerary"):
        trip_context_parts.append(f"Itinerary Plan:\n{state.get('itinerary')[:1000]}")
    if state.get("budget_results"):
        trip_context_parts.append(f"Budget Details:\n{state.get('budget_results')[:500]}")
    if state.get("hotel_results"):
        trip_context_parts.append(f"Hotel Recommendations:\n{state.get('hotel_results')[:500]}")
    if state.get("flight_results"):
        trip_context_parts.append(f"Flight Details:\n{state.get('flight_results')[:500]}")

    trip_context_str = "\n".join(trip_context_parts)

    # Format recent conversation messages (last 6 messages)
    msgs = state.get("messages") or []
    recent_msgs_formatted = []
    for m in msgs[-6:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = str(m.content)[:300]
        recent_msgs_formatted.append(f"{role}: {content}")
    recent_msgs_str = "\n".join(recent_msgs_formatted)

    prompt = supervisor_agent_prompt(
        query=query,
        recent_messages=recent_msgs_str,
        current_trip_context=trip_context_str
    )

    try:
        raw = _llm_text(
            "Classify requests and route travel tasks. Return strict JSON only.",
            prompt
        )
        parsed = _json_from_llm(raw)
    except Exception as e:
        parsed = {
            "request_type": "out_of_scope",
            "direct_response": "I'm sorry, I couldn't understand that request. How can I help you with your travel planning?",
            "selected_agents": [],
            "trip_constraints": {},
            "reasoning": f"Parsing failed: {e}"
        }

    valid_types = {"social", "out_of_scope", "travel_new", "travel_followup", "travel_general"}
    request_type = parsed.get("request_type")
    if request_type not in valid_types:
        request_type = "out_of_scope"

    direct_response = parsed.get("direct_response", "").strip()

    # Safe fallback direct responses
    if request_type == "social" and not direct_response:
        direct_response = "Hello! I can help you plan your next trip. Where would you like to go?"
    elif request_type == "out_of_scope" and not direct_response:
        direct_response = "I'm a travel-planning assistant. I can help with destinations, flights, hotels, budgets, weather, and itineraries."

    allowed_agents = {
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    }

    selected_agents = [
        agent for agent in parsed.get("selected_agents", [])
        if agent in allowed_agents
    ]

    # For social or out_of_scope, no specialist agents may run
    if request_type in {"social", "out_of_scope"}:
        selected_agents = []

    # Merge or reset constraints
    new_constraints = parsed.get("trip_constraints", {})
    if request_type == "travel_new":
        trip_constraints = new_constraints if isinstance(new_constraints, dict) else {}
    else:
        trip_constraints = dict(constraints)
        if isinstance(new_constraints, dict):
            for k, v in new_constraints.items():
                if v:
                    trip_constraints[k] = v

    # If follow-up references a previous trip that does not exist or cannot be recovered
    if request_type == "travel_followup" and not trip_context_str and not selected_agents and not direct_response:
        direct_response = "I don't have a record of a previous trip in our conversation. Which destination or trip are you referring to?"

    update_dict = {
        "request_type": request_type,
        "direct_response": direct_response,
        "selected_agents": selected_agents,
        "trip_constraints": trip_constraints,
        "supervisor_reasoning": parsed.get("reasoning", ""),
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

    if direct_response:
        update_dict["messages"] = [AIMessage(content=direct_response)]

    # If user explicitly starts a new trip, clear previous trip outputs
    if request_type == "travel_new":
        update_dict["flight_results"] = ""
        update_dict["hotel_results"] = ""
        update_dict["weather_results"] = ""
        update_dict["budget_results"] = ""
        update_dict["itinerary"] = ""
        update_dict["approval_request"] = ""
        update_dict["final_response"] = ""

    return update_dict

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
### Review & Approval Required
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
    request_type = state.get("request_type", "travel_new")
    user_query = state.get("user_query", "")

    if request_type == "travel_followup":
        itinerary = state.get("itinerary", "")
        budget = state.get("budget_results", "")
        hotels = state.get("hotel_results", "")
        flights = state.get("flight_results", "")
        constraints = state.get("trip_constraints", {})

        if not (itinerary or budget or hotels or flights):
            result = "I couldn't find a previously planned trip in our conversation. Please specify the destination or trip details you would like to ask about."
        else:
            prompt = f"""Answer the user's follow-up question using the previously established trip details below.
Do NOT invent hotels, prices, flights, dates, or activities that were not in the provided trip information.

User Question:
{user_query}

Trip Constraints:
{constraints}

Planned Itinerary:
{itinerary}

Hotel Recommendations:
{hotels}

Flight Information:
{flights}

Budget Details:
{budget}

Provide a direct, accurate, and concise answer to the question using the existing information above.
"""
            result = _llm_text(
                "You are a helpful travel assistant answering a follow-up question based strictly on the existing trip plan.",
                prompt,
            )

    elif request_type == "travel_general":
        weather = state.get("weather_results", "")
        flights = state.get("flight_results", "")
        prompt = f"""Answer the user's travel question directly, practically, and concisely.
Do NOT generate a full day-by-day itinerary unless explicitly asked.

User Question:
{user_query}

Weather / MCP Data (if available):
{weather}

Flight / MCP Data (if available):
{flights}

Provide helpful, accurate travel advice.
"""
        result = _llm_text(
            "You are a helpful travel planning specialist answering an independent travel question.",
            prompt,
        )

    elif state.get("approved", True):
        prompt = f"""The human approved this draft itinerary.

Produce the final polished travel plan.

Draft itinerary:
{state.get('itinerary', '')}

Budget notes:
{state.get('budget_results', '')}
"""
        result = _llm_text(
            "You produce final user-ready travel plans.",
            prompt,
        )

    else:
        prompt = f"""The human requested changes to the draft.

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