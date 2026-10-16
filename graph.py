
import psycopg

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
)

from state import travelState
from config import DATABASE_URL


AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


ROUTE_MAP = {
    **{agent: agent for agent in AGENT_ORDER},
    "final_response": "final_response",
    END: END,
}


def _selected_agents(state: travelState) -> list[str]:
    selected = state.get("selected_agents") or []

    if not isinstance(selected, list):
        return []

    return [
        agent
        for agent in AGENT_ORDER
        if agent in selected
    ]


# Route after supervisor
def route_from_supervisor(state: travelState) -> str:
    request_type = state.get("request_type")

    # Greetings and unrelated requests stop here
    if request_type in {"social", "out_of_scope"}:
        return END

    # If supervisor produced a direct response and no agents were selected, stop here
    if state.get("direct_response") and not state.get("selected_agents"):
        return END

    selected = _selected_agents(state)

    if selected:
        return selected[0]

    # Follow-up questions or general inquiries that can be answered from existing context
    return "final_response"


# Route between specialist agents
def route_after_agent(current_agent: str):
    def route(state: travelState) -> str:
        request_type = state.get("request_type")

        if request_type in {"social", "out_of_scope"}:
            return END

        selected = _selected_agents(state)

        if current_agent not in AGENT_ORDER:
            return "final_response"

        current_index = AGENT_ORDER.index(current_agent)

        # Run the next selected agent in the defined order
        for next_agent in AGENT_ORDER[current_index + 1:]:
            if next_agent in selected:
                return next_agent

        # If itinerary_agent was explicitly selected, route to it
        if "itinerary_agent" in selected:
            return "itinerary_agent"

        # Otherwise, no itinerary is needed; route to final_response
        return "final_response"

    return route


def build_graph(checkpointer=None):
    graph = StateGraph(travelState)

    # Register nodes
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor can stop or route to a specialist / final_response
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ROUTE_MAP,
    )

    # Specialist routing
    graph.add_conditional_edges(
        "flight_agent",
        route_after_agent("flight_agent"),
        ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "hotel_agent",
        route_after_agent("hotel_agent"),
        ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "weather_agent",
        route_after_agent("weather_agent"),
        ROUTE_MAP,
    )

    graph.add_conditional_edges(
        "budget_agent",
        route_after_agent("budget_agent"),
        ROUTE_MAP,
    )

    # Preserve the existing approval and response workflow
    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_response")
    graph.add_edge("final_response", END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)

    # Optional PostgreSQL persistence
    if DATABASE_URL:
        try:
            conn = psycopg.connect(
                DATABASE_URL,
                autocommit=True,
            )

            pg_checkpointer = PostgresSaver(conn)
            pg_checkpointer.setup()

            return graph.compile(checkpointer=pg_checkpointer)

        except Exception as e:
            print(f"Warning: PostgresSaver setup failed: {e}")
            from langgraph.checkpoint.memory import MemorySaver
            return graph.compile(checkpointer=MemorySaver())

    from langgraph.checkpoint.memory import MemorySaver
    return graph.compile(checkpointer=MemorySaver())


app = build_graph()
