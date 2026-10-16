
from typing import TypedDict, Literal, Annotated, Any
from langchain_core.messages import AnyMessage
import operator

class travelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_id: str
    user_query: str
    request_type: Literal[
        "social",
        "travel_new",
        "travel_followup",
        "travel_general",
        "out_of_scope",
    ]
    direct_response: str
    trip_constraints: dict[str, Any]
    selected_agents: list[str]
    supervisor_reasoning: str
   
    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results:str
    itinerary: str

    approval_request:str 
    human_feedback:str
    approved:bool
    final_response:str

    llm_calls: int
    