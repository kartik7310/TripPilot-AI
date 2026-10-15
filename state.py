from narwhals.selectors import boolean
from typing import TypedDict,Annotated,Any
from langchain_core.messages import AnyMessage
import operator
class travelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_id:str
    user_query: str

    trip_constraints:dict[str,Any]
    selected_agents:list[str]
    supervisor_reasoning:str

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
    