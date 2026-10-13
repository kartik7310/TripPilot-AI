import streamlit as st
import os
import uuid
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import compiled LangGraph agent from existing backend logic
try:
    from main import app as travel_agent_app
    BACKEND_LOADED = True
    BACKEND_ERROR = None
except Exception as err:
    travel_agent_app = None
    BACKEND_LOADED = False
    BACKEND_ERROR = str(err)


# Page Configuration
st.set_page_config(
    page_title="Voyage | Travel Planner",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="auto"
)

# Theme-Consistent Responsive CSS
st.markdown("""
<style>
    /* Main Layout Responsiveness */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    
    /* Clean Subheaders & Captions */
    .sidebar-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
    }

    /* Responsive adjustments for mobile */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"trip_{uuid.uuid4().hex[:8]}"

if "sample_prompt" not in st.session_state:
    st.session_state.sample_prompt = ""


# Sidebar Configuration
with st.sidebar:
    st.title("🧭 Voyage")
    st.caption("Personalized Travel Planner")
    
    st.markdown("---")
    
    # Session Management
    st.markdown("<div class='sidebar-label'>Current Session</div>", unsafe_allow_html=True)
    custom_thread_id = st.text_input(
        "Session ID",
        value=st.session_state.thread_id,
        help="Saves your trip conversation memory."
    )
    if custom_thread_id != st.session_state.thread_id:
        st.session_state.thread_id = custom_thread_id

    if st.button("New Trip Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"trip_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("---")
    
    # Quick Inspiration
    st.markdown("<div class='sidebar-label'>Trip Inspiration</div>", unsafe_allow_html=True)
    sample_queries = [
        "5-day trip to Paris for a couple",
        "Weekend in New York City on a budget",
        "7 days in Tokyo exploring food & sights",
        "4-day beach getaway to Bali"
    ]
    
    for sq in sample_queries:
        if st.button(sq, key=f"sample_{sq}", use_container_width=True):
            st.session_state.sample_prompt = sq
            st.rerun()

    # Backend load error footer
    if not BACKEND_LOADED:
        st.markdown("---")
        st.error(f"Backend Warning: {BACKEND_ERROR}")


# Main Page Header
st.title("Plan Your Next Journey")
st.caption("Get live flight schedules, curated hotel options, and a custom day-by-day itinerary.")

st.markdown("---")

# Backend error banner
if not BACKEND_LOADED:
    st.error(
        f"**Unable to connect to travel service.**\n\n"
        f"Error details: `{BACKEND_ERROR}`\n\n"
        "Please check your `.env` configuration file and try again."
    )
    st.stop()


# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render raw search data breakdown
        if msg.get("details"):
            details = msg["details"]
            with st.expander("View search data & source details"):
                tab1, tab2, tab3, tab4 = st.tabs(["Flight Availability", "Hotel Listings", "Weather Forecast", "Draft Itinerary"])
                
                with tab1:
                    st.markdown("**Flight Search Output**")
                    st.code(details.get("flight_results", "No flight data available."), language="markdown")
                    
                with tab2:
                    st.markdown("**Hotel Search Output**")
                    st.code(details.get("hotel_results", "No hotel data available."), language="markdown")
                    
                with tab3:
                    st.markdown("**Weather Search Output**")
                    st.markdown(details.get("weather_results", "No weather data available."))

                with tab4:
                    st.markdown("**Itinerary Draft**")
                    st.markdown(details.get("itinerary", "No itinerary draft available."))


# Handle User Input
user_query = st.chat_input("Where would you like to go? (e.g., 5 days in Paris next month)")

# Check if sample prompt button was clicked
if st.session_state.sample_prompt:
    user_query = st.session_state.sample_prompt
    st.session_state.sample_prompt = ""


if user_query:
    user_query = user_query.strip()
    
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        final_text = ""
        agent_details = {
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        }
        
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id
            }
        }
        
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        }

        try:
            with st.status("Searching flights, hotels, and building your plan...", expanded=True) as status_box:
                status_box.write("Searching live flight options...")
                
                # Stream node updates from LangGraph backend
                for step in travel_agent_app.stream(initial_state, config=config, stream_mode="updates"):
                    if "flight_agent" in step:
                        agent_details["flight_results"] = step["flight_agent"].get("flight_results", "")
                        status_box.write("Flight options found. Searching top hotels & stays...")
                        
                    elif "hotel_agent" in step:
                        agent_details["hotel_results"] = step["hotel_agent"].get("hotel_results", "")
                        status_box.write("Hotels found. Checking weather forecast...")

                    elif "weather_agent" in step:
                        agent_details["weather_results"] = step["weather_agent"].get("weather_results", "")
                        status_box.write("Weather info fetched. Structuring day-by-day itinerary...")
                        
                    elif "itinerary_agent" in step:
                        agent_details["itinerary"] = step["itinerary_agent"].get("itinerary", "")
                        final_msgs = step["itinerary_agent"].get("messages", [])
                        if final_msgs:
                            final_text = final_msgs[-1].content
                        agent_details["llm_calls"] = step["itinerary_agent"].get("llm_calls", 0)

                status_box.update(label="Travel plan ready", state="complete", expanded=False)

            if final_text:
                st.markdown(final_text)
                
                # Raw search data expander
                with st.expander("View search data & source details"):
                    tab1, tab2, tab3, tab4 = st.tabs(["Flight Availability", "Hotel Listings", "Weather Forecast", "Draft Itinerary"])
                    
                    with tab1:
                        st.markdown("**Flight Search Output**")
                        st.code(agent_details["flight_results"] or "No flight data found.", language="markdown")
                        
                    with tab2:
                        st.markdown("**Hotel Search Output**")
                        st.code(agent_details["hotel_results"] or "No hotel data found.", language="markdown")

                    with tab3:
                        st.markdown("**Weather Search Output**")
                        st.markdown(agent_details["weather_results"] or "No weather data found.")
                        
                    with tab4:
                        st.markdown("**Itinerary Draft**")
                        st.markdown(agent_details["itinerary"] or "No itinerary draft available.")

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "details": agent_details
                })
            else:
                st.error("Completed without returning a travel response.")

        except Exception as e:
            st.error(f"An error occurred while generating your travel plan: {str(e)}")
