import streamlit as st
import os
import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import compiled LangGraph agent from existing backend logic
try:
    from graph import app as travel_agent_app
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


# LangGraph thread configuration
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# Check if graph is paused at an interrupt (e.g. human approval)
is_interrupted = False
if BACKEND_LOADED and travel_agent_app:
    try:
        curr_state = travel_agent_app.get_state(config)
        is_interrupted = bool(curr_state.next)
    except Exception:
        is_interrupted = False

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render raw search data breakdown
        if msg.get("details"):
            details = msg["details"]
            with st.expander("View search data & source details"):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Flight Availability", "Hotel Listings", "Weather Forecast", "Budget Analysis", "Draft Itinerary"])
                
                with tab1:
                    st.markdown("**Flight Search Output**")
                    st.code(details.get("flight_results") or "No flight data available.", language="markdown")
                    
                with tab2:
                    st.markdown("**Hotel Search Output**")
                    st.code(details.get("hotel_results") or "No hotel data available.", language="markdown")
                    
                with tab3:
                    st.markdown("**Weather Search Output**")
                    st.markdown(details.get("weather_results") or "No weather data available.")

                with tab4:
                    st.markdown("**Budget Analysis**")
                    st.markdown(details.get("budget_results") or "No budget data available.")

                with tab5:
                    st.markdown("**Itinerary Draft**")
                    st.markdown(details.get("itinerary") or "No itinerary draft available.")

# Interactive Human-in-the-Loop Review Box
if is_interrupted:
    with st.container():
        st.info("✋ **Human in the Loop: Review Required**\n\nThe draft itinerary has been generated above. Please review it and either approve to finalize or request adjustments:")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("✅ Approve Itinerary", type="primary", use_container_width=True, key="btn_approve_itin"):
                st.session_state.pending_action = {"approved": True, "feedback": ""}
                st.rerun()
        with col2:
            with st.expander("✏️ Request Revisions / Add Feedback", expanded=True):
                user_feedback = st.text_input("Enter change requests:", placeholder="e.g., Change duration to 4 days, adjust budget...", key="feedback_input")
                if st.button("Submit Revisions", use_container_width=True, key="btn_submit_feedback"):
                    if user_feedback.strip():
                        st.session_state.pending_action = {"approved": False, "feedback": user_feedback.strip()}
                        st.rerun()
                    else:
                        st.warning("Please provide your feedback before submitting.")

# Handle User Input
chat_placeholder = "Enter approval ('yes') or feedback/changes..." if is_interrupted else "Where would you like to go? (e.g., 5 days in Paris next month)"
user_query = st.chat_input(chat_placeholder)

# Check if sample prompt button was clicked
if st.session_state.sample_prompt:
    user_query = st.session_state.sample_prompt
    st.session_state.sample_prompt = ""

# Check for pending approval button action
pending_action = st.session_state.pop("pending_action", None)

if user_query or pending_action:
    if pending_action:
        if pending_action.get("approved"):
            display_user_msg = "✅ Approved draft itinerary"
        else:
            display_user_msg = f"✏️ Revision requested: {pending_action.get('feedback')}"
        stream_input = Command(resume=pending_action)
    elif is_interrupted:
        display_user_msg = user_query.strip()
        stream_input = Command(resume=display_user_msg)
    else:
        display_user_msg = user_query.strip()
        stream_input = {
            "messages": [HumanMessage(content=display_user_msg)],
            "user_id": st.session_state.thread_id,
            "user_query": display_user_msg,
            "trip_constraints": {},
            "selected_agents": [],
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "human_feedback": "",
            "approved": False,
            "final_response": "",
            "llm_calls": 0,
        }
    
    # Save user message
    st.session_state.messages.append({"role": "user", "content": display_user_msg})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(display_user_msg)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        final_text = ""
        agent_details = {
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "llm_calls": 0
        }
        
        status_title = "Processing your feedback..." if (pending_action or is_interrupted) else "Planning your trip across specialist agents..."
        
        try:
            with st.status(status_title, expanded=True) as status_box:
                for step in travel_agent_app.stream(stream_input, config=config, stream_mode="updates"):
                    if "supervisor" in step:
                        status_box.write("🧠 Supervisor analyzed request and routed agents...")
                        
                    elif "flight_agent" in step:
                        agent_details["flight_results"] = step["flight_agent"].get("flight_results", "")
                        status_box.write("✈️ Flight options gathered. Searching top stays...")
                        
                    elif "hotel_agent" in step:
                        agent_details["hotel_results"] = step["hotel_agent"].get("hotel_results", "")
                        status_box.write("🏨 Hotels found. Checking destination weather...")

                    elif "weather_agent" in step:
                        agent_details["weather_results"] = step["weather_agent"].get("weather_results", "")
                        status_box.write("☀️ Weather forecast fetched. Analyzing budget feasibility...")

                    elif "budget_agent" in step:
                        agent_details["budget_results"] = step["budget_agent"].get("budget_results", "")
                        status_box.write("💰 Budget analyzed. Creating draft itinerary...")
                        
                    elif "itinerary_agent" in step:
                        agent_details["itinerary"] = step["itinerary_agent"].get("itinerary", "")
                        final_msgs = step["itinerary_agent"].get("messages", [])
                        if final_msgs:
                            final_text = final_msgs[-1].content
                        agent_details["llm_calls"] = step["itinerary_agent"].get("llm_calls", 0)
                        status_box.write("📋 Draft itinerary ready for human review...")

                    elif "human_approval" in step:
                        status_box.write("👤 Human feedback processed...")

                    elif "final_response" in step:
                        final_msgs = step["final_response"].get("messages", [])
                        if final_msgs:
                            final_text = final_msgs[-1].content
                        agent_details["llm_calls"] = step["final_response"].get("llm_calls", 0)
                        status_box.write("✨ Final travel plan generated!")

                # Check if graph paused for approval or finished
                post_state = travel_agent_app.get_state(config)
                if post_state.next:
                    status_box.update(label="📋 Draft ready — waiting for your approval", state="complete", expanded=False)
                else:
                    status_box.update(label="✨ Travel plan completed", state="complete", expanded=False)

            if final_text:
                st.markdown(final_text)
                
                # Raw search data expander
                with st.expander("View search data & source details"):
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Flight Availability", "Hotel Listings", "Weather Forecast", "Budget Analysis", "Draft Itinerary"])
                    
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
                        st.markdown("**Budget Analysis**")
                        st.markdown(agent_details["budget_results"] or "No budget data found.")
                        
                    with tab5:
                        st.markdown("**Itinerary Draft**")
                        st.markdown(agent_details["itinerary"] or "No itinerary draft available.")

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "details": agent_details
                })
                st.rerun()
            else:
                st.error("Completed without returning a travel response.")

        except Exception as e:
            st.error(f"An error occurred while generating your travel plan: {str(e)}")
