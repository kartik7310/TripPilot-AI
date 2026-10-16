import os
import sys
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from graph import app

# Set UTF-8 encoding for Windows terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if __name__ == "__main__":
    is_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            is_streamlit = True
    except Exception:
        pass

    if is_streamlit:
        import runpy
        app_file = os.path.join(os.path.dirname(__file__), "app.py")
        runpy.run_path(app_file, run_name="__main__")
    else:
        config = {
            "configurable": {
                "thread_id": "user_kartik"
            }
        }

        print("Travel Agent started. Type 'exit' to stop.\n")

        while True:
            # Check if there is an active interrupt waiting for human input
            current_state = app.get_state(config)
            is_interrupted = bool(current_state.next)

            prompt_label = "[Human in the Loop] Your approval/feedback (e.g. 'yes' or comments): " if is_interrupted else "You: "
            user_input = input(prompt_label).strip()

            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            if is_interrupted:
                stream_input = Command(resume=user_input)
            else:
                stream_input = {
                    "messages": [
                        HumanMessage(content=user_input)
                    ],
                    "user_id": "user_kartik",
                    "user_query": user_input,
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

            print()
            for chunk in app.stream(
                stream_input,
                config=config,
                stream_mode="messages"
            ):
                message, metadata = chunk
                if message.content:
                    print(message.content, end="\n\n", flush=True)

            print()


    
