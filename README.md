# 🧭 Voyage: Multi-Agent Travel Planner with LangGraph & MCP

[![Python Version](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangGraph-1.2%2B-green.svg)](https://github.com/langchain-ai/langgraph)
[![Protocol](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-purple.svg)](https://modelcontextprotocol.io/)
[![Database](https://img.shields.io/badge/Neon-PostgreSQL-teal.svg)](https://neon.tech/)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

**Voyage** is an autonomous, production-grade multi-agent travel planning system built with **LangGraph**, **LangChain**, and the **Model Context Protocol (MCP)**. It coordinates specialist agents to research live flight schedules, top-rated hotels, destination weather forecasts, and budget constraints to synthesize day-by-day itineraries—featuring interactive **Human-in-the-Loop (HITL)** approval and **context-aware follow-up memory**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Specialist Agents](#-specialist-agents)
- [Core Features](#-core-features)
- [Security & Input Guardrails](#-security--input-guardrails)
- [Edge Cases Handled](#-edge-cases-handled)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Setup & Installation Guide](#-setup--installation-guide)
- [How to Run](#-how-to-run)
- [Automated Testing](#-automated-testing)

---

## 📖 Overview

Planning a trip involves juggling multiple disparate sources: flight availability, lodging, regional climate, local activities, and financial constraints. Voyage automates this end-to-end using a team of specialized AI agents coordinated by an intelligent supervisor.

Instead of generating static, unchangeable travel suggestions, Voyage:
1. **Validates User Input:** Rejects unrelated requests and responds directly to conversational pleasantries.
2. **Collects Real-Time Data:** Queries live MCP tools for flights (AviationStack), accommodations (Tavily), and weather (OpenWeatherMap).
3. **Drafts & Pauses for Approval:** Generates a structured draft plan and pauses execution via LangGraph's native `interrupt()`.
4. **Remembers Trip Context:** Allows natural follow-up questions (*"What was the budget for that trip?"* or *"Change the duration to 5 days"*) without forgetting previous details or fabricating random destinations.

---

## 🏗️ System Architecture & Workflow

The system is orchestrated as a state machine using LangGraph's `StateGraph`.

```mermaid
flowchart TD
    START([🚀 User Input]) --> Supervisor[🧠 supervisor_agent]
    
    Supervisor -->|social / out_of_scope| DirectReply[💬 Direct Response]
    DirectReply --> END([🏁 END])
    
    Supervisor -->|travel_followup without new search| FinalResponse[✨ final_response_agent]
    Supervisor -->|travel_new / specialists needed| RouteSupervisor{route_from_supervisor}
    
    RouteSupervisor -->|flight_agent| Flight[✈️ flight_agent]
    RouteSupervisor -->|hotel_agent| Hotel[🏨 hotel_agent]
    RouteSupervisor -->|weather_agent| Weather[☀️ weather_agent]
    RouteSupervisor -->|budget_agent| Budget[💰 budget_agent]
    RouteSupervisor -->|itinerary_agent| Itinerary[📋 itinerary_agent]
    
    Flight & Hotel & Weather & Budget --> RouteAgent{route_after_agent}
    RouteAgent -->|Next selected agent| Hotel
    RouteAgent -->|Next selected agent| Weather
    RouteAgent -->|Next selected agent| Budget
    RouteAgent -->|itinerary_agent in selected| Itinerary
    RouteAgent -->|itinerary_agent NOT selected| FinalResponse
    
    Itinerary --> Approval[✋ human_approval_agent<br/><i>pauses via interrupt</i>]
    Approval -->|Command resume| FinalResponse
    FinalResponse --> END
```

### Execution Lifecycle:
1. **User Turn:** User inputs a prompt via the Streamlit UI or Terminal CLI.
2. **Supervisor Guardrail:** Classifies the request into one of 5 distinct categories:
   - `social` $\to$ Returns a conversational reply and halts.
   - `out_of_scope` $\to$ Politely declines non-travel tasks and halts.
   - `travel_new` $\to$ Resets prior trip artifacts and routes to specialist agents.
   - `travel_followup` $\to$ Reuses active trip state; routes to `final_response` or selectively triggers re-planning.
   - `travel_general` $\to$ Queries only relevant information (e.g. weather) without generating an itinerary.
3. **Specialist Research:** Agents execute asynchronously or via MCP adapters.
4. **Itinerary Synthesis:** `itinerary_agent` drafts the day-by-day plan with budget integration.
5. **Human-in-the-Loop Interruption:** The graph pauses using `interrupt()` and awaits user review.
6. **Execution Resume:** Resumed with `Command(resume=...)` carrying user approval or revision instructions.
7. **Final Plan Delivery:** `final_response_agent` outputs the polished plan or answers follow-up inquiries.

---

## 🤖 Specialist Agents

| Agent | Responsibility | Data Source / Integration |
|---|---|---|
| **Supervisor Agent** | Classifies intent, enforces guardrails, extracts constraints, routes workflow | Groq LLM + Intent Guardrail Schema |
| **Flight Agent** | Identifies departure/arrival airports, airlines, duration, fare estimates | AviationStack MCP Server (`list_airports`, `list_airlines`) |
| **Hotel Agent** | Finds curated accommodations, neighborhood vibes, price tiers | Tavily Search MCP (`tavily_search`) |
| **Weather Agent** | Fetches destination conditions, forecasts, packing requirements | OpenWeatherMap MCP (`get_current_weather`, `get_forecast`) |
| **Budget Agent** | Categorizes costs, evaluates feasibility, offers cost-saving tips | Financial Analysis Prompt + LLM |
| **Itinerary Agent** | Combines flights, hotels, weather, and budget into a cohesive draft | Multi-Agent Synthesis Prompt |
| **Human Approval Agent** | Intercepts execution for user sign-off or modifications | LangGraph `interrupt()` |
| **Final Response Agent** | Delivers final polished itinerary or answers follow-up inquiries | Context-Grounded LLM Prompt |

---

## ⚡ Core Features

- **Multi-Agent Orchestration:** Complex logic split into decoupled nodes with clear state contracts.
- **Model Context Protocol (MCP) Native:** Connects to live external services using standardized protocol clients (`MultiServerMCPClient`).
- **Human-in-the-Loop (HITL) Workflow:** Pauses graph execution cleanly at the approval step and resumes without losing context.
- **State Persistence:** Backed by **Neon Serverless PostgreSQL** via `PostgresSaver`, with an in-memory `MemorySaver` fallback.
- **Non-Destructive Session State:** Subsequent chat turns pass delta inputs (`messages`, `user_query`), protecting previously planned trip context from being overwritten.
- **Context-Aware Follow-Up Routing:** Answers questions about past trip details (e.g., *"What was the budget?"* or *"Which hotel was recommended?"*) directly from state without re-running web searches or creating new trips.
- **Dual User Interface:** Full-featured **Streamlit Web Dashboard** and interactive **Terminal CLI**.

---

## 🛡️ Security & Input Guardrails

Voyage treats all user instructions as untrusted:

1. **Domain Boundary Enforcement:** Unrelated requests (e.g., coding, writing essays, math, or prompt injections) are intercepted by the supervisor and declined before reaching specialist agents.
2. **Schema & Agent Name Validation:** Supervisor output must conform to a strict JSON schema. Unknown or hallucinated agent names are filtered out against an allowed whitelist.
3. **Fail-Closed Fallbacks:** If the supervisor generates malformed JSON or encounters an unexpected classification, the system safely falls back to a polite assistance message rather than executing undefined routes.
4. **Credential Isolation:** API keys and database strings are managed through `.env` and kept strictly out of Git version control.

---

## 🧩 Edge Cases Handled

- **Follow-up with Missing History:** If a user in a fresh session asks *"What was the budget for that trip?"*, the system detects the absence of active trip context and asks for clarification instead of inventing a destination.
- **Trip Duration Adjustments:** When asked to *"Change duration to 5 days"*, the supervisor updates `trip_constraints` and re-runs only `itinerary_agent`, preserving already fetched hotel and flight data.
- **General Inquiries Without Itineraries:** Asking *"What should I pack for Shimla in December?"* routes only to `weather_agent` $\to$ `final_response_agent`, answering the question without producing an unneeded multi-day schedule.
- **Robust Feedback Parsing:** User responses on resumption can be structured dictionaries (`{"approved": True}`), booleans, or natural language strings (`"yes"`, `"approve"`, or custom text).
- **Database Connection Resiliency:** If PostgreSQL is unreachable, the system automatically falls back to in-memory checkpointing so that graph execution never crashes.

---

## 💻 Tech Stack

- **Core:** Python 3.14+
- **Agent Orchestration:** LangGraph (v1.2+), LangChain Core (v1.4+)
- **LLM Provider:** Groq (`openai/gpt-oss-20b`, `llama-3.3-70b-versatile`)
- **Tool Protocol:** Model Context Protocol (`langchain-mcp-adapters`)
- **External APIs:**
  - Tavily Search API (Web research & hotels)
  - AviationStack API (Airports & airline data)
  - OpenWeatherMap API (Current weather & forecasts)
- **State Persistence:** Neon PostgreSQL, `langgraph-checkpoint-postgres`, `psycopg3`
- **Web Application:** Streamlit (v1.63+)

---

## 📁 Project Directory Structure

```text
travel-plan-agent-mcp/
├── agents.py                     # Agent node implementations & LLM invocations
├── app.py                        # Streamlit web interface with interactive HITL UI
├── config.py                     # Environment variables & LLM client configuration
├── graph.py                      # LangGraph definition, conditional edges, & checkpointer
├── main.py                       # CLI terminal entry point with streaming & resume
├── mcp_client.py                 # Multi-server MCP clients (Tavily, Weather, Aviation)
├── prompts.py                    # Structured supervisor prompt & classification schema
├── state.py                      # TypedDict definition of travelState
├── weather-mcp-server.py         # Standalone OpenWeatherMap MCP server script
├── pyproject.toml                # Project metadata & dependencies
├── .env.example                  # Environment variable template
├── .gitignore                    # Git exclusion rules (.env, venv, caches)
├── test_followup_and_guardrails.py # Comprehensive unit test suite (12 test cases)
├── test_aviation.py              # AviationStack MCP test script
├── test_weather_mcp_server.py    # Weather MCP test script
└── README.md                     # Project documentation
```

---

## ⚙️ Setup & Installation Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.12 or 3.14 recommended)
- A **Groq** API Key ([console.groq.com](https://console.groq.com/))
- A **Tavily** API Key ([tavily.com](https://tavily.com/))
- An **OpenWeatherMap** API Key ([openweathermap.org](https://openweathermap.org/api))
- An **AviationStack** API Key ([aviationstack.com](https://aviationstack.com/))
- *(Optional)* A **Neon PostgreSQL** Connection String ([neon.tech](https://neon.tech/))

### 2. Clone the Repository
```bash
git clone https://github.com/kartik7310/travel-agent-mcp.git
cd travel-agent-mcp
```

### 3. Set Up Virtual Environment

**Using `uv` (Fastest):**
```bash
uv venv
.\.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On macOS/Linux
uv pip install -r pyproject.toml
```

**Using standard `pip`:**
```bash
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
source .venv/bin/activate  # On macOS/Linux
pip install -e .
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your credentials in `.env`:
```ini
GROQ_API_KEY=gsk_your_groq_api_key
MODEL=openai/gpt-oss-20b

AVIATIONSTACK_API_KEY=your_aviationstack_key
TAVILY_API_KEY=tvly-your_tavily_key
OPENWEATHER_API_KEY=your_openweather_key

DATABASE_URL=postgresql://user:password@ep-sample.neon.tech/neondb?sslmode=require
```

---

## 🚀 How to Run

### Option 1: Streamlit Web UI (Recommended)
Launch the responsive web app:
```bash
streamlit run app.py
```
Or with `uv`:
```bash
uv run streamlit run app.py
```
Open **http://localhost:8501** in your browser.
- Enter a query like *"Plan a 4-day trip to Paris for a couple"*.
- Review live search results under the collapsible inspection tabs (Flights, Hotels, Weather, Budget).
- Approve or request revisions directly via the **Review Required** interactive panel.

### Option 2: Terminal CLI
Run the conversational terminal loop:
```bash
python main.py
```
- Provide your travel request at the `You: ` prompt.
- When an itinerary is drafted, approve by typing `yes` or submit feedback at the `[Human in the Loop]` prompt.

---

## 🧪 Automated Testing

Voyage includes a test suite covering the 12 core guardrail, routing, and memory scenarios using deterministic mock LLMs:

```bash
python -m unittest test_followup_and_guardrails.py -v
```

### Test Coverage:
1. **Greeting Guardrail:** `Hi` $\to$ Direct reply, 0 specialist agents invoked.
2. **Out-of-Scope Guardrail:** `Write an essay about pollution` $\to$ Polite rejection, 0 specialist agents invoked.
3. **Trip Planning:** `Plan 3-day Manali trip` $\to$ Specialist agents + itinerary synthesis + HITL interrupt.
4. **Budget Follow-Up:** `What was the budget for that trip?` $\to$ Read from context without re-planning.
5. **Hotel Follow-Up:** `What hotels did you recommend?` $\to$ Read from context without re-planning.
6. **Trip Duration Modification:** `Change duration to 5 days` $\to$ Updates constraints and regenerates draft.
7. **Missing History Handling:** Follow-up in fresh session $\to$ Clarification question, no hallucinations.
8. **Independent Travel Query:** `What should I pack for Shimla?` $\to$ Weather agent only, no itinerary draft.
9. **Thread State Continuity:** Sequential turns accumulate context properly on the same `thread_id`.
10. **Thread Isolation:** Distinct `thread_id` sessions maintain strict state separation.
11. **Malformed JSON Recovery:** Invalid LLM outputs fail-closed to safe responses.
12. **Anti-Fabrication:** Missing trip information returns an honest missing-record notice.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
