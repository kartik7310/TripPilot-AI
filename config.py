import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

AVIATIONSTACK_API_KEY=os.getenv("AVIATIONSTACK_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
DB_URL = DATABASE_URL

def get_llm():
    return ChatGroq(model=os.getenv("MODEL"),
                    groq_api_key=GROQ_API_KEY)
