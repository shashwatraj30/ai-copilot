from fastapi import FastAPI
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient
import os
import json

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@app.get("/")
def root():
    return {"status": "alive"}

@app.get("/ask")
def ask(query: str):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": query}]
    )
    return {"answer": response.choices[0].message.content}

@app.get("/research")
def research(topic: str):
    # Step 1: Search the web
    search_results = tavily.search(query=topic, max_results=3)
    context = "\n".join([r["content"] for r in search_results["results"]])

    # Step 2: Summarize with Groq
    prompt = f"""You are a research assistant. Based on the following search results, research the topic: "{topic}"

Search Results:
{context}

Respond ONLY with a JSON object in this exact format, nothing else:
{{
  "topic": "{topic}",
  "summary": "2-3 sentence summary based on search results",
  "key_points": ["point 1", "point 2", "point 3"],
  "use_cases": ["use case 1", "use case 2"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    return json.loads(raw)