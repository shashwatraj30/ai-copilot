from fastapi import FastAPI
from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

@app.get("/research")       # ← new endpoint goes here
def research(topic: str):
    prompt = f"""You are a research assistant. Research the topic: "{topic}"
    
Respond ONLY with a JSON object in this exact format, nothing else:
{{
  "topic": "{topic}",
  "summary": "2-3 sentence summary",
  "key_points": ["point 1", "point 2", "point 3"],
  "use_cases": ["use case 1", "use case 2"]
}}"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = response.choices[0].message.content
    return json.loads(raw)