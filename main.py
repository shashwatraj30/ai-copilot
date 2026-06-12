from fastapi import FastAPI
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient
from newspaper import Article
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from pydantic import BaseModel
import os
import json
conversation_store = {}

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

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


@app.get("/compare")
def compare(topic_a: str, topic_b: str):
    # Search both topics
    results_a = tavily.search(query=topic_a, max_results=2)
    results_b = tavily.search(query=topic_b, max_results=2)
    
    context_a = "\n".join([r["content"] for r in results_a["results"]])
    context_b = "\n".join([r["content"] for r in results_b["results"]])

    prompt = f"""You are a research assistant. Compare these two topics based on the search results below.

Topic A: {topic_a}
Search Results A: {context_a}

Topic B: {topic_b}
Search Results B: {context_b}

Respond ONLY with a JSON object in this exact format, nothing else:
{{
  "topic_a": "{topic_a}",
  "topic_b": "{topic_b}",
  "summary": "2-3 sentence overall comparison",
  "similarities": ["similarity 1", "similarity 2"],
  "differences": ["difference 1", "difference 2", "difference 3"],
  "verdict": "one sentence on which is better and for what use case"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

@app.get("/summarize")
def summarize(url: str):
    # Fetch article content
    article = Article(url)
    article.download()
    article.parse()
    
    prompt = f"""You are a research assistant. Summarize the following article content.

Title: {article.title}
Content: {article.text[:3000]}

Respond ONLY with a JSON object in this exact format, nothing else:
{{
  "url": "{url}",
  "title": "{article.title}",
  "summary": "2-3 sentence summary",
  "key_points": ["point 1", "point 2", "point 3"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

@app.get("/trending")
def trending(category: str):
    results = tavily.search(query=f"trending {category} news today", max_results=5)
    context = "\n".join([r["content"] for r in results["results"]])

    prompt = f"""You are a research assistant. Based on these search results, identify trending topics in {category}.

Search Results:
{context}

Respond ONLY with a JSON object in this exact format, nothing else:
{{
  "category": "{category}",
  "trending_topics": ["topic 1", "topic 2", "topic 3"],
  "summaries": ["one sentence summary 1", "one sentence summary 2", "one sentence summary 3"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id
    
    # Create new session if doesn't exist
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    # Append user message to history
    conversation_store[session_id].append({
        "role": "user",
        "content": request.message
    })
    
    # Send full history to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_store[session_id][-10:]
    )
    
    assistant_reply = response.choices[0].message.content
    
    # Append assistant reply to history
    conversation_store[session_id].append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    return {
        "session_id": session_id,
        "reply": assistant_reply,
        "history_length": len(conversation_store[session_id])
    }

@app.get("/fact-check")
def fact_check(claim: str):
    # Search for evidence
    search_results = tavily.search(query=claim, max_results=3)
    context = "\n".join([r["content"] for r in search_results["results"]])

    prompt = f"""You are a fact-checking assistant. Analyze the following claim based on the search results.

Claim: "{claim}"

Search Results:
{context}

Respond ONLY with a JSON object in this exact format, nothing else:
{{
  "claim": "{claim}",
  "verdict": "True/False/Partially True",
  "confidence": "High/Medium/Low",
  "reasoning": "2-3 sentence explanation",
  "sources_support": true/false
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

#save research to supabase
class SaveRequest(BaseModel):
    topic: str
    result: dict
    user_id: str

@app.post("/save-research")
def save_research(request: SaveRequest):
    data = supabase_client.table("saved_research").insert({
        "topic": request.topic,
        "result": request.result
        "user_id": request.user_id
    }).execute()
    
    return {"message": "Research saved successfully", "data": data.data}
#get research from supabase
@app.get("/get-research")
def get_research(user_id: str):
    data = supabase_client.table("saved_research").select("*").eq("user_id", user_id).execute()
    return {"saved_research": data.data}