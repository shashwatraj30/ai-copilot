from fastapi import FastAPI
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient
from newspaper import Article
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