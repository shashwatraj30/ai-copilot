from fastapi import FastAPI, Request, HTTPException,UploadFile, File
from groq import Groq
from dotenv import load_dotenv
from tavily import TavilyClient
from newspaper import Article
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sentence_transformers import SentenceTransformer
import fitz
import os
import json

conversation_store = {}
limiter = Limiter(key_func=get_remote_address)

load_dotenv()

app = FastAPI()
app.state.limiter = limiter
from fastapi.responses import JSONResponse

async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
        headers={"Access-Control-Allow-Origin": "*"}
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

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
@limiter.limit("50/minute")
def research(request: Request, topic: str):
    if not topic or len(topic.strip()) == 0:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    if len(topic) > 500:
        raise HTTPException(status_code=400, detail="Topic too long, max 500 characters")
    search_results = tavily.search(query=topic, max_results=3)
    context = "\n".join([r["content"] for r in search_results["results"]])

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
    return json.loads(response.choices[0].message.content)


@app.get("/compare")
@limiter.limit("50/minute")
def compare(request: Request, topic_a: str, topic_b: str):
    if not topic_a or not topic_b:
        raise HTTPException(status_code=400, detail="Both topics required")
    if len(topic_a) > 500 or len(topic_b) > 500:
        raise HTTPException(status_code=400, detail="Topic too long, max 500 characters")
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
    return json.loads(response.choices[0].message.content)


@app.get("/summarize")
@limiter.limit("50/minute")
def summarize(request: Request, url: str):
    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Valid URL required")
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
    return json.loads(response.choices[0].message.content)


@app.get("/trending")
@limiter.limit("50/minute")
def trending(request: Request, category: str):
    if not category or len(category.strip()) == 0:
        raise HTTPException(status_code=400, detail="Category cannot be empty")
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
    return json.loads(response.choices[0].message.content)


class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
@limiter.limit("50/minute")
def chat(request: Request, body: ChatRequest):
    if not body.message or len(body.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(body.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long, max 2000 characters")
    session_id = body.session_id

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append({
        "role": "user",
        "content": body.message
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_store[session_id][-10:]
    )

    assistant_reply = response.choices[0].message.content

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
@limiter.limit("50/minute")
def fact_check(request: Request, claim: str):
    if not claim or len(claim.strip()) == 0:
        raise HTTPException(status_code=400, detail="Claim cannot be empty")
    if len(claim) > 1000:
        raise HTTPException(status_code=400, detail="Claim too long, max 1000 characters")
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
    return json.loads(response.choices[0].message.content)


class SaveRequest(BaseModel):
    topic: str
    result: dict
    user_id: str

@app.post("/save-research")
@limiter.limit("50/minute")
def save_research(request: Request, body: SaveRequest):
    if not body.topic or len(body.topic.strip()) == 0:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    data = supabase_client.table("saved_research").insert({
        "topic": body.topic,
        "result": body.result,
        "user_id": body.user_id
    }).execute()
    return {"message": "Research saved successfully", "data": data.data}


@app.get("/get-research")
def get_research(user_id: str):
    data = supabase_client.table("saved_research").select("*").eq("user_id", user_id).execute()
    return {"saved_research": data.data}


class AgentRequest(BaseModel):
    session_id: str
    query: str

@app.post("/agent")
@limiter.limit("10/minute")
def agent(request: Request, body: AgentRequest):
    if not body.query or len(body.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(body.query) > 1000:
        raise HTTPException(status_code=400, detail="Query too long, max 1000 characters")
    session_id = body.session_id
    query = body.query

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    plan_prompt = f"""You are an advanced AI research agent. A user has asked: "{query}"

Your job is to think critically and decide what steps are needed to answer this properly.

Available tools:
- search(query): searches the web for current information
- fact_check(claim): verifies if a claim is true
- compare(a, b): compares two topics
- reason(question): uses pure reasoning without search

Respond ONLY with a JSON object:
{{
  "thinking": "your reasoning about what this query needs",
  "steps": [
    {{"tool": "search", "input": "what to search"}},
    {{"tool": "fact_check", "input": "claim to verify"}},
    {{"tool": "reason", "input": "question to reason about"}}
  ]
}}

Keep steps to maximum 3. Only include steps that are truly needed."""

    plan_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": plan_prompt}]
    )

    raw_plan = plan_response.choices[0].message.content.strip()
    if raw_plan.startswith("```"):
        raw_plan = raw_plan.split("```")[1]
        if raw_plan.startswith("json"):
            raw_plan = raw_plan[4:]
    plan = json.loads(raw_plan.strip())

    tool_results = []
    for step in plan.get("steps", []):
        tool = step.get("tool")
        inp = step.get("input")

        if tool == "search":
            results = tavily.search(query=inp, max_results=3)
            context = "\n".join([r["content"] for r in results["results"]])
            tool_results.append(f"[Search: {inp}]\n{context}")

        elif tool == "fact_check":
            results = tavily.search(query=inp, max_results=2)
            context = "\n".join([r["content"] for r in results["results"]])
            tool_results.append(f"[Fact Check: {inp}]\n{context}")

        elif tool == "compare":
            parts = inp.split(" vs ")
            if len(parts) == 2:
                r_a = tavily.search(query=parts[0], max_results=2)
                r_b = tavily.search(query=parts[1], max_results=2)
                ca = "\n".join([r["content"] for r in r_a["results"]])
                cb = "\n".join([r["content"] for r in r_b["results"]])
                tool_results.append(f"[Compare: {inp}]\nA: {ca}\nB: {cb}")

        elif tool == "reason":
            tool_results.append(f"[Reasoning needed: {inp}]")

    synthesis_prompt = f"""You are an advanced AI research agent.

User query: "{query}"

Your reasoning: {plan.get('thinking')}

Tool results:
{chr(10).join(tool_results)}

Now synthesize a comprehensive, critical, well-reasoned answer.
- Don't just summarize — analyze and give your own reasoned conclusions
- Point out nuances, contradictions, or uncertainties
- Be specific and actionable
- Format clearly with sections if needed"""

    conversation_store[session_id].append({
        "role": "user",
        "content": synthesis_prompt
    })

    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_store[session_id][-10:]
    )

    final_answer = final_response.choices[0].message.content

    conversation_store[session_id].append({
        "role": "assistant",
        "content": final_answer
    })

    return {
        "query": query,
        "thinking": plan.get("thinking"),
        "steps_taken": plan.get("steps"),
        "answer": final_answer
    }


class PDFRequest(BaseModel):
    session_id: str
    instruction: str
    pdf_text: str

@app.post("/pdf-agent")
@limiter.limit("10/minute")
def pdf_agent(request: Request, body: PDFRequest):
    if not body.instruction or len(body.instruction.strip()) == 0:
        raise HTTPException(status_code=400, detail="Instruction cannot be empty")
    if not body.pdf_text or len(body.pdf_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="PDF text cannot be empty")
    session_id = body.session_id

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    prompt = f"""You are an advanced PDF analysis agent. A user has uploaded a document and given you an instruction.

Instruction: "{body.instruction}"

Document Content:
{body.pdf_text[:12000]}

Respond based exactly on what the user asked:
- If they want a summary → summarize clearly
- If they want to learn/be taught → explain like a teacher, step by step
- If they want extraction → extract exactly what they asked for
- If they want critical analysis → think deeply, find patterns, contradictions, insights
- If they want anything else → do exactly that

Be thorough, intelligent and flexible. Format your response clearly."""

    conversation_store[session_id].append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation_store[session_id][-10:]
    )

    answer = response.choices[0].message.content

    conversation_store[session_id].append({
        "role": "assistant",
        "content": answer
    })

    return {
        "answer": answer,
        "chars_processed": len(body.pdf_text[:12000])
    }

@app.delete("/delete-research/{item_id}")
@limiter.limit("50/minute")
def delete_research(request: Request, item_id: int, user_id: str):
    data = supabase_client.table("saved_research").delete().eq("id", item_id).eq("user_id", user_id).execute()
    return {"message": "Deleted successfully"}

    

@app.get("/rag-search")
def rag_search(query: str, match_count: int = 5):
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search Supabase for similar chunks
    result = supabase_client.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": match_count
    }).execute()
    
    return {
        "query": query,
        "results": result.data
    }

@app.get("/rag-query")
def rag_query(query: str, match_count: int = 5):
    # Step 1: Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Step 2: Retrieve relevant chunks
    result = supabase_client.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": match_count
    }).execute()
    
    chunks = result.data
    
    if not chunks:
        return {"query": query, "answer": "No relevant documents found.", "sources": []}
    
    # Step 3: Build context from chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}]: {chunk['content']}\n\n"
    
    # Step 4: Send to Groq with context
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
For each point you make, cite the source number like [Source 1], [Source 2] etc.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    
    # Step 5: Return answer + sources
    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "chunk_index": c["metadata"].get("chunk_index"),
                "source": c["metadata"].get("source"),
                "similarity": c["similarity"],
                "content_preview": c["content"][:200]
            }
            for c in chunks
        ]
    }


@app.post("/ingest-pdf")
@limiter.limit("10/minute")
async def ingest_pdf(request: Request, file: UploadFile = File(...), user_id: str = "anonymous"):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    
    # Read PDF bytes
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    
    # Extract text
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    if not full_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Chunk text
    words = full_text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+200])
        chunks.append(chunk)
        i += 200 - 50
    
    # Embed and store each chunk
    stored = 0
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        supabase_client.table("documents").insert({
            "user_id": user_id if user_id != "anonymous" else None,
            "content": chunk,
            "embedding": embedding,
            "metadata": {"source": file.filename, "chunk_index": idx}
        }).execute()
        stored += 1
    
    return {
        "message": "PDF ingested successfully",
        "filename": file.filename,
        "chunks_stored": stored
    }
@app.get("/my-documents")
def my_documents(user_id: str):
    result = supabase_client.table("documents")\
        .select("id, metadata, created_at")\
        .eq("user_id", user_id)\
        .execute()
    
    # Group by source filename
    files = {}
    for row in result.data:
        source = row["metadata"].get("source", "unknown")
        if source not in files:
            files[source] = {"filename": source, "chunks": 0, "created_at": row["created_at"]}
        files[source]["chunks"] += 1
    
    return {"documents": list(files.values())}
@app.delete("/delete-document")
@limiter.limit("20/minute")
def delete_document(request: Request, user_id: str, filename: str):
    result = supabase_client.table("documents")\
        .delete()\
        .eq("user_id", user_id)\
        .eq("metadata->>source", filename)\
        .execute()
    
    return {"message": f"Deleted all chunks for {filename}"}