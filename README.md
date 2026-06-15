# 🔬 AI Research Copilot

An intelligent research assistant powered by Groq, Tavily, and Supabase. Built as Phase 1 of a 22-week AI + Web3 engineering roadmap.

**Live Demo:** [ai-copilot-sand.vercel.app](https://ai-copilot-sand.vercel.app)

---

## What it does

- **Research** any topic with real-time web search and AI synthesis
- **Compare** two topics side by side with similarities, differences, and verdict
- **Summarize** any URL into key points
- **Trending** topics in any category
- **Fact Check** any claim with evidence and confidence score
- **Smart Agent** — ReAct-pattern agent that plans, uses tools, and reasons before answering
- **PDF Agent** — upload any PDF and ask it to summarize, teach, extract, or critically analyze
- **Chat** with full conversation memory and context injection from research cards
- **Save** any result to your personal library with per-user data isolation
- **Sidebar** to browse, load, and delete saved results grouped by type

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI + Python 3.11 |
| AI | Groq (llama-3.3-70b-versatile) |
| Search | Tavily API |
| Database | Supabase (Postgres) |
| Auth | Supabase Auth |
| Frontend | Vanilla HTML/CSS/JS |
| Backend Deploy | Railway |
| Frontend Deploy | Vercel |

---

## Architecture

- Frontend handles auth via Supabase JS client
- Every API request is rate limited (50 req/min standard, 10 req/min for agent/PDF)
- Input validation on all endpoints
- Per-user data isolation via user_id tagging
- ReAct agent: Plan → Execute tools → Synthesize answer

---

## Features by Week

**Week 1 (Days 1-7):** Core research endpoints, Tavily integration, compare, summarize, trending, CORS setup, Railway + Vercel deployment

**Week 2 (Days 8-14):** Conversation history, fact-check endpoint, Supabase persistence, save/retrieve research, formatted output, silent context injection

**Week 3 (Days 15-21):** Supabase Auth, per-user data isolation, ReAct smart agent, PDF upload + analysis, rate limiting, input validation, saved history sidebar, mobile responsive UI

---

## Running Locally

```bash
# Clone
git clone https://github.com/shashwatraj30/ai-copilot.git
cd ai-copilot

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Fill in your keys

# Run backend
uvicorn main:app --reload
```

**Required environment variables:**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /research | Research a topic |
| GET | /compare | Compare two topics |
| GET | /summarize | Summarize a URL |
| GET | /trending | Get trending topics |
| GET | /fact-check | Fact check a claim |
| POST | /chat | Chat with memory |
| POST | /agent | ReAct smart agent |
| POST | /pdf-agent | PDF analysis |
| POST | /save-research | Save result |
| GET | /get-research | Get saved results |
| DELETE | /delete-research/{id} | Delete saved result |

---

## Roadmap

- [x] Phase 1 — AI Research Copilot
- [ ] Phase 2 — RAG Knowledge System
- [ ] Phase 3 — Multi-Agent System
- [ ] Phase 4 — Automation (n8n)
- [ ] Phase 5-8 — Web3 Integration

---

Built by [@shashwatraj30](https://github.com/shashwatraj30) as part of a compressed 22-week AI + Web3 engineering roadmap.