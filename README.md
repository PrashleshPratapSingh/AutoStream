# AutoStream AI Agent — Social-to-Lead Agentic Workflow

> **Machine Learning Intern Assignment — ServiceHive (Inflx)**

A production-grade conversational AI agent for **AutoStream**, a SaaS platform providing automated video editing tools for content creators. The agent understands user intent, answers product questions using RAG, identifies high-intent leads, and triggers backend lead capture — demonstrating a real-world GenAI agentic workflow.

---

## Table of Contents

- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [State Management Deep Dive](#-state-management-deep-dive)
- [RAG Pipeline](#-rag-pipeline)
- [WhatsApp Deployment via Webhooks](#-whatsapp-deployment-via-webhooks)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)

---

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| Google AI API Key | [Get free key](https://aistudio.google.com/apikey) |

### Installation & Run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
cp .env.example .env
# Open .env and paste your Google AI API key

# 5. Launch the agent
python main.py
```

### Example Conversation

```
You: Hi, tell me about your pricing.
AutoStream: Hey there! AutoStream offers two plans — our Basic plan at $29/month
            with 10 videos and 720p, and our Pro plan at $79/month with unlimited
            videos, 4K resolution, and AI captions. Which one interests you?

You: That sounds good, I want to try the Pro plan for my YouTube channel.
AutoStream: Great choice! I'd love to get you started. Could I get your name
            and email address to set up your account?

You: I'm Pranav, pranav@email.com
AutoStream: Awesome, you're all set! I've registered your interest:
            * Name: Pranav
            * Email: pranav@email.com
            * Platform: YouTube
            Our team will reach out shortly with your Pro plan trial details.
```

---

## System Architecture

### High-Level Overview (~200 words)

#### Why LangGraph?

I chose **LangGraph** over AutoGen because it provides a **state machine abstraction** that maps perfectly to the agentic workflow required by this assignment. Each step in the conversation pipeline — intent classification, knowledge retrieval, lead qualification, and tool execution — becomes an explicit **node** in the graph, with **conditional edges** handling routing logic. This makes the agent's decision-making process **transparent, debuggable, and extensible**, unlike a purely prompt-driven approach where routing happens implicitly inside a single LLM call.

LangGraph also provides first-class support for **typed state** that flows through the graph, making it trivial to accumulate lead information across multiple conversation turns without losing context.

#### How State is Managed

The agent uses a **typed `AgentState` dictionary** (Python `TypedDict`) that persists across graph invocations. It contains:

- **`messages`** — Full conversation history using LangGraph's `add_messages` reducer, which automatically handles message deduplication and ordering.
- **`intent`** — Classified intent of the current user message (`greeting`, `product_inquiry`, or `high_intent`).
- **`lead_info`** — A dictionary progressively accumulating the user's name, email, and platform across turns.
- **`lead_captured`** — A boolean flag that prevents duplicate tool invocations.

This explicit state design ensures **memory retention across 5–6+ conversation turns** without relying on fragile prompt-based memory injection.

---

### Agent Graph Architecture

The agent is implemented as a **LangGraph StateGraph** with four processing nodes connected by conditional edges:

```
                    ┌─────────────────────────────┐
                    │        USER MESSAGE          │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     CLASSIFY INTENT NODE     │
                    │  (LLM-based 3-class router)  │
                    └──────────────┬──────────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
             "greeting"    "product_inquiry"  "high_intent"
                   │               │               │
                   ▼               ▼               ▼
          ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
          │   GREETING   │ │  RAG NODE   │ │ LEAD QUALIFIER   │
          │    NODE      │ │             │ │      NODE         │
          │              │ │ 1. Embed    │ │                   │
          │ Friendly     │ │    query    │ │ 1. Extract info   │
          │ response     │ │ 2. FAISS    │ │    from message   │
          │              │ │    search   │ │ 2. Check missing  │
          │              │ │ 3. Generate │ │    fields         │
          │              │ │    answer   │ │ 3. Ask or capture │
          └──────┬───────┘ └──────┬──────┘ └────────┬─────────┘
                 │                │                  │
                 │                │          ┌───────┴───────┐
                 │                │          │               │
                 │                │    Missing info?   All collected?
                 │                │          │               │
                 │                │          ▼               ▼
                 │                │     Ask user      ┌───────────┐
                 │                │                   │TOOL: mock │
                 │                │                   │lead_capture│
                 │                │                   └─────┬─────┘
                 │                │                         │
                 ▼                ▼                         ▼
                    ┌─────────────────────────────┐
                    │      AI RESPONSE TO USER     │
                    └─────────────────────────────┘
```

### Node Descriptions

| Node | Purpose | Input | Output |
|---|---|---|---|
| **Classify Intent** | Routes user message to the correct handler using an LLM with a few-shot system prompt | Latest user message + conversation context | Intent label: `greeting` / `product_inquiry` / `high_intent` |
| **Handle Greeting** | Responds warmly to casual messages | Conversation history | Friendly AI response |
| **Handle Product Inquiry** | Retrieves relevant info from knowledge base via RAG and generates an accurate answer | User query → FAISS similarity search → top-3 chunks | Context-grounded AI response |
| **Handle Lead Qualification** | Extracts user details from message, asks for missing fields, or triggers lead capture when complete | Lead state + latest message | Updated lead_info + AI response / tool execution |

### Data Flow Per Turn

```
1. User types message
2. Message appended to state.messages
3. Graph invoked with current state
4. classify_intent_node → reads last 6 messages → LLM classifies intent
5. Conditional edge routes to handler
6. Handler node processes → returns updated state + AI message
7. State persisted for next turn
```

---

## How It Works

### Intent Classification

The intent classifier uses a dedicated LLM call with a carefully crafted **few-shot system prompt** that maps user messages to one of three categories:

| Intent | Trigger Examples | Agent Action |
|---|---|---|
| `greeting` | "Hi", "Hello", "What's up?" | Friendly intro, ask how to help |
| `product_inquiry` | "What are your plans?", "How much?", "Tell me about features" | RAG retrieval → informed answer |
| `high_intent` | "I want to try Pro", "Sign me up", providing name/email | Lead qualification flow |

**Key design decision:** The classifier receives the **last 6 messages** (not just the latest) to correctly detect **intent shifts** — e.g., when a user transitions from asking about pricing to expressing buying intent.

### Lead Capture Safeguards

The `mock_lead_capture()` tool is **never triggered prematurely**. The qualification node:

1. Uses an LLM to **extract** name, email, and platform from the user's message
2. **Merges** extracted values into the persistent `lead_info` state
3. **Validates** all three fields are present before calling the tool
4. Sets `lead_captured = True` to **prevent duplicate captures**

---

## RAG Pipeline

```
┌──────────────────┐      ┌─────────────────┐      ┌──────────────┐
│  autostream_kb   │      │  Google Generative│     │    FAISS      │
│     .json        │─────▶│  AI Embeddings   │────▶│  Vector Store │
│                  │      │  (embedding-001)  │     │  (in-memory)  │
│  13 documents:   │      └─────────────────┘      └──────┬───────┘
│  - Company info  │                                       │
│  - Basic Plan    │      ┌─────────────────┐              │
│  - Pro Plan      │      │   User Query    │              │
│  - Comparison    │      │  "What's your   │──── embed ──▶│
│  - Policies (5)  │      │   pricing?"     │              │
│  - FAQ (4)       │      └─────────────────┘      similarity
└──────────────────┘                                search (k=3)
                                                           │
                                                           ▼
                                                   ┌──────────────┐
                                                   │  Top 3 Chunks │
                                                   │  (context)    │
                                                   └──────┬───────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │  LLM generates│
                                                   │  grounded     │
                                                   │  response     │
                                                   └──────────────┘
```

The knowledge base is structured as **13 semantic documents** covering:
- Company overview and product description
- Individual plan details (Basic & Pro) with full feature lists
- A dedicated plan comparison document for side-by-side queries
- 5 policy documents (refund, support, free trial, cancellation, data privacy)
- 4 FAQ entries covering common questions

---

## State Management Deep Dive

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Auto-managed history
    intent: str                                            # Current turn's intent
    lead_info: LeadInfo                                    # Accumulated across turns
    lead_captured: bool                                    # One-time flag
```

### State Lifecycle Across Turns

```
Turn 1: "Hi, tell me about pricing"
  state = { messages: [H1], intent: "product_inquiry", lead_info: {}, lead_captured: False }

Turn 2: "I want to try Pro for YouTube"  
  state = { messages: [H1,A1,H2], intent: "high_intent", lead_info: {platform:"YouTube"}, ... }

Turn 3: "I'm Pranav"
  state = { messages: [H1,A1,H2,A2,H3], intent: "high_intent", lead_info: {platform:"YouTube", name:"Pranav"}, ... }

Turn 4: "pranav@email.com"
  state = { ..., lead_info: {name:"Pranav", email:"pranav@email.com", platform:"YouTube"}, lead_captured: True }
  → mock_lead_capture() CALLED ✓
```

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, I would use the **WhatsApp Business API (Cloud API)** with a webhook-based architecture:

### Deployment Architecture

```
┌────────────┐     ┌─────────────────┐     ┌──────────────────────────┐
│            │     │                 │     │    Cloud Server           │
│  WhatsApp  │────▶│  Meta Cloud     │────▶│  (Railway / AWS Lambda)  │
│   User     │     │  Platform       │     │                          │
│            │◀────│  (webhooks)     │◀────│  ┌──────────────────┐   │
│            │     │                 │     │  │  FastAPI/Flask    │   │
└────────────┘     └─────────────────┘     │  │  Webhook Server  │   │
                                           │  └────────┬─────────┘   │
                                           │           │              │
                                           │  ┌────────▼─────────┐   │
                                           │  │  LangGraph Agent  │   │
                                           │  └────────┬─────────┘   │
                                           │           │              │
                                           │  ┌────────▼─────────┐   │
                                           │  │  Redis / Firestore│   │
                                           │  │  (session state)  │   │
                                           │  └──────────────────┘   │
                                           └──────────────────────────┘
```

### Implementation Steps

1. **Meta Business Account** — Register a WhatsApp Business phone number through the Meta Developer Portal and configure the webhook URL.

2. **Webhook Endpoint** — Build a FastAPI/Flask server deployed on a cloud platform (Railway, AWS Lambda, GCP Cloud Run). It receives incoming messages as HTTP POST requests from Meta's servers.

3. **Webhook Verification** — Implement the GET verification challenge (matching a verify token) that Meta sends during initial setup.

4. **Message Processing** — On each incoming POST, extract the sender's phone number and message text. Load their `AgentState` from the session store, invoke the LangGraph agent, and persist the updated state.

5. **Session State Management** — Use **Redis** or **Firestore** keyed by phone number to persist `AgentState` across stateless HTTP requests. Set a 30-minute TTL to clear stale conversations.

6. **Response Delivery** — After processing, send the agent's reply back via the WhatsApp Cloud API `/messages` endpoint with the appropriate message format.

7. **Production Hardening** — Validate the `X-Hub-Signature-256` header on every incoming webhook for security. Implement exponential backoff for API calls and message queuing (Celery / Cloud Tasks) for reliability.

### Key Considerations

- **Security:** Signature verification on every webhook prevents spoofed requests.
- **Scalability:** Stateless webhook + external state store allows horizontal scaling.
- **Rich UX:** WhatsApp interactive buttons and list messages could enhance the lead capture flow with structured input options.

---

## Project Structure

```
Assignment/
├── main.py                       # Interactive CLI entry point
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── .gitignore
├── README.md
│
├── knowledge_base/
│   └── autostream_kb.json        # RAG knowledge base (pricing, features, policies)
│
└── src/
    ├── __init__.py
    ├── config.py                 # LLM initialization (Gemini 1.5 Flash)
    ├── state.py                  # AgentState TypedDict + LeadInfo schema
    ├── intent.py                 # LLM-based 3-class intent classifier
    ├── rag.py                    # FAISS vector store + Google Embeddings pipeline
    ├── tools.py                  # mock_lead_capture() function
    └── agent.py                  # LangGraph StateGraph (nodes, edges, routing)
```

---

## Evaluation Criteria Coverage

| Criteria | Implementation | Details |
|---|---|---|
| Agent reasoning & intent detection | LLM-based 3-class classifier | Few-shot prompt with 6-message context window for intent shift detection |
| Correct use of RAG | FAISS + Google Embeddings | 13 semantic documents, top-3 similarity retrieval, context-grounded generation |
| Clean state management | LangGraph typed state | `AgentState` with `add_messages` reducer, persistent `lead_info` across turns |
| Proper tool calling logic | Guarded tool execution | Tool fires only after all 3 fields validated; `lead_captured` flag prevents duplicates |
| Code clarity & structure | Modular Python package | 7 focused modules with docstrings, type hints, and separation of concerns |
| Real-world deployability | WhatsApp webhook architecture | Detailed deployment plan with session management, security, and scaling considerations |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph (LangChain) |
| LLM | Gemini 1.5 Flash |
| Embeddings | Google Generative AI Embeddings (`embedding-001`) |
| Vector Store | FAISS (CPU) |
| Knowledge Base | Structured JSON |
| Environment | python-dotenv |
