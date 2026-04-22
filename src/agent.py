"""
LangGraph Agent for AutoStream - Social-to-Lead Agentic Workflow.

Architecture: Single-call-per-turn design using LangGraph StateGraph.
Each user message triggers exactly ONE LLM call that simultaneously:
  1. Classifies intent
  2. Generates the response
  3. Extracts any lead info provided

This eliminates the latency of chaining multiple sequential LLM calls.
"""

import json
import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.state import AgentState
from src.config import get_llm
from src.rag import retrieve
from src.tools import mock_lead_capture


# ──────────────────────────────────────────────
# System Prompt — Single unified prompt that handles all cases
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are the AI sales assistant for AutoStream, a SaaS platform providing automated video editing tools for content creators.

YOUR CAPABILITIES:
1. Greet users warmly and introduce AutoStream
2. Answer product questions ONLY from the provided knowledge base context
3. Identify when users want to sign up (high intent) and collect their details
4. Capture leads by collecting: Name, Email, and Creator Platform

RESPONSE FORMAT — You MUST respond with valid JSON in this exact structure:
{{
  "intent": "<greeting|product_inquiry|high_intent>",
  "extracted_info": {{
    "name": "<extracted name or null>",
    "email": "<extracted email or null>",
    "platform": "<extracted platform or null>"
  }},
  "response": "<your conversational response to the user>"
}}

INTENT RULES:
- "greeting": Casual hellos, small talk, general chit-chat
- "product_inquiry": Questions about features, pricing, plans, policies, comparisons
- "high_intent": User wants to sign up, try a plan, shows buying intent, OR is providing personal details (name/email/platform) after being asked

BEHAVIOR RULES:
1. For greetings: Be friendly, briefly introduce AutoStream, ask how you can help.
2. For product inquiries: Answer accurately using ONLY the knowledge base context below. Never invent features or prices.
3. For pricing specifically: YOU MUST ALWAYS include the dollar sign ($) when quoting prices (e.g. say "$29/month", NEVER just "29/month").
4. For high intent: If you don't have all 3 fields (name, email, platform), ask for the MISSING ones naturally. Don't re-ask for info already collected.
5. Extract name, email, and platform from the user's message if they provide any. Set to null if not present.
6. Keep responses concise (2-4 sentences).
7. IMPORTANT: Respond ONLY with the JSON object. No markdown, no code blocks, no extra text.

{context_section}

ALREADY COLLECTED LEAD INFO:
- Name: {lead_name}
- Email: {lead_email}
- Platform: {lead_platform}
- Lead Already Captured: {lead_captured}"""


# ──────────────────────────────────────────────
# The single processing node
# ──────────────────────────────────────────────
def process_message(state: AgentState) -> dict:
    """
    Single node that processes each user message in one LLM call.
    Handles intent classification, RAG retrieval, response generation,
    and info extraction all at once.
    """
    llm = get_llm(temperature=0.2)
    
    lead_info = state.get("lead_info", {})
    is_captured = state.get("lead_captured", False)
    
    # Get the latest user message for RAG
    latest_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_msg = msg.content
            break
    
    # Always retrieve context (cheap operation, ensures accuracy)
    context = retrieve(latest_msg, k=3)
    context_section = f"KNOWLEDGE BASE CONTEXT (use this to answer product questions):\n{context}"
    
    # Build the system prompt with current state
    system = SYSTEM_PROMPT.format(
        context_section=context_section,
        lead_name=lead_info.get("name", "not collected"),
        lead_email=lead_info.get("email", "not collected"),
        lead_platform=lead_info.get("platform", "not collected"),
        lead_captured="Yes" if is_captured else "No",
    )
    
    # Single LLM call with conversation history
    messages = [SystemMessage(content=system)]
    # Include last 10 messages for context (covers 5-6 turn requirement)
    recent = state["messages"][-10:] if len(state["messages"]) > 10 else state["messages"]
    messages.extend(recent)
    
    response = llm.invoke(messages)
    raw = response.content.strip()
    
    # Parse the structured JSON response
    intent = "greeting"
    reply = raw
    extracted = {}
    
    try:
        # Clean markdown code blocks if LLM wraps in ```json
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        
        parsed = json.loads(cleaned)
        intent = parsed.get("intent", "greeting")
        reply = parsed.get("response", raw)
        extracted = parsed.get("extracted_info", {})
    except (json.JSONDecodeError, KeyError):
        # If JSON parsing fails, use the raw text as response
        # Try to detect intent from keywords
        lower = raw.lower()
        if any(w in lower for w in ["sign up", "try", "interested", "start"]):
            intent = "high_intent"
        elif any(w in lower for w in ["price", "plan", "feature", "cost", "support"]):
            intent = "product_inquiry"
    
    # Update lead info with any newly extracted values
    updated_lead = dict(lead_info)
    if extracted:
        if extracted.get("name") and extracted["name"] != "null":
            updated_lead["name"] = extracted["name"]
        if extracted.get("email") and extracted["email"] != "null":
            updated_lead["email"] = extracted["email"]
        if extracted.get("platform") and extracted["platform"] != "null":
            updated_lead["platform"] = extracted["platform"]
    
    # Check if we should trigger lead capture
    captured = is_captured
    has_all = (
        updated_lead.get("name")
        and updated_lead.get("email")
        and updated_lead.get("platform")
    )
    
    if has_all and not captured:
        # Call the mock lead capture tool
        result = mock_lead_capture(
            updated_lead["name"],
            updated_lead["email"],
            updated_lead["platform"],
        )
        captured = True
        
        # Append confirmation to the response
        reply += (
            f"\n\nLead captured successfully! "
            f"Our team will reach out to {updated_lead['name']} at {updated_lead['email']} shortly."
        )
    
    return {
        "messages": [AIMessage(content=reply)],
        "intent": intent,
        "lead_info": updated_lead,
        "lead_captured": captured,
    }


# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────
def build_agent():
    """
    Build and compile the LangGraph agent.
    
    Simple, clean graph: entry -> process_message -> END
    All intelligence is in the single node, keeping the graph
    focused on state management (which is LangGraph's strength).
    """
    graph = StateGraph(AgentState)
    graph.add_node("process_message", process_message)
    graph.set_entry_point("process_message")
    graph.add_edge("process_message", END)
    return graph.compile()
