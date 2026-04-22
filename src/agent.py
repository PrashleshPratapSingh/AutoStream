"""
LangGraph Agent for AutoStream - Social-to-Lead Agentic Workflow.

This module defines the core conversational agent using LangGraph's StateGraph.
The agent routes user messages through intent classification, RAG-powered knowledge
retrieval, lead qualification, and tool execution nodes.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.state import AgentState
from src.config import get_llm
from src.intent import classify_intent
from src.rag import retrieve
from src.tools import mock_lead_capture


# ──────────────────────────────────────────────
# System prompt for the AutoStream agent
# ──────────────────────────────────────────────
AGENT_SYSTEM_PROMPT = """You are the AI assistant for AutoStream — a SaaS platform that provides automated video editing tools for content creators.

Your personality:
- Friendly, professional, and enthusiastic about helping creators
- Concise but informative — avoid overly long responses
- Use occasional emojis to keep the tone warm (but don't overdo it)

Your core rules:
1. Only answer questions about AutoStream using the provided knowledge base context.
2. If you don't know something, say so honestly — never make up information.
3. When a user shows buying intent, smoothly transition to collecting their details.
4. Never ask for details the user has already provided.
5. Keep responses to 2-4 sentences unless the user asks for detailed information.
"""


# ──────────────────────────────────────────────
# Node Functions
# ──────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> dict:
    """
    Node: Classify the user's latest message intent.
    Updates the state with the detected intent.
    """
    llm = get_llm(temperature=0.0)
    intent = classify_intent(state["messages"], llm)
    
    # If lead was already captured and user is still chatting, treat as greeting
    if state.get("lead_captured") and intent == "high_intent":
        intent = "greeting"
    
    return {"intent": intent}


def handle_greeting_node(state: AgentState) -> dict:
    """
    Node: Respond to casual greetings and small talk.
    """
    llm = get_llm()
    
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        SystemMessage(content=(
            "The user is sending a casual greeting or making small talk. "
            "Respond warmly and briefly. If this is the first message, introduce yourself "
            "as AutoStream's AI assistant and ask how you can help. "
            "If the lead has already been captured, thank them and ask if there's anything else."
        )),
    ]
    messages.extend(state["messages"][-4:])
    
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def handle_product_inquiry_node(state: AgentState) -> dict:
    """
    Node: Answer product/pricing questions using RAG retrieval.
    Retrieves relevant context from the knowledge base and generates an informed response.
    """
    llm = get_llm()
    
    # Get the latest user message for retrieval
    latest_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    # Retrieve relevant context from knowledge base
    context = retrieve(latest_message)
    
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        SystemMessage(content=(
            f"Answer the user's question using ONLY the following knowledge base context. "
            f"Be accurate and helpful. If the context doesn't contain the answer, say you'll "
            f"need to check with the team.\n\n"
            f"--- KNOWLEDGE BASE CONTEXT ---\n{context}\n--- END CONTEXT ---"
        )),
    ]
    messages.extend(state["messages"][-6:])
    
    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=response.content)]}


def handle_lead_qualification_node(state: AgentState) -> dict:
    """
    Node: Handle high-intent users by collecting lead information.
    Checks what info is missing and asks for it, or triggers lead capture if complete.
    """
    llm = get_llm()
    lead_info = state.get("lead_info", {})
    
    # Determine what info we already have
    has_name = bool(lead_info.get("name"))
    has_email = bool(lead_info.get("email"))
    has_platform = bool(lead_info.get("platform"))
    
    # Try to extract info from the latest message
    latest_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    # Use LLM to extract any lead info from the message
    extraction_prompt = f"""Extract any of the following information from the user's message. 
Return ONLY a JSON object with keys: "name", "email", "platform". 
Use null for any field not found in the message.
Do NOT infer or make up values — only extract explicitly stated information.

Already collected:
- Name: {lead_info.get('name', 'not yet collected')}
- Email: {lead_info.get('email', 'not yet collected')}  
- Platform: {lead_info.get('platform', 'not yet collected')}

User message: "{latest_message}"

Return ONLY valid JSON, nothing else."""

    extraction_llm = get_llm(temperature=0.0)
    extraction_response = extraction_llm.invoke([HumanMessage(content=extraction_prompt)])
    
    # Parse extracted info
    import json
    try:
        # Clean the response - remove markdown code blocks if present
        raw = extraction_response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        extracted = json.loads(raw.strip())
        
        # Update lead_info with newly extracted values
        if extracted.get("name") and not has_name:
            lead_info["name"] = extracted["name"]
            has_name = True
        if extracted.get("email") and not has_email:
            lead_info["email"] = extracted["email"]
            has_email = True
        if extracted.get("platform") and not has_platform:
            lead_info["platform"] = extracted["platform"]
            has_platform = True
    except (json.JSONDecodeError, KeyError, AttributeError):
        pass  # Extraction failed, will ask for info instead
    
    # Check if all info is collected
    if has_name and has_email and has_platform:
        # All info collected — trigger lead capture
        result = mock_lead_capture(
            lead_info["name"],
            lead_info["email"],
            lead_info["platform"],
        )
        
        response_msg = (
            f"Awesome, you're all set! I've registered your interest:\n\n"
            f"* Name: {lead_info['name']}\n"
            f"* Email: {lead_info['email']}\n"
            f"* Platform: {lead_info['platform']}\n\n"
            f"Our team will reach out to you shortly with your Pro plan trial details. "
            f"Is there anything else I can help you with?"
        )
        
        return {
            "messages": [AIMessage(content=response_msg)],
            "lead_info": lead_info,
            "lead_captured": True,
        }
    
    # Build a prompt to ask for missing info naturally
    missing = []
    if not has_name:
        missing.append("name")
    if not has_email:
        missing.append("email")
    if not has_platform:
        missing.append("creator platform (e.g., YouTube, Instagram, TikTok)")
    
    collected_str = ""
    if has_name:
        collected_str += f"  Name: {lead_info['name']}\n"
    if has_email:
        collected_str += f"  Email: {lead_info['email']}\n"
    if has_platform:
        collected_str += f"  Platform: {lead_info['platform']}\n"
    
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        SystemMessage(content=(
            f"The user has shown interest in signing up for AutoStream. "
            f"You need to collect their details for registration.\n\n"
            f"Already collected:\n{collected_str if collected_str else '  Nothing yet'}\n\n"
            f"Still needed: {', '.join(missing)}\n\n"
            f"Ask for the missing information naturally and conversationally. "
            f"Don't list all missing fields robotically — weave it into a friendly response. "
            f"Acknowledge any info they just provided."
        )),
    ]
    messages.extend(state["messages"][-4:])
    
    response = llm.invoke(messages)
    return {
        "messages": [AIMessage(content=response.content)],
        "lead_info": lead_info,
    }


# ──────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge: Route to the appropriate handler node based on classified intent.
    """
    intent = state.get("intent", "greeting")
    
    if intent == "greeting":
        return "handle_greeting"
    elif intent == "product_inquiry":
        return "handle_product_inquiry"
    elif intent == "high_intent":
        return "handle_lead_qualification"
    else:
        return "handle_greeting"  # Fallback


# ──────────────────────────────────────────────
# Graph Construction
# ──────────────────────────────────────────────

def build_agent() -> StateGraph:
    """
    Build and compile the LangGraph agent with all nodes and edges.
    
    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Create the graph with our state schema
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("handle_greeting", handle_greeting_node)
    graph.add_node("handle_product_inquiry", handle_product_inquiry_node)
    graph.add_node("handle_lead_qualification", handle_lead_qualification_node)
    
    # Set entry point
    graph.set_entry_point("classify_intent")
    
    # Add conditional routing from intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_product_inquiry": "handle_product_inquiry",
            "handle_lead_qualification": "handle_lead_qualification",
        },
    )
    
    # All handler nodes terminate the graph (response is sent back to user)
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_product_inquiry", END)
    graph.add_edge("handle_lead_qualification", END)
    
    return graph.compile()
