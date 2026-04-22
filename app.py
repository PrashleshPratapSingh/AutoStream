"""
AutoStream AI Agent — Streamlit Web Interface

A polished chat UI for the AutoStream conversational AI agent.
Demonstrates intent detection, RAG-powered knowledge retrieval,
and lead capture workflow in a professional web interface.

Usage:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import build_agent


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI Agent",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for Premium Look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
    }
    .badge-greeting { background: #dbeafe; color: #1d4ed8; }
    .badge-inquiry { background: #fef3c7; color: #b45309; }
    .badge-high { background: #d1fae5; color: #065f46; }
    .badge-captured { background: #ede9fe; color: #5b21b6; }
    
    /* Lead info card */
    .lead-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 1px solid #667eea30;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
    }
    .lead-card h4 {
        margin: 0 0 0.5rem;
        font-size: 0.85rem;
        color: #667eea;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .lead-card p {
        margin: 0.15rem 0;
        font-size: 0.9rem;
        color: #374151;
    }
    
    /* Success banner */
    .success-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-banner h3 {
        margin: 0;
        font-size: 1.1rem;
    }
    .success-banner p {
        margin: 0.25rem 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e5e7eb;
    }
    .sidebar-section h4 {
        margin: 0 0 0.5rem;
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Chat input area */
    .stChatInput {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "intent": "",
        "lead_info": {},
        "lead_captured": False,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Agent Dashboard")
    
    # Current intent
    intent = st.session_state.state.get("intent", "")
    if intent:
        badge_class = {
            "greeting": "badge-greeting",
            "product_inquiry": "badge-inquiry",
            "high_intent": "badge-high",
        }.get(intent, "badge-greeting")
        
        intent_labels = {
            "greeting": "👋 Greeting",
            "product_inquiry": "🔍 Product Inquiry",
            "high_intent": "🎯 High Intent",
        }
        
        st.markdown(f"""
        <div class="sidebar-section">
            <h4>Current Intent</h4>
            <span class="status-badge {badge_class}">{intent_labels.get(intent, intent)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Lead info
    lead_info = st.session_state.state.get("lead_info", {})
    lead_captured = st.session_state.state.get("lead_captured", False)
    
    if lead_info or lead_captured:
        st.markdown(f"""
        <div class="lead-card">
            <h4>{"✅ Lead Captured" if lead_captured else "📋 Collecting Lead Info"}</h4>
            <p><strong>Name:</strong> {lead_info.get('name', '—')}</p>
            <p><strong>Email:</strong> {lead_info.get('email', '—')}</p>
            <p><strong>Platform:</strong> {lead_info.get('platform', '—')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Conversation stats
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>Session Info</h4>
        <p style="font-size: 0.85rem; color: #374151; margin: 0.15rem 0;">
            Turns: <strong>{st.session_state.turn_count}</strong>
        </p>
        <p style="font-size: 0.85rem; color: #374151; margin: 0.15rem 0;">
            Messages: <strong>{len(st.session_state.state.get('messages', []))}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture info
    with st.expander("🏗️ Architecture"):
        st.markdown("""
        **Stack:** LangGraph + Gemini 2.0 Flash
        
        **Nodes:**
        1. Intent Classifier
        2. Greeting Handler
        3. RAG Product Inquiry
        4. Lead Qualifier + Tool
        
        **State:** Typed `AgentState` with `add_messages` reducer
        """)
    
    # Reset button
    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.state = {
            "messages": [],
            "intent": "",
            "lead_info": {},
            "lead_captured": False,
        }
        st.session_state.chat_history = []
        st.session_state.turn_count = 0
        st.rerun()


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎬 AutoStream AI Agent</h1>
    <p>Social-to-Lead Agentic Workflow — Powered by LangGraph</p>
</div>
""", unsafe_allow_html=True)

# Display lead captured banner
if st.session_state.state.get("lead_captured"):
    li = st.session_state.state.get("lead_info", {})
    st.markdown(f"""
    <div class="success-banner">
        <h3>✅ Lead Successfully Captured!</h3>
        <p>{li.get('name', '')} • {li.get('email', '')} • {li.get('platform', '')}</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message... (e.g., 'Hi, tell me about your pricing')"):
    # Display user message
    with st.chat_message("user", avatar="🙋"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Update state with user message
    st.session_state.state["messages"].append(HumanMessage(content=prompt))
    st.session_state.turn_count += 1
    
    # Run the agent
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.invoke(st.session_state.state)
                
                # Update state
                st.session_state.state["messages"] = result["messages"]
                st.session_state.state["intent"] = result.get("intent", st.session_state.state["intent"])
                st.session_state.state["lead_info"] = result.get("lead_info", st.session_state.state["lead_info"])
                st.session_state.state["lead_captured"] = result.get("lead_captured", st.session_state.state["lead_captured"])
                
                # Get and display response
                response = result["messages"][-1].content
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = str(e)
                if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                    st.error("⚠️ API rate limit reached. Please wait a moment and try again.")
                else:
                    st.error(f"Error: {error_msg}")
    
    st.rerun()
