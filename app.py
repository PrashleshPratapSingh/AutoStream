"""
AutoStream AI Agent — Minimal Streamlit Interface

A clean, straightforward UI focused directly on the assignment requirements:
Intent Classification, RAG, and Tool Execution (Lead Capture).

Usage:
    streamlit run app.py
"""

import json
import os
import streamlit as st
from langchain_core.messages import HumanMessage
from src.agent import build_agent


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Load Knowledge Base for Sidebar
# ──────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base", "autostream_kb.json")
with open(KB_PATH, "r", encoding="utf-8") as f:
    KB_DATA = json.load(f)


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


# ──────────────────────────────────────────────
# Sidebar - Agent Status & KB Reference
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Internal Agent State")
    
    # 1. State Information
    st.subheader("1. Intent & Lead Data")
    st.write(f"**Current Intent:** `{st.session_state.state.get('intent', 'waiting...')}`")
    
    lead_info = st.session_state.state.get("lead_info", {})
    captured = st.session_state.state.get("lead_captured", False)
    
    st.write("**Collected Fields:**")
    st.write(f"- Name: `{lead_info.get('name', 'None')}`")
    st.write(f"- Email: `{lead_info.get('email', 'None')}`")
    st.write(f"- Platform: `{lead_info.get('platform', 'None')}`")
    
    if captured:
        st.success("✅ mock_lead_capture() executed successfully!")
    else:
        st.info("Waiting for all fields to capture lead...")
        
    st.divider()
    
    # 2. Knowledge Base Reference
    st.subheader("2. RAG Knowledge Base")
    with st.expander("Show Pricing Details"):
        st.write("**Basic Plan:** $29/month")
        for feature in KB_DATA["pricing"]["basic"]["features"]:
            st.write(f"- {feature}")
        
        st.write("**Pro Plan:** $79/month")
        for feature in KB_DATA["pricing"]["pro"]["features"]:
            st.write(f"- {feature}")

    with st.expander("Show Policies"):
        for policy, details in KB_DATA["policies"].items():
            st.write(f"**{policy.replace('_', ' ').title()}:** {details}")
            
    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.state = {
            "messages": [], "intent": "", "lead_info": {}, "lead_captured": False,
        }
        st.session_state.chat_history = []
        st.rerun()


# ──────────────────────────────────────────────
# Main Chat Application
# ──────────────────────────────────────────────
st.title("🎬 AutoStream AI Sales Agent")
st.markdown("Hello! I'm here to help you learn about AutoStream. Try asking:")
st.markdown("> *\"Hi, tell me about your pricing.\"*")

# Display conversation history
for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role, avatar="🙋" if role == "user" else "🤖"):
        st.markdown(content)

# Handle user input
if prompt := st.chat_input("Type your message here..."):
    # Render user message
    with st.chat_message("user", avatar="🙋"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Run backend agent logic
    st.session_state.state["messages"].append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing..."):
            try:
                # Single LLM call per turn
                result = st.session_state.agent.invoke(st.session_state.state)
                
                # Update frontend state variables
                st.session_state.state["messages"] = result["messages"]
                st.session_state.state["intent"] = result.get("intent", "")
                st.session_state.state["lead_info"] = result.get("lead_info", {})
                st.session_state.state["lead_captured"] = result.get("lead_captured", False)
                
                # Render AI response
                response_text = result["messages"][-1].content
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    st.error("API Quota Reached. Please use a fresh GOOGLE_API_KEY.")
                else:
                    st.error(f"Error: {e}")
                    
    # Force a rerun to update the sidebar values instantly
    st.rerun()
