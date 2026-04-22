"""
AutoStream AI Agent — Streamlit Web Interface

A polished SaaS-style chat UI for the AutoStream conversational AI agent.
Features a pricing section, agent dashboard, and professional chat interface.

Usage:
    streamlit run app.py
"""

import json
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import build_agent


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI Agent | Automated Video Editing for Creators",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Load Knowledge Base for Pricing Display
# ──────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base", "autostream_kb.json")
with open(KB_PATH, "r", encoding="utf-8") as f:
    KB_DATA = json.load(f)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Hero Section ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border: 1px solid #667eea40;
        color: #667eea;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.25rem 0;
        line-height: 1.2;
    }
    .hero p {
        color: #6b7280;
        font-size: 1.05rem;
        margin: 0.5rem auto 0;
        max-width: 600px;
        line-height: 1.5;
    }
    
    /* ── Pricing Cards ── */
    .pricing-grid {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        padding: 1rem 0 2rem;
        flex-wrap: wrap;
    }
    .pricing-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 2rem 1.75rem;
        width: 320px;
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pricing-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
    }
    .pricing-card.featured {
        border: 2px solid #667eea;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.12);
    }
    .pricing-card.featured::before {
        content: 'MOST POPULAR';
        position: absolute;
        top: -12px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 1rem;
        border-radius: 999px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    .pricing-card h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0 0 0.25rem;
    }
    .pricing-card .price {
        font-size: 2.5rem;
        font-weight: 800;
        color: #111827;
        margin: 0.5rem 0 0.25rem;
    }
    .pricing-card .price span {
        font-size: 1rem;
        font-weight: 400;
        color: #6b7280;
    }
    .pricing-card .price-desc {
        color: #9ca3af;
        font-size: 0.85rem;
        margin: 0 0 1.25rem;
    }
    .pricing-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    .pricing-card ul li {
        padding: 0.4rem 0;
        font-size: 0.9rem;
        color: #374151;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .pricing-card ul li::before {
        content: '✓';
        color: #10b981;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .pricing-card .cta-btn {
        display: block;
        width: 100%;
        padding: 0.75rem;
        margin-top: 1.5rem;
        border-radius: 10px;
        font-size: 0.9rem;
        font-weight: 600;
        text-align: center;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .btn-outline {
        background: white;
        color: #667eea;
        border: 2px solid #667eea;
    }
    .btn-outline:hover { background: #667eea10; }
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .btn-primary:hover { opacity: 0.9; transform: scale(1.02); }
    
    /* ── Policy Info ── */
    .policy-grid {
        display: flex;
        gap: 1rem;
        justify-content: center;
        padding: 0 0 1.5rem;
        flex-wrap: wrap;
    }
    .policy-item {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        font-size: 0.85rem;
        color: #4b5563;
        text-align: center;
        min-width: 180px;
    }
    .policy-item strong {
        color: #1f2937;
        display: block;
        font-size: 0.8rem;
        margin-bottom: 0.2rem;
    }
    
    /* ── Section Divider ── */
    .section-divider {
        text-align: center;
        padding: 1.5rem 0;
        position: relative;
    }
    .section-divider::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    }
    .section-divider span {
        background: white;
        padding: 0 1.5rem;
        position: relative;
        color: #9ca3af;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* ── Status Badges ── */
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
    
    /* ── Sidebar ── */
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
    
    /* ── Lead Card ── */
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
    
    /* ── Success Banner ── */
    .success-banner {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-banner h3 { margin: 0; font-size: 1.1rem; }
    .success-banner p { margin: 0.25rem 0 0; font-size: 0.85rem; opacity: 0.9; }
    
    /* ── Agent Workflow Diagram ── */
    .workflow-bar {
        display: flex;
        gap: 0;
        justify-content: center;
        padding: 0.75rem 0 1.5rem;
        flex-wrap: wrap;
    }
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0;
    }
    .workflow-node {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.4rem 0.85rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: #374151;
        white-space: nowrap;
    }
    .workflow-node.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    .workflow-arrow {
        color: #d1d5db;
        padding: 0 0.35rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session State
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

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Chat"


# ──────────────────────────────────────────────
# Sidebar — Agent Dashboard
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 AutoStream")
    st.caption("AI-Powered Video Editing for Creators")
    st.divider()
    
    st.markdown("#### Agent Dashboard")
    
    # Current intent
    intent = st.session_state.state.get("intent", "")
    if intent:
        badge_map = {
            "greeting": ("badge-greeting", "👋 Greeting"),
            "product_inquiry": ("badge-inquiry", "🔍 Product Inquiry"),
            "high_intent": ("badge-high", "🎯 High Intent — Lead"),
        }
        badge_class, badge_label = badge_map.get(intent, ("badge-greeting", intent))

        st.markdown(f"""
        <div class="sidebar-section">
            <h4>Detected Intent</h4>
            <span class="status-badge {badge_class}">{badge_label}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow visualization
    intent_step = {
        "": 0, "greeting": 1, "product_inquiry": 2, "high_intent": 3,
    }
    step = intent_step.get(intent, 0)
    if st.session_state.state.get("lead_captured"):
        step = 4
    
    nodes = ["Intent", "Greeting", "RAG", "Lead Qual.", "Tool"]
    node_html = ""
    for i, name in enumerate(nodes):
        cls = "workflow-node active" if i == step else "workflow-node"
        arrow = '<span class="workflow-arrow">→</span>' if i < len(nodes) - 1 else ""
        node_html += f'<span class="{cls}">{name}</span>{arrow}'
    
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>Agent Workflow</h4>
        <div style="display:flex;flex-wrap:wrap;gap:0.2rem;align-items:center;">{node_html}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Lead info
    lead_info = st.session_state.state.get("lead_info", {})
    lead_captured = st.session_state.state.get("lead_captured", False)
    
    if lead_info or lead_captured:
        status = "✅ Lead Captured" if lead_captured else "📋 Collecting Info..."
        st.markdown(f"""
        <div class="lead-card">
            <h4>{status}</h4>
            <p><strong>Name:</strong> {lead_info.get('name', '—')}</p>
            <p><strong>Email:</strong> {lead_info.get('email', '—')}</p>
            <p><strong>Platform:</strong> {lead_info.get('platform', '—')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>Session Stats</h4>
        <p style="font-size:0.85rem;color:#374151;margin:0.15rem 0;">
            Conversation Turns: <strong>{st.session_state.turn_count}</strong>
        </p>
        <p style="font-size:0.85rem;color:#374151;margin:0.15rem 0;">
            Total Messages: <strong>{len(st.session_state.state.get('messages', []))}</strong>
        </p>
        <p style="font-size:0.85rem;color:#374151;margin:0.15rem 0;">
            Lead Captured: <strong>{'Yes ✅' if lead_captured else 'No'}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture
    with st.expander("🏗️ Architecture Details"):
        st.markdown("""
        **Framework:** LangGraph StateGraph  
        **LLM:** Gemini 2.0 Flash  
        **Embeddings:** Google AI (`embedding-001`)  
        **Vector Store:** FAISS (in-memory)  
        **State:** Typed `AgentState` with `add_messages` reducer
        
        **Graph Nodes:**
        1. `classify_intent` — 3-class LLM router
        2. `handle_greeting` — Casual response
        3. `handle_product_inquiry` — RAG retrieval
        4. `handle_lead_qualification` — Info extraction + tool
        """)
    
    # Reset
    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.state = {
            "messages": [], "intent": "", "lead_info": {}, "lead_captured": False,
        }
        st.session_state.chat_history = []
        st.session_state.turn_count = 0
        st.rerun()


# ──────────────────────────────────────────────
# Main Content — Tabs
# ──────────────────────────────────────────────
tab_chat, tab_pricing = st.tabs(["💬 AI Agent Chat", "💰 Pricing & Plans"])


# ──────────────── Chat Tab ────────────────
with tab_chat:
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Powered by LangGraph + Gemini</div>
        <h1>AutoStream AI Agent</h1>
        <p>Chat with our AI assistant to learn about AutoStream's video editing tools, 
        explore pricing plans, and get started as a creator.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lead captured banner
    if st.session_state.state.get("lead_captured"):
        li = st.session_state.state.get("lead_info", {})
        st.markdown(f"""
        <div class="success-banner">
            <h3>✅ Lead Successfully Captured!</h3>
            <p>{li.get('name', '')} &bull; {li.get('email', '')} &bull; {li.get('platform', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about AutoStream... (e.g., 'What are your pricing plans?')"):
        with st.chat_message("user", avatar="🙋"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        st.session_state.state["messages"].append(HumanMessage(content=prompt))
        st.session_state.turn_count += 1
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.agent.invoke(st.session_state.state)
                    
                    st.session_state.state["messages"] = result["messages"]
                    st.session_state.state["intent"] = result.get("intent", st.session_state.state["intent"])
                    st.session_state.state["lead_info"] = result.get("lead_info", st.session_state.state["lead_info"])
                    st.session_state.state["lead_captured"] = result.get("lead_captured", st.session_state.state["lead_captured"])
                    
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


# ──────────────── Pricing Tab ────────────────
with tab_pricing:
    # Pricing hero
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">Simple, Transparent Pricing</div>
        <h1>Choose Your Plan</h1>
        <p>Start creating professional videos today. No hidden fees, cancel anytime.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pricing cards
    basic = KB_DATA["pricing"]["basic"]
    pro = KB_DATA["pricing"]["pro"]
    
    basic_features = "".join(f"<li>{f}</li>" for f in basic["features"])
    pro_features = "".join(f"<li>{f}</li>" for f in pro["features"])
    
    st.markdown(f"""
    <div class="pricing-grid">
        <div class="pricing-card">
            <h3>🎬 {basic['plan_name']}</h3>
            <div class="price">{basic['price']}<span> /month</span></div>
            <div class="price-desc">Perfect for getting started</div>
            <ul>{basic_features}</ul>
            <div class="cta-btn btn-outline">Get Started</div>
        </div>
        <div class="pricing-card featured">
            <h3>🚀 {pro['plan_name']}</h3>
            <div class="price">{pro['price']}<span> /month</span></div>
            <div class="price-desc">For serious content creators</div>
            <ul>{pro_features}</ul>
            <div class="cta-btn btn-primary">Start Free Trial</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Policies
    policies = KB_DATA["policies"]
    st.markdown(f"""
    <div class="section-divider"><span>Policies & Support</span></div>
    <div class="policy-grid">
        <div class="policy-item">
            <strong>🔄 Refund Policy</strong>
            {policies['refund']}
        </div>
        <div class="policy-item">
            <strong>🎧 Support</strong>
            {policies['support']}
        </div>
        <div class="policy-item">
            <strong>🆓 Free Trial</strong>
            {policies['free_trial']}
        </div>
        <div class="policy-item">
            <strong>❌ Cancellation</strong>
            {policies['cancellation']}
        </div>
        <div class="policy-item">
            <strong>🔒 Data Privacy</strong>
            {policies['data_privacy']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # FAQ
    st.markdown('<div class="section-divider"><span>Frequently Asked Questions</span></div>', unsafe_allow_html=True)
    
    for faq in KB_DATA["faq"]:
        with st.expander(faq["question"]):
            st.write(faq["answer"])
    
    # CTA
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem;">
        <p style="color:#6b7280; font-size:0.95rem;">
            Have questions? Chat with our AI agent in the <strong>💬 AI Agent Chat</strong> tab!
        </p>
    </div>
    """, unsafe_allow_html=True)
