"""
AutoStream AI Agent — Premium Streamlit Interface

A dark-themed, glassmorphic SaaS-style UI for the AutoStream
conversational AI agent with animated backgrounds and premium UX.

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
    page_title="AutoStream AI | Conversational Lead Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Load Knowledge Base
# ──────────────────────────────────────────────
KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base", "autostream_kb.json")
with open(KB_PATH, "r", encoding="utf-8") as f:
    KB_DATA = json.load(f)


# ──────────────────────────────────────────────
# Premium Dark Theme CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ── Reset & Global ── */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        background: #0a0a0f;
        color: #e2e8f0;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    /* Animated mesh gradient background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            radial-gradient(ellipse at 20% 50%, rgba(120, 80, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(255, 100, 200, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 60% 80%, rgba(50, 200, 255, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    /* ── Navbar ── */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin: -1rem -1rem 0;
        background: rgba(10, 10, 15, 0.8);
        backdrop-filter: blur(20px);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }
    .nav-logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    .nav-logo-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
    }
    .nav-logo-text span {
        color: #818cf8;
    }
    .nav-pills {
        display: flex;
        gap: 0.25rem;
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.2rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .nav-pill {
        padding: 0.45rem 1.1rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #94a3b8;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .nav-pill.active {
        background: rgba(124, 58, 237, 0.2);
        color: #c4b5fd;
        border: 1px solid rgba(124, 58, 237, 0.3);
    }
    .nav-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        color: #64748b;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.4);
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* ── Glass Card Base ── */
    .glass {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        backdrop-filter: blur(12px);
    }
    
    /* ── Hero ── */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem 2rem;
        position: relative;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.25);
        color: #a78bfa;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    .hero-chip .chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #a78bfa;
        animation: pulse-dot 2s infinite;
    }
    .hero-section h1 {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
        margin: 0 0 0.75rem;
        letter-spacing: -0.03em;
    }
    .hero-section h1 .gradient-text {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-section .hero-sub {
        color: #64748b;
        font-size: 1.05rem;
        max-width: 550px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* ── Metrics Row ── */
    .metrics-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        padding: 1.5rem 0;
        flex-wrap: wrap;
    }
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 140px;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0;
    }
    .metric-card .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.2rem 0 0;
    }
    
    /* ── Chat Container ── */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Streamlit chat styling overrides */
    .stChatMessage {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stChatInput"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #e2e8f0 !important;
    }
    
    /* ── Pricing Section ── */
    .pricing-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .pricing-header h2 {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .pricing-header h2 .gradient-text {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pricing-header p {
        color: #64748b;
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }
    
    .pricing-grid {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        padding: 1rem 0 2rem;
        flex-wrap: wrap;
    }
    .plan-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 2rem;
        width: 340px;
        position: relative;
        transition: all 0.3s ease;
    }
    .plan-card:hover {
        transform: translateY(-6px);
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 20px 60px rgba(124, 58, 237, 0.1);
    }
    .plan-card.pro {
        border-color: rgba(124, 58, 237, 0.35);
        background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(37,99,235,0.05) 100%);
    }
    .plan-card.pro::before {
        content: 'RECOMMENDED';
        position: absolute;
        top: -10px;
        right: 24px;
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 6px;
        font-size: 0.6rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }
    .plan-name {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin: 0 0 0.5rem;
    }
    .plan-price {
        font-size: 3rem;
        font-weight: 900;
        color: #f1f5f9;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .plan-price span {
        font-size: 1rem;
        font-weight: 400;
        color: #64748b;
    }
    .plan-desc {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0.25rem 0 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .plan-features {
        list-style: none;
        padding: 0;
        margin: 0 0 1.5rem;
    }
    .plan-features li {
        padding: 0.45rem 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .plan-features li .check {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        flex-shrink: 0;
    }
    .check-basic {
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
    }
    .check-pro {
        background: rgba(124, 58, 237, 0.15);
        color: #a78bfa;
    }
    .plan-btn {
        display: block;
        width: 100%;
        padding: 0.85rem;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        border: none;
    }
    .plan-btn-outline {
        background: transparent;
        color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .plan-btn-outline:hover {
        background: rgba(255,255,255,0.04);
        border-color: rgba(255,255,255,0.2);
        color: #e2e8f0;
    }
    .plan-btn-primary {
        background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
        color: white;
    }
    .plan-btn-primary:hover {
        opacity: 0.9;
        transform: scale(1.01);
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3);
    }
    
    /* ── Policy Cards ── */
    .policy-section {
        padding: 2rem 0;
    }
    .policy-section h3 {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 1rem;
    }
    .policy-grid {
        display: flex;
        gap: 0.75rem;
        justify-content: center;
        flex-wrap: wrap;
        max-width: 900px;
        margin: 0 auto;
    }
    .policy-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        flex: 1 1 250px;
        max-width: 280px;
    }
    .policy-card .policy-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.4rem;
    }
    .policy-card .policy-text {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.5;
        margin: 0;
    }
    
    /* ── Agent Info Panel / Sidebar ── */
    .agent-panel {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .agent-panel h4 {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin: 0 0 0.75rem;
    }
    .intent-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .intent-greeting {
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    .intent-inquiry {
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .intent-high {
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    /* Lead progress */
    .lead-progress {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .lead-field {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.75rem;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .lead-field .field-label {
        font-size: 0.78rem;
        color: #64748b;
    }
    .lead-field .field-value {
        font-size: 0.82rem;
        font-weight: 500;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
    }
    .lead-field .field-pending {
        color: #475569;
        font-style: italic;
    }
    
    /* ── Success State ── */
    .capture-success {
        background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(16,185,129,0.05) 100%);
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        margin: 1rem 0;
    }
    .capture-success h3 {
        color: #4ade80;
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 0.3rem;
    }
    .capture-success p {
        color: #86efac;
        font-size: 0.85rem;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* ── Workflow Steps ── */
    .workflow-steps {
        display: flex;
        align-items: center;
        gap: 0.15rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .wf-step {
        padding: 0.25rem 0.55rem;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 600;
        background: rgba(255,255,255,0.04);
        color: #475569;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .wf-step.active {
        background: rgba(124,58,237,0.15);
        color: #a78bfa;
        border-color: rgba(124,58,237,0.25);
    }
    .wf-arrow {
        color: #334155;
        font-size: 0.7rem;
    }
    
    /* ── Tabs Override ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.25rem;
        border: 1px solid rgba(255,255,255,0.06);
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124,58,237,0.15) !important;
        color: #c4b5fd !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    
    /* ── FAQ ── */
    .stExpander {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }
    .stExpander summary {
        color: #cbd5e1 !important;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
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
        "messages": [], "intent": "", "lead_info": {}, "lead_captured": False,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0


# ──────────────────────────────────────────────
# Navbar
# ──────────────────────────────────────────────
lead_captured = st.session_state.state.get("lead_captured", False)
status_text = "Lead Captured" if lead_captured else "Agent Online"

st.markdown(f"""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-logo-icon">🎬</div>
        <div class="nav-logo-text">Auto<span>Stream</span></div>
    </div>
    <div class="nav-status">
        <div class="status-dot"></div>
        <span>{status_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Layout: Main + Right Panel
# ──────────────────────────────────────────────
col_main, col_panel = st.columns([3, 1], gap="large")


# ──────────────── Right Panel (Agent Dashboard) ────────────────
with col_panel:
    # Intent Detection
    intent = st.session_state.state.get("intent", "")
    st.markdown('<div class="agent-panel"><h4>🧠 Detected Intent</h4>', unsafe_allow_html=True)
    if intent:
        badge_map = {
            "greeting": ("intent-greeting", "👋 Greeting"),
            "product_inquiry": ("intent-inquiry", "🔍 Product Inquiry"),
            "high_intent": ("intent-high", "🎯 High Intent"),
        }
        cls, label = badge_map.get(intent, ("intent-greeting", intent))
        st.markdown(f'<span class="intent-badge {cls}">{label}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#475569;font-size:0.85rem;">Waiting for input...</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow
    steps = ["Intent", "Greeting", "RAG", "Qualify", "Capture"]
    step_map = {"": 0, "greeting": 1, "product_inquiry": 2, "high_intent": 3}
    active_step = step_map.get(intent, 0)
    if lead_captured:
        active_step = 4
    
    steps_html = ""
    for i, s in enumerate(steps):
        cls = "wf-step active" if i == active_step else "wf-step"
        arrow = '<span class="wf-arrow">→</span>' if i < len(steps) - 1 else ""
        steps_html += f'<span class="{cls}">{s}</span>{arrow}'
    
    st.markdown(f"""
    <div class="agent-panel">
        <h4>⚡ Agent Pipeline</h4>
        <div class="workflow-steps">{steps_html}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Lead Info
    lead_info = st.session_state.state.get("lead_info", {})
    name_val = lead_info.get("name")
    email_val = lead_info.get("email")
    plat_val = lead_info.get("platform")
    
    st.markdown(f"""
    <div class="agent-panel">
        <h4>📋 Lead Information</h4>
        <div class="lead-progress">
            <div class="lead-field">
                <span class="field-label">Name</span>
                <span class="field-value {'field-pending' if not name_val else ''}">{name_val or 'pending'}</span>
            </div>
            <div class="lead-field">
                <span class="field-label">Email</span>
                <span class="field-value {'field-pending' if not email_val else ''}">{email_val or 'pending'}</span>
            </div>
            <div class="lead-field">
                <span class="field-label">Platform</span>
                <span class="field-value {'field-pending' if not plat_val else ''}">{plat_val or 'pending'}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if lead_captured:
        st.markdown(f"""
        <div class="capture-success">
            <h3>✅ Lead Captured</h3>
            <p>mock_lead_capture() called</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Session stats
    st.markdown(f"""
    <div class="agent-panel">
        <h4>📊 Session</h4>
        <div class="lead-progress">
            <div class="lead-field">
                <span class="field-label">Turns</span>
                <span class="field-value">{st.session_state.turn_count}</span>
            </div>
            <div class="lead-field">
                <span class="field-label">Messages</span>
                <span class="field-value">{len(st.session_state.state.get('messages', []))}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tech stack
    with st.expander("🏗️ Architecture"):
        st.markdown("""
        <div style="font-size:0.82rem;color:#94a3b8;line-height:1.8;">
        <strong style="color:#c4b5fd;">Framework:</strong> LangGraph<br>
        <strong style="color:#c4b5fd;">LLM:</strong> Gemini 2.0 Flash<br>
        <strong style="color:#c4b5fd;">Embeddings:</strong> Google AI<br>
        <strong style="color:#c4b5fd;">Vector DB:</strong> FAISS<br>
        <strong style="color:#c4b5fd;">State:</strong> TypedDict + add_messages
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔄 Reset", use_container_width=True, type="secondary"):
        st.session_state.state = {
            "messages": [], "intent": "", "lead_info": {}, "lead_captured": False,
        }
        st.session_state.chat_history = []
        st.session_state.turn_count = 0
        st.rerun()


# ──────────────── Main Content ────────────────
with col_main:
    tab_chat, tab_pricing = st.tabs(["💬  Chat with Agent", "💎  Pricing & Plans"])
    
    # ── Chat Tab ──
    with tab_chat:
        st.markdown("""
        <div class="hero-section">
            <div class="hero-chip"><span class="chip-dot"></span> Live Agent</div>
            <h1><span class="gradient-text">Social-to-Lead</span><br>AI Agent</h1>
            <p class="hero-sub">Ask about AutoStream's video editing tools, explore plans, 
            or sign up — and watch the agentic workflow in action.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <p class="metric-value">3</p>
                <p class="metric-label">Intent Classes</p>
            </div>
            <div class="metric-card">
                <p class="metric-value">13</p>
                <p class="metric-label">KB Documents</p>
            </div>
            <div class="metric-card">
                <p class="metric-value">4</p>
                <p class="metric-label">Graph Nodes</p>
            </div>
            <div class="metric-card">
                <p class="metric-value">RAG</p>
                <p class="metric-label">Retrieval</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Success banner
        if lead_captured:
            li = st.session_state.state.get("lead_info", {})
            st.markdown(f"""
            <div class="capture-success">
                <h3>✅ Lead Successfully Captured</h3>
                <p>{li.get('name', '')} • {li.get('email', '')} • {li.get('platform', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])
        
        # Input
        if prompt := st.chat_input("Ask about AutoStream... (try: 'What are your pricing plans?')"):
            with st.chat_message("user", avatar="🙋"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.state["messages"].append(HumanMessage(content=prompt))
            st.session_state.turn_count += 1
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Agent thinking..."):
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
                        err = str(e)
                        if "RESOURCE_EXHAUSTED" in err or "429" in err:
                            st.error("⏳ API rate limit reached. Please wait a moment and try again.")
                        else:
                            st.error(f"Error: {err}")
            st.rerun()
    
    # ── Pricing Tab ──
    with tab_pricing:
        basic = KB_DATA["pricing"]["basic"]
        pro = KB_DATA["pricing"]["pro"]
        
        st.markdown("""
        <div class="pricing-header">
            <h2>Simple, <span class="gradient-text">Transparent</span> Pricing</h2>
            <p>Start creating professional videos today. No hidden fees.</p>
        </div>
        """, unsafe_allow_html=True)
        
        basic_features = "".join(
            f'<li><span class="check check-basic">✓</span>{f}</li>' for f in basic["features"]
        )
        pro_features = "".join(
            f'<li><span class="check check-pro">✓</span>{f}</li>' for f in pro["features"]
        )
        
        st.markdown(f"""
        <div class="pricing-grid">
            <div class="plan-card">
                <div class="plan-name">Basic</div>
                <div class="plan-price">{basic['price'].replace('/month','')}<span>/month</span></div>
                <div class="plan-desc">Great for creators just getting started with automated editing.</div>
                <ul class="plan-features">{basic_features}</ul>
                <div class="plan-btn plan-btn-outline">Get Started</div>
            </div>
            <div class="plan-card pro">
                <div class="plan-name">Pro</div>
                <div class="plan-price">{pro['price'].replace('/month','')}<span>/month</span></div>
                <div class="plan-desc">For serious creators who need unlimited power and AI features.</div>
                <ul class="plan-features">{pro_features}</ul>
                <div class="plan-btn plan-btn-primary">Start Free Trial →</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Policies
        policies = KB_DATA["policies"]
        policy_items = [
            ("Refund", policies["refund"]),
            ("Support", policies["support"]),
            ("Free Trial", policies["free_trial"]),
            ("Cancellation", policies["cancellation"]),
            ("Privacy", policies["data_privacy"]),
        ]
        
        cards = "".join(
            f'<div class="policy-card"><div class="policy-title">{title}</div><p class="policy-text">{text}</p></div>'
            for title, text in policy_items
        )
        
        st.markdown(f"""
        <div class="policy-section">
            <h3>Policies & Support</h3>
            <div class="policy-grid">{cards}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # FAQ
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Frequently Asked Questions")
        for faq in KB_DATA["faq"]:
            with st.expander(faq["question"]):
                st.write(faq["answer"])
