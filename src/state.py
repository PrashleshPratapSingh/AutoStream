"""
State management for the AutoStream AI Agent.
Defines the typed state schema used by LangGraph.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class LeadInfo(TypedDict, total=False):
    """Information collected during lead qualification."""
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    """
    Central state for the AutoStream conversational agent.
    
    Attributes:
        messages: Conversation history (auto-managed by LangGraph's add_messages reducer).
        intent: Classified intent of the latest user message.
        lead_info: Accumulated lead details across conversation turns.
        lead_captured: Flag to prevent duplicate tool invocations.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: LeadInfo
    lead_captured: bool
