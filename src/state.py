"""
State management for the AutoStream AI Agent.
Defines the typed state schema used by LangGraph to maintain conversation context.
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
        messages: Conversation history with automatic message merging via add_messages.
        intent: The classified intent of the latest user message.
                One of: 'greeting', 'product_inquiry', 'high_intent', or empty string.
        lead_info: Dictionary holding collected lead details (name, email, platform).
        lead_captured: Flag indicating whether the lead has been successfully captured.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_info: LeadInfo
    lead_captured: bool
