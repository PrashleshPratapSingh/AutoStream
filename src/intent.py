"""
Intent classification module for the AutoStream AI Agent.
Uses LLM-based classification to determine user intent from conversation context.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm


# System prompt for intent classification with few-shot examples
INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for AutoStream, a SaaS video editing platform.

Analyze the user's latest message in the context of the conversation and classify their intent into EXACTLY one of these categories:

1. **greeting** — Casual hello, small talk, or general conversation starters.
2. **product_inquiry** — Questions about AutoStream's features, pricing, plans, policies, comparisons, or how the product works.
3. **high_intent** — The user is ready to sign up, wants to try a plan, expresses clear buying intent, or is providing their personal details (name, email, platform) for sign-up.

RULES:
- If the user is providing their name, email, or platform after being asked, classify as "high_intent".
- If the user says things like "I want to try", "sign me up", "I'm interested in the Pro plan", "let's go", classify as "high_intent".
- If the user asks "what are your plans?" or "how much does it cost?", classify as "product_inquiry".
- If unsure, lean toward "product_inquiry" over "greeting".

Respond with ONLY the intent label (one word): greeting, product_inquiry, or high_intent"""


def classify_intent(messages: list, llm=None) -> str:
    """
    Classify the intent of the latest user message given conversation history.
    
    Args:
        messages: Full conversation history (list of BaseMessage objects).
        llm: Optional LLM instance. If None, creates a new one.
    
    Returns:
        One of: 'greeting', 'product_inquiry', 'high_intent'
    """
    if llm is None:
        llm = get_llm(temperature=0.0)  # Deterministic for classification
    
    # Build the classification request with conversation context
    classification_messages = [
        SystemMessage(content=INTENT_CLASSIFICATION_PROMPT),
    ]
    
    # Include last 6 messages for context (to handle intent shifts)
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    classification_messages.extend(recent_messages)
    
    # Ask the LLM to classify
    response = llm.invoke(classification_messages)
    intent = response.content.strip().lower()
    
    # Validate the response
    valid_intents = {"greeting", "product_inquiry", "high_intent"}
    if intent not in valid_intents:
        # Fallback: try to extract a valid intent from the response
        for valid in valid_intents:
            if valid in intent:
                return valid
        return "product_inquiry"  # Safe default
    
    return intent
