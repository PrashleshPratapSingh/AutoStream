"""
Configuration module for the AutoStream AI Agent.
Initializes the LLM (Gemini 2.0 Flash) and loads environment variables.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Gemini 2.0 Flash LLM instance.
    
    Args:
        temperature: Controls randomness. Lower = more deterministic.
    
    Returns:
        ChatGoogleGenerativeAI instance.
    
    Raises:
        ValueError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. "
            "Set it in your .env file. "
            "Get a key at: https://aistudio.google.com/apikey"
        )
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=temperature,
    )
