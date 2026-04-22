"""
Configuration module for the AutoStream AI Agent.
Initializes the LLM (Gemini 1.5 Flash) and loads environment variables.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


def get_llm(temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Gemini 1.5 Flash LLM instance.
    
    Args:
        temperature: Controls randomness in responses. Lower = more deterministic.
    
    Returns:
        ChatGoogleGenerativeAI instance configured with Gemini 1.5 Flash.
    
    Raises:
        ValueError: If GOOGLE_API_KEY is not set in environment.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set it in your .env file. "
            "Get your key at: https://aistudio.google.com/apikey"
        )
    
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )
