"""
AutoStream AI Agent — CLI Entry Point

Interactive command-line interface for conversing with the AutoStream
conversational AI agent. Demonstrates intent detection, RAG-powered
knowledge retrieval, and lead capture workflow.

Usage:
    python main.py
"""

from langchain_core.messages import HumanMessage
from src.agent import build_agent


def print_banner():
    """Display the application banner."""
    print("\n" + "=" * 60)
    print("  AutoStream AI Agent  -  Social-to-Lead Workflow")
    print("  Powered by LangGraph + Gemini 1.5 Flash")
    print("=" * 60)
    print("  Type your message to chat with the agent.")
    print("  Type 'quit' or 'exit' to end the conversation.")
    print("  Type 'reset' to start a new conversation.")
    print("=" * 60 + "\n")


def main():
    """Run the interactive CLI agent loop."""
    print_banner()
    
    # Build the LangGraph agent
    print("[...] Initializing agent and knowledge base...")
    agent = build_agent()
    
    # Initialize state
    state = {
        "messages": [],
        "intent": "",
        "lead_info": {},
        "lead_captured": False,
    }
    
    print("[OK] Agent ready! Start chatting.\n")
    
    turn_count = 0
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Thanks for chatting with AutoStream.")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye! Thanks for chatting with AutoStream.")
            break
        
        if user_input.lower() == "reset":
            state = {
                "messages": [],
                "intent": "",
                "lead_info": {},
                "lead_captured": False,
            }
            turn_count = 0
            print("\n[RESET] Conversation reset. Start fresh!\n")
            continue
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))
        turn_count += 1
        
        # Run the agent
        try:
            result = agent.invoke(state)
            
            # Update state with agent's response
            state["messages"] = result["messages"]
            state["intent"] = result.get("intent", state["intent"])
            state["lead_info"] = result.get("lead_info", state["lead_info"])
            state["lead_captured"] = result.get("lead_captured", state["lead_captured"])
            
            # Display the agent's response
            if result["messages"]:
                last_message = result["messages"][-1]
                print(f"\nAutoStream: {last_message.content}\n")
            
            # Debug info (optional, can be removed for clean demo)
            print(f"   [Debug] Intent: {state['intent']} | "
                  f"Lead Info: {state['lead_info']} | "
                  f"Captured: {state['lead_captured']} | "
                  f"Turn: {turn_count}")
            print()
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("   Please try again or type 'reset' to start over.\n")


if __name__ == "__main__":
    main()
