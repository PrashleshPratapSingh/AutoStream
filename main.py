"""
AutoStream AI Agent - CLI Entry Point

Interactive command-line interface for the AutoStream agent.
Demonstrates the full conversation flow: greeting -> RAG retrieval ->
intent detection -> lead qualification -> tool execution.

Usage:
    python main.py
"""

from langchain_core.messages import HumanMessage
from src.agent import build_agent


def main():
    """Run the interactive CLI agent."""
    print("\n" + "=" * 60)
    print("  AutoStream AI Agent - Social-to-Lead Workflow")
    print("  Powered by LangGraph + Gemini 2.0 Flash")
    print("=" * 60)
    print("  Type 'quit' to exit | 'reset' to restart")
    print("=" * 60 + "\n")
    
    print("Initializing agent...")
    agent = build_agent()
    
    state = {
        "messages": [],
        "intent": "",
        "lead_info": {},
        "lead_captured": False,
    }
    
    print("Agent ready!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break
        if user_input.lower() == "reset":
            state = {"messages": [], "intent": "", "lead_info": {}, "lead_captured": False}
            print("\nConversation reset.\n")
            continue
        
        state["messages"].append(HumanMessage(content=user_input))
        
        try:
            result = agent.invoke(state)
            state["messages"] = result["messages"]
            state["intent"] = result.get("intent", "")
            state["lead_info"] = result.get("lead_info", state["lead_info"])
            state["lead_captured"] = result.get("lead_captured", False)
            
            last_msg = result["messages"][-1]
            print(f"\nAgent: {last_msg.content}")
            print(f"  [intent={state['intent']} | lead_info={state['lead_info']} | captured={state['lead_captured']}]\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
