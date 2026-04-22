"""
Intent classification is handled directly in the agent's unified LLM call.
This module is kept for reference but is not used at runtime.

The agent uses structured JSON output to classify intent as part of its
single-call-per-turn architecture, eliminating the need for a separate
classification step.

Supported intents:
  - greeting: Casual greetings, small talk
  - product_inquiry: Questions about features, pricing, plans, policies
  - high_intent: User wants to sign up, try a plan, or is providing details
"""
