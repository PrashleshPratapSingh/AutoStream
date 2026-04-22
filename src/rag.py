"""
RAG (Retrieval-Augmented Generation) pipeline for the AutoStream AI Agent.
Loads the knowledge base, creates embeddings, and retrieves relevant context.
"""

import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# Path to the knowledge base file
KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base",
    "autostream_kb.json",
)

# Module-level cache for the vector store
_vector_store = None


def _load_knowledge_base() -> list[Document]:
    """
    Load the AutoStream knowledge base JSON and convert it into
    chunked Document objects suitable for embedding and retrieval.
    
    Returns:
        List of Document objects, each representing a chunk of knowledge.
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    
    documents = []
    
    # Company information
    company = kb["company"]
    documents.append(Document(
        page_content=(
            f"Company: {company['name']}\n"
            f"Tagline: {company['tagline']}\n"
            f"Description: {company['description']}"
        ),
        metadata={"source": "company_info"},
    ))
    
    # Pricing - Basic Plan
    basic = kb["pricing"]["basic"]
    features_str = "\n".join(f"  - {f}" for f in basic["features"])
    documents.append(Document(
        page_content=(
            f"AutoStream {basic['plan_name']}:\n"
            f"Price: {basic['price']} (billed {basic['billing']})\n"
            f"Features:\n{features_str}"
        ),
        metadata={"source": "pricing_basic"},
    ))
    
    # Pricing - Pro Plan
    pro = kb["pricing"]["pro"]
    features_str = "\n".join(f"  - {f}" for f in pro["features"])
    documents.append(Document(
        page_content=(
            f"AutoStream {pro['plan_name']}:\n"
            f"Price: {pro['price']} (billed {pro['billing']})\n"
            f"Features:\n{features_str}"
        ),
        metadata={"source": "pricing_pro"},
    ))
    
    # Pricing comparison
    documents.append(Document(
        page_content=(
            f"AutoStream Plan Comparison:\n"
            f"- {basic['plan_name']}: {basic['price']} — "
            f"{', '.join(basic['features'][:3])}\n"
            f"- {pro['plan_name']}: {pro['price']} — "
            f"{', '.join(pro['features'][:4])}\n"
            f"The Pro plan includes everything in Basic plus unlimited videos, "
            f"4K resolution, AI captions, and 24/7 support."
        ),
        metadata={"source": "pricing_comparison"},
    ))
    
    # Policies
    policies = kb["policies"]
    for policy_name, policy_text in policies.items():
        documents.append(Document(
            page_content=f"AutoStream {policy_name.replace('_', ' ').title()} Policy: {policy_text}",
            metadata={"source": f"policy_{policy_name}"},
        ))
    
    # FAQ
    for faq in kb["faq"]:
        documents.append(Document(
            page_content=f"Q: {faq['question']}\nA: {faq['answer']}",
            metadata={"source": "faq"},
        ))
    
    return documents


def _get_vector_store() -> FAISS:
    """
    Get or create the FAISS vector store with embedded knowledge base documents.
    Uses module-level caching to avoid re-embedding on every query.
    
    Returns:
        FAISS vector store instance.
    """
    global _vector_store
    
    if _vector_store is None:
        documents = _load_knowledge_base()
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        )
        _vector_store = FAISS.from_documents(documents, embeddings)
    
    return _vector_store


def retrieve(query: str, k: int = 3) -> str:
    """
    Retrieve relevant context from the knowledge base for a given query.
    
    Args:
        query: The user's question or search query.
        k: Number of top results to retrieve.
    
    Returns:
        Concatenated string of relevant knowledge base chunks.
    """
    store = _get_vector_store()
    results = store.similarity_search(query, k=k)
    
    context_parts = []
    for i, doc in enumerate(results, 1):
        context_parts.append(f"[Source {i}]\n{doc.page_content}")
    
    return "\n\n".join(context_parts)
