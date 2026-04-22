"""
RAG (Retrieval-Augmented Generation) pipeline for the AutoStream AI Agent.
Loads knowledge base, creates embeddings, and retrieves relevant context.
"""

import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base",
    "autostream_kb.json",
)

_vector_store = None


def _load_knowledge_base() -> list[Document]:
    """
    Load AutoStream KB and convert to chunked Documents for embedding.
    Each chunk is a semantically meaningful unit for retrieval.
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    
    documents = []
    
    # Company info
    company = kb["company"]
    documents.append(Document(
        page_content=(
            f"Company: {company['name']}\n"
            f"Tagline: {company['tagline']}\n"
            f"Description: {company['description']}"
        ),
        metadata={"source": "company_info"},
    ))
    
    # Basic Plan
    basic = kb["pricing"]["basic"]
    features = "\n".join(f"  - {f}" for f in basic["features"])
    documents.append(Document(
        page_content=(
            f"AutoStream {basic['plan_name']}:\n"
            f"Price: {basic['price']}\n"
            f"Features:\n{features}"
        ),
        metadata={"source": "pricing_basic"},
    ))
    
    # Pro Plan
    pro = kb["pricing"]["pro"]
    features = "\n".join(f"  - {f}" for f in pro["features"])
    documents.append(Document(
        page_content=(
            f"AutoStream {pro['plan_name']}:\n"
            f"Price: {pro['price']}\n"
            f"Features:\n{features}"
        ),
        metadata={"source": "pricing_pro"},
    ))
    
    # Plan Comparison
    documents.append(Document(
        page_content=(
            f"AutoStream Plan Comparison:\n"
            f"- {basic['plan_name']}: {basic['price']} - "
            f"{', '.join(basic['features'][:3])}\n"
            f"- {pro['plan_name']}: {pro['price']} - "
            f"{', '.join(pro['features'][:4])}\n"
            f"Pro includes everything in Basic plus unlimited videos, "
            f"4K resolution, AI captions, and 24/7 support."
        ),
        metadata={"source": "pricing_comparison"},
    ))
    
    # Policies
    for name, text in kb["policies"].items():
        documents.append(Document(
            page_content=f"AutoStream {name.replace('_', ' ').title()} Policy: {text}",
            metadata={"source": f"policy_{name}"},
        ))
    
    # FAQ
    for faq in kb["faq"]:
        documents.append(Document(
            page_content=f"Q: {faq['question']}\nA: {faq['answer']}",
            metadata={"source": "faq"},
        ))
    
    return documents


def get_vector_store() -> FAISS:
    """
    Get or create the FAISS vector store (cached after first call).
    """
    global _vector_store
    if _vector_store is None:
        documents = _load_knowledge_base()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        _vector_store = FAISS.from_documents(documents, embeddings)
    return _vector_store


def retrieve(query: str, k: int = 3) -> str:
    """
    Retrieve relevant context from the knowledge base.
    
    Args:
        query: User's question.
        k: Number of top results.
    
    Returns:
        Concatenated relevant chunks as context string.
    """
    store = get_vector_store()
    results = store.similarity_search(query, k=k)
    return "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(results))
