# ==========================================
# rag.py
# Zyro Dynamics HR Help Desk
# ==========================================

import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from prompt import HR_PROMPT


# ==========================================
# API Keys
# ==========================================

import streamlit as st

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

# ==========================================
# Embedding Model
# ==========================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("✅ Embedding model loaded")


# ==========================================
# Load FAISS Vector Store
# ==========================================

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True,
)

print("✅ FAISS Vector Store Loaded")

# ==========================================
# Similarity Threshold
# ==========================================

SIMILARITY_THRESHOLD = 0.80

print(f"✅ Similarity Threshold: {SIMILARITY_THRESHOLD}")

# ==========================================
# Retriever
# ==========================================

# ==========================================
# Retriever (Optimized)
# ==========================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 7,
        "fetch_k": 30,
        "lambda_mult": 0.5,
    },
)

print("✅ Optimized MMR Retriever Ready")


# ==========================================
# Groq LLM
# ==========================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=1024,
)

print("✅ Groq LLM Loaded")

# ==========================================
# Helper Function
# ==========================================

def format_docs(docs):

    formatted_docs = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        formatted_docs.append(
f"""
==================================================
Document {i}

Source : {source}

Page   : {page}

Policy Content

{doc.page_content.strip()}
==================================================
"""
        )

    return "\n".join(formatted_docs)

# ==========================================
# RAG Chain
# ==========================================

rag_chain = (
    HR_PROMPT
    | llm
    | StrOutputParser()
)

print("✅ Optimized RAG Chain Ready")
# ==========================================
# Main Chat Function
# ==========================================

# ============================================
# Chatbot Function (Optimized)
# ============================================

SIMILARITY_THRESHOLD = 0.80

def hr_chatbot(question):

    # Retrieve with similarity scores
    results = vectorstore.similarity_search_with_score(
        question,
        k=7
    )

    top_doc, top_score = results[0]

    print(f"Question : {question}")
    print(f"Top Score: {top_score}")

    # Refuse if similarity is poor
    if top_score > SIMILARITY_THRESHOLD:

        return {
            "answer": "I can only answer HR-related questions based on Zyro Dynamics policy documents.",
            "sources": []
        }

    # Retrieve documents using MMR
    docs = retriever.invoke(question)

    # Build context
    context = format_docs(docs)

    # Generate answer
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    # Build sources
    sources = []

    seen = set()

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        key = (source, page)

        if key not in seen:

            seen.add(key)

            sources.append({
                "document": source,
                "page": page
            })

    return {
        "answer": answer,
        "sources": sources
    }

# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    question = "How many earned leaves are allowed?"

    response = ask_hr_question(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")

    for source in response["sources"]:
        print(
            source["document"],
            "- Page",
            source["page"]
        )
