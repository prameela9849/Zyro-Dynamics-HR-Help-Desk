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

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7,
    },
)

print("✅ Retriever Ready")


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

    context = []

    for doc in docs:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context.append(
            f"""
Document: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(context)


# ==========================================
# RAG Chain
# ==========================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | HR_PROMPT
    | llm
    | StrOutputParser()
)

print("✅ RAG Chain Ready")

# ==========================================
# Main Chat Function
# ==========================================

def ask_hr_question(question):

    # Retrieve documents with scores
    results = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    top_doc, top_score = results[0]

    # Print score (for tuning)
    print("\n================================")
    print("Question :", question)
    print("Top Score:", top_score)
    print("================================")

    # Threshold check
    if top_score > SIMILARITY_THRESHOLD:

        return {
            "answer": "I can only answer HR-related questions based on Zyro Dynamics policy documents.",
            "sources": []
        }

    # Continue with MMR retrieval
    docs = retriever.invoke(question)

    # Generate answer
    answer = rag_chain.invoke(question)

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