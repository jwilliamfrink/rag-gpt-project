import os
import openai
import faiss
import numpy as np
from fastapi import FastAPI

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Sample documents (should match what was indexed)
documents = [
    "AI is transforming the world.",
    "Machine learning improves automation.",
    "FAISS is great for vector search.",
    "Natural Language Processing enables chatbots.",
    "GPT models can generate human-like text."
]

# Initialize FastAPI
app = FastAPI()

# Function to get embeddings
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype='float32')

# Function to query FAISS and retrieve context
def search_faiss(query_text, k=3):
    query_embedding = get_embedding(query_text)
    distances, indices = index.search(np.array([query_embedding]), k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs

# Function to generate GPT response
def generate_gpt_response(query_text):
    retrieved_docs = search_faiss(query_text)
    prompt = f"Answer the following question using the provided context.\n\nContext:\n{retrieved_docs}\n\nQuestion: {query_text}\n\nAnswer:"

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# API endpoint to handle queries
@app.get("/query")
def query_rag(question: str):
    response = generate_gpt_response(question)
    return {"query": question, "response": response}