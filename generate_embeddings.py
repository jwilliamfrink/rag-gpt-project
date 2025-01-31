import os
import openai
import faiss
import numpy as np

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sample documents to embed
documents = [
    "AI is transforming the world.",
    "Machine learning improves automation.",
    "FAISS is great for vector search.",
    "Natural Language Processing enables chatbots.",
    "GPT models can generate human-like text."
]

# Function to get embeddings from OpenAI
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype='float32')

# Convert all documents to embeddings
embedding_dim = 1536  # OpenAI embeddings have 1536 dimensions
vectors = np.array([get_embedding(doc) for doc in documents])

# Create FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(vectors)

# Save the index
faiss.write_index(index, "faiss_index.bin")

print(f"Stored {index.ntotal} text embeddings in FAISS.")

# Load the FAISS index
index = faiss.read_index("faiss_index.bin")

# Function to query FAISS
def search_faiss(query_text, k=3):
    query_embedding = get_embedding(query_text)
    distances, indices = index.search(np.array([query_embedding]), k)
    
    print("\nQuery:", query_text)
    print("Most similar documents:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]:.4f})")

# Test the search function
test_query = "How does AI impact the world?"
search_faiss(test_query)