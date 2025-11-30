import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Load FAISS index and filenames
index = faiss.read_index("ncert_faiss.index")
with open("faiss_filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Folder where chunks are stored
chunk_folder = "ncert_chunks"

def get_top_k_chunks(question, k=3):
    q_vector = model.encode(question).astype('float32')
    distances, indices = index.search(np.array([q_vector]), k)
    results = []
    for idx in indices[0]:
        chunk_file = filenames[idx]
        with open(f"{chunk_folder}/{chunk_file}", "r", encoding="utf-8") as f:
            results.append(f.read())
    return results

# Example usage
if __name__ == "__main__":
    question = "What are the types of forces?"
    top_chunks = get_top_k_chunks(question)
    for i, c in enumerate(top_chunks):
        print(f"\n---Chunk {i+1}---\n")
        print(c)
