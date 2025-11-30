import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

chunk_folder = "ncert_chunks"
embedding_file = "ncert_embeddings.pkl"

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings_dict = {}

for filename in os.listdir(chunk_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(chunk_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        embedding = model.encode(text)
        embeddings_dict[filename] = embedding

# Save embeddings
with open(embedding_file, "wb") as f:
    pickle.dump(embeddings_dict, f)

print(f"All embeddings saved in {embedding_file}")
