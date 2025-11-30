import faiss
import pickle
import numpy as np

embedding_file = "ncert_embeddings.pkl"
index_file = "ncert_faiss.index"

# Load embeddings
with open(embedding_file, "rb") as f:
    embeddings_dict = pickle.load(f)

# Prepare data for FAISS
filenames = list(embeddings_dict.keys())
vectors = np.array([embeddings_dict[f] for f in filenames], dtype='float32')

# Build FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Save FAISS index and filenames mapping
faiss.write_index(index, index_file)
with open("faiss_filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print(f"FAISS index saved as {index_file} and filenames mapping saved.")
