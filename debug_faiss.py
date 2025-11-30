import pickle
import faiss

# Load FAISS
index = faiss.read_index("ncert_faiss.index")
print("FAISS index size:", index.ntotal)

# Load metadata
with open("faiss_filenames.pkl", "rb") as f:
    meta = pickle.load(f)

print("Metadata size:", len(meta))

print("First few keys:", list(meta.keys())[:5])
