import os
import pickle

CHUNKS_DIR = "chunks"

metadata = []

print("📌 Scanning chunk directory...")

for filename in sorted(os.listdir(CHUNKS_DIR)):
    if filename.endswith(".txt"):
        full_path = os.path.join(CHUNKS_DIR, filename)

        # Each chunk is one row in FAISS → one metadata entry
        metadata.append(filename)

print(f"🔢 Total chunks found: {len(metadata)}")

# Save metadata
with open("faiss_filenames.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Saved faiss_filenames.pkl successfully!")
