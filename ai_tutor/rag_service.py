# rag_service.py
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ollama import chat
from ai_tutor import tutor_retrieval

class RAGService:
    def __init__(self, index_path="ncert_faiss.index", metadata_path="faiss_metadata.pkl"):
        """
        index_path: path to FAISS index
        metadata_path: path to pickle file containing chunk_id -> chunk_text
        """
        print("🔍 Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        print("📄 Loading metadata...")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)  # dict: chunk_id -> chunk_text

        # Keep a list of chunk IDs for indexing
        self.chunk_ids = list(self.metadata.keys())

        print("🤖 Loading embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        print("✅ RAG system initialized successfully!\n")

    def search(self, query, k=5):
        """Search FAISS index and return top-k text chunks as strings."""
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vec, k)

        chunks = []
        for idx in indices[0]:
            chunk_id = self.chunk_ids[int(idx)]
            chunk_text = self.metadata[chunk_id]
            chunks.append(chunk_text)

        return chunks

    def ask(self, question, top_k=5):
        """Retrieve context and query LLM (Ollama)."""
        context_chunks = self.search(question, k=top_k)
        full_context = "\n\n".join(context_chunks)

        prompt = f"Use the context to answer.\n\nContext:\n{full_context}\n\nQuestion: {question}\nAnswer:"

        response = chat(
            model="llama3:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content
