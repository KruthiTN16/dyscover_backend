from rag_service import RAGService
from ollama import chat

# Initialize RAG service
rag = RAGService()

while True:
    q = input("\nAsk NCERT → ")
    if q.lower() == "exit":
        break

    # --- Section 1: Retrieved Chunks ---
    context_chunks = rag.search(q, k=5)
    print("\n================ Retrieved Chunks =================\n")
    for i, chunk_text in enumerate(context_chunks):
        chunk_id = rag.chunk_ids[i]
        snippet = chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
        print(f"{i+1}. {chunk_id}: {snippet}\n")

    # --- Section 2: LLM answer using context ---
    print("\n🤖 LLM is generating answer using context...\n")
    prompt_with_context = f"Answer the following question using the given NCERT context. " \
                          f"Provide a clear, concise, and student-friendly explanation.\n\n" \
                          f"Context:\n{chr(10).join(context_chunks)}\n\nQuestion: {q}\nAnswer:"
    
    response_context = chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt_with_context}]
    )
    answer_with_context = response_context.message.content
    print("\n================ Answer Using Context ================\n")
    print(answer_with_context)

    # --- Section 3: LLM independent answer ---
    print("\n🤖 LLM is generating independent answer...\n")
    prompt_independent = f"Answer the following question independently, without using any external context. " \
                         f"Explain clearly in a way a school student can understand, with examples if needed.\n\n" \
                         f"Question: {q}\nAnswer:"
    
    response_independent = chat(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt_independent}]
    )
    answer_independent = response_independent.message.content
    print("\n================ Independent Answer =================\n")
    print(answer_independent)
    print("\n=====================================================\n")
