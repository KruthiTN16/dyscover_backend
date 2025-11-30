import streamlit as st
from rag_service import RAGService
from ollama import chat

# Initialize RAG
st.set_page_config(page_title="NCERT RAG Assistant", layout="wide")
rag = RAGService()

st.title("📚 NCERT RAG Assistant")
st.markdown("Ask any question from NCERT textbooks. You will get **retrieved chunks**, **context-based answer**, and **independent answer**.")

# Input
question = st.text_input("Ask NCERT →", "")

if st.button("Get Answer") and question.strip() != "":
    st.subheader("🔹 Retrieved Chunks")
    context_chunks = rag.search(question, k=5)
    for i, chunk_text in enumerate(context_chunks):
        chunk_id = rag.chunk_ids[i]
        snippet = chunk_text[:200] + ("..." if len(chunk_text) > 200 else "")
        st.markdown(f"**{i+1}. {chunk_id}**: {snippet}")

    # Context-based answer
    st.subheader("🤖 Answer Using Context")
    with st.spinner("LLM is generating answer using context..."):
        prompt_with_context = f"Answer the following question using the given NCERT context. " \
                              f"Provide a clear, concise, and student-friendly explanation.\n\n" \
                              f"Context:\n{chr(10).join(context_chunks)}\n\nQuestion: {question}\nAnswer:"
        response_context = chat(
            model="llama3:latest",
            messages=[{"role": "user", "content": prompt_with_context}]
        )
        answer_with_context = response_context.message.content
        st.write(answer_with_context)

    # Independent answer
    st.subheader("🤖 Independent Answer")
    with st.spinner("LLM is generating independent answer..."):
        prompt_independent = f"Answer the following question independently, without using any external context. " \
                             f"Explain clearly in a way a school student can understand, with examples if needed.\n\n" \
                             f"Question: {question}\nAnswer:"
        response_independent = chat(
            model="llama3:latest",
            messages=[{"role": "user", "content": prompt_independent}]
        )
        answer_independent = response_independent.message.content
        st.write(answer_independent)
