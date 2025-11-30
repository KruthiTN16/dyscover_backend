# ai_tutor/controller.py
"""
AI Tutor Controller (HYBRID MODE):
- NCERT RAG + YouTube RAG + general LLM knowledge
- Allows deeper, richer explanations (NOT restricted)
- Still domain-guarded and syllabus-aligned
"""

from typing import Dict, Any, List, Optional
import uuid
import time

# ================================================================
#  SESSION STORAGE
# ================================================================

SESSIONS: Dict[str, Dict[str, Any]] = {}

def new_session() -> str:
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "topic": None,
        "keywords": [],
        "mode": "chat",
        "current_video": None,
        "last_updated": time.time()
    }
    return sid


def set_session_context(session_id: str, topic=None, keywords=None, mode=None, current_video=None):
    if session_id not in SESSIONS:
        raise KeyError("session_id not found")

    s = SESSIONS[session_id]

    if topic is not None:
        s["topic"] = topic
    if keywords is not None:
        s["keywords"] = keywords
    if mode is not None:
        s["mode"] = mode
    if current_video is not None:
        s["current_video"] = current_video

    s["last_updated"] = time.time()
    return s


def get_session_context(session_id: str):
    if session_id not in SESSIONS:
        raise KeyError("session_id not found")
    return SESSIONS[session_id]


# ================================================================
#  NCERT RAG SERVICE
# ================================================================

try:
    from .rag_service import RAGService

    print("[ai_tutor.controller] Initializing NCERT RAGService…")
    try:
        rag_service = RAGService("ncert_faiss.index", "faiss_metadata.pkl")
        print("[ai_tutor.controller] NCERT RAGService loaded successfully.")
    except Exception as err:
        print(f"[NCERT] Failed init: {err}")
        rag_service = None

    def _search_ncert(query: str, k=8):
        if rag_service is None:
            return {"source": "ncert_stub", "results": []}
        try:
            return {"source": "ncert", "results": rag_service.search(query, k)}
        except Exception:
            return {"source": "ncert_stub", "results": []}

except:
    def _search_ncert(query: str, k=8):
        return {"source": "ncert_stub", "results": []}


# ================================================================
#  YOUTUBE RAG (FIXED)
# ================================================================

try:
    from .rag_youtube import YouTubeRAG
    yt_rag_backend = YouTubeRAG()

    def _search_youtube(query: str):
        try:
            videos = yt_rag_backend.search_youtube(query)
            return {"source": "youtube", "results": videos}
        except Exception:
            return {"source": "youtube_stub", "results": []}

except:
    def _search_youtube(query: str):
        return {"source": "youtube_stub", "results": []}


# ================================================================
#  LLM IMPORT
# ================================================================

try:
    from .llm_loader import ask_llm
except:
    def ask_llm(prompt: str):
        return "LLM unavailable."


# ================================================================
#  DOMAIN GUARD
# ================================================================

def domain_allowed(question: str, keywords: List[str]):
    q = question.lower()
    if not keywords:
        return True
    return any(k in q for k in keywords)


# ================================================================
#  SAFE CONVERSION HELPERS
# ================================================================

def yt_to_text(item: dict) -> str:
    if not isinstance(item, dict):
        return str(item)
    return (
        f"Video Title: {item.get('title','')}\n"
        f"Channel: {item.get('channel','')}\n"
        f"Link: {item.get('link','')}"
    )


# ================================================================
#  MAIN ORCHESTRATOR
# ================================================================

def handle_student_query(session_id: str, question: str, use_youtube=False):

    # Validate
    if session_id not in SESSIONS:
        return {"ok": False, "error": "Invalid session_id"}

    sess = SESSIONS[session_id]
    keywords = sess.get("keywords", [])

    # Domain guard
    if not domain_allowed(question, keywords):
        return {"ok": False, "error": "I can only answer questions related to the topic."}

    # Retrieval
    ncert = _search_ncert(question)
    youtube = _search_youtube(question) if use_youtube else None

    ncert_chunks = [str(x) for x in ncert.get("results", [])[:10]]
    youtube_chunks = [yt_to_text(v) for v in youtube.get("results", [])[:5]] if youtube else []

    # Background
    llm_background = (
        "General Science Knowledge:\n"
        "- Acids release H+ ions.\n"
        "- Bases release OH- ions.\n"
        "- Neutralisation forms salt and water.\n"
        "Use background ONLY to enrich explanations."
    )

    all_chunks = ncert_chunks + youtube_chunks + [llm_background]
    context_text = "\n\n".join(all_chunks)

    # NEW IMPROVED PROMPT (OPEN KNOWLEDGE MODE)
    prompt = f"""
You are an intelligent NCERT science tutor.

Use the HYBRID CONTEXT as reference, but you are allowed to use your own
scientific knowledge to explain concepts clearly, deeply, and with helpful examples.

INSTRUCTIONS:
- Use NCERT FIRST.
- Use YouTube content as supportive info.
- Use your own knowledge to expand, clarify, and give real-life examples.
- Give a detailed explanation suitable for Class 6–10 students.
- Structure the answer: introduction → explanation → examples → summary.
- Do NOT stay limited only to the context.
- Do NOT hallucinate wrong facts.

------------------------
HYBRID CONTEXT:
{context_text}
------------------------

Question: {question}

Write a clear, structured, detailed explanation:
"""

    try:
        answer = ask_llm(prompt)
    except Exception:
        answer = "AI Tutor failed to generate an answer."

    return {
        "ok": True,
        "session_id": session_id,
        "question": question,
        "session": sess,
        "ncert": ncert,
        "youtube": youtube,
        "answer": answer
    }


# ================================================================
#  LOCAL TEST
# ================================================================

if __name__ == "__main__":
    print("HYBRID Tutor Smoke Test Running…")
    sid = new_session()
    set_session_context(
        sid,
        topic="Acids, Bases and Salts",
        keywords=["acid", "base", "salt", "ph"],
        mode="chat"
    )
    print(handle_student_query(sid, "Explain acids and bases"))
