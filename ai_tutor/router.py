# ai_tutor/router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai_tutor.controller import (
    new_session,
    set_session_context,
    get_session_context,
    handle_student_query
)

router = APIRouter(prefix="/ai_tutor", tags=["AI Tutor"])

# Request models
class ContextUpdateRequest(BaseModel):
    session_id: str
    topic: str
    keywords: list[str]
    mode: str = "chat"   # chat or video

class ChatRequest(BaseModel):
    session_id: str
    question: str
    use_youtube: bool = False


# Route 1 — start session
@router.get("/start_session")
def start_session():
    sid = new_session()
    return {"session_id": sid}


# Route 2 — set context (topic + keywords)
@router.post("/set_context")
def set_context(req: ContextUpdateRequest):
    try:
        updated = set_session_context(
            req.session_id,
            topic=req.topic,
            keywords=req.keywords,
            mode=req.mode
        )
        return {"ok": True, "session": updated}
    except KeyError:
        raise HTTPException(status_code=404, detail="Invalid session_id")


# Route 3 — student chat query
@router.post("/chat")
def chat(req: ChatRequest):
    result = handle_student_query(req.session_id, req.question, req.use_youtube)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result
