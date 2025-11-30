from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

# -------------------------------
# IMPORT RAG MODULES
# -------------------------------
from .rag_youtube import YouTubeRAG

# -------------------------------
# IMPORT AI TUTOR ORCHESTRATOR
# -------------------------------
from .controller import (
    new_session,
    set_session_context,
    get_session_context,
    handle_student_query
)

# -------------------------------
# INITIALIZE YOUTUBE RAG
# -------------------------------
yt_rag = YouTubeRAG()


# ==========================================================
#  YOUTUBE ENDPOINTS (ALREADY WORKING)
# ==========================================================

class YouTubeSearchView(APIView):
    def get(self, request):
        query = request.query_params.get("q")
        if not query:
            return Response({"detail": "Missing ?q= parameter"}, status=400)
        videos = yt_rag.search_youtube(query)
        return Response(videos)


class YouTubePrepareView(APIView):
    def post(self, request):
        video_id = request.data.get("video_id")
        if not video_id:
            return Response({"detail": "video_id required"}, status=400)

        yt_rag.prepare_video(video_id)
        return Response({"status": "prepared"})


class YouTubeAskView(APIView):
    def post(self, request):
        video_id = request.data.get("video_id")
        question = request.data.get("question")
        timestamp = request.data.get("timestamp")

        if not video_id or not question:
            return Response({"detail": "video_id and question are required"}, status=400)

        answer = yt_rag.ask_video(question, video_id, timestamp)
        return Response({"answer": answer})


# ==========================================================
#  NEW AI TUTOR ENDPOINTS (PUBLIC)
# ==========================================================

class StartSessionView(APIView):
    permission_classes = [AllowAny]   # <--- allows public access

    """
    GET /ai_tutor/start_session/
    Returns a new session_id.
    """
    def get(self, request):
        sid = new_session()
        return Response({"session_id": sid})


class SetContextView(APIView):
    permission_classes = [AllowAny]   # <--- allows public access

    """
    POST /ai_tutor/set_context/
    Body:
      {
        "session_id": "...",
        "topic": "Acids, Bases and Salts",
        "keywords": ["acid", "base", "salt", "ph"],
        "mode": "chat"
      }
    """
    def post(self, request):
        data = request.data

        sid = data.get("session_id")
        if not sid:
            return Response({"detail": "session_id required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            updated = set_session_context(
                sid,
                topic=data.get("topic"),
                keywords=data.get("keywords"),
                mode=data.get("mode")
            )
            return Response({"ok": True, "session": updated})
        except KeyError:
            return Response({"detail": "Invalid session_id"}, status=status.HTTP_404_NOT_FOUND)


class ChatView(APIView):
    permission_classes = [AllowAny]   # <--- allows public access

    """
    POST /ai_tutor/chat/
    Body:
      {
        "session_id": "...",
        "question": "What is an acid?",
        "use_youtube": false
      }
    """
    def post(self, request):
        data = request.data

        sid = data.get("session_id")
        question = data.get("question")

        if not sid or not question:
            return Response(
                {"detail": "session_id and question are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        use_youtube = bool(data.get("use_youtube", False))
        result = handle_student_query(sid, question, use_youtube=use_youtube)

        if not result.get("ok"):
            # domain guard or invalid session
            return Response({"detail": result.get("error")}, status=status.HTTP_400_BAD_REQUEST)

        return Response(result)
