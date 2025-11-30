from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .rag_youtube import YouTubeRAG

yt_rag = YouTubeRAG()

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
