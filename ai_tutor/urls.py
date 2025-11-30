from django.urls import path

from .views import (
    # YouTube endpoints
    YouTubeSearchView,
    YouTubePrepareView,
    YouTubeAskView,

    # AI Tutor endpoints
    StartSessionView,
    SetContextView,
    ChatView
)

urlpatterns = [
    # -------------------------
    # YouTube RAG API
    # -------------------------
    path("search/", YouTubeSearchView.as_view()),
    path("prepare/", YouTubePrepareView.as_view()),
    path("ask/", YouTubeAskView.as_view()),

    # -------------------------
    # AI Tutor API
    # -------------------------
    path("start_session/", StartSessionView.as_view()),
    path("set_context/", SetContextView.as_view()),
    path("chat/", ChatView.as_view()),
]
