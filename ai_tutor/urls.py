from django.urls import path
from .views import YouTubeSearchView, YouTubePrepareView, YouTubeAskView

urlpatterns = [
    path("search/", YouTubeSearchView.as_view()),
    path("prepare/", YouTubePrepareView.as_view()),
    path("ask/", YouTubeAskView.as_view()),
]
