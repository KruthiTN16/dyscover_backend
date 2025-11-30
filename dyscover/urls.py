"""
URL configuration for dyscover project.
"""

from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse


# --- Root Home Endpoint (Prevents 404 on '/') ---
def home(request):
    return JsonResponse({
        "status": "Dyscover backend running",
        "message": "Welcome to the Dyscover API"
    })


urlpatterns = [
    path("", home, name="home"),                    # ⭐ root path: http://127.0.0.1:8000/
    path("admin/", admin.site.urls),               # Django admin
    path("api/core/", include("core.urls")),       # Authentication + Assessment API
    path("ai_tutor/", include("ai_tutor.urls")),   # YouTube RAG AI Tutor API
]
