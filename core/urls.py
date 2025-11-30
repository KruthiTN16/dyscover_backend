# core/urls.py
from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from .views import (
    RegisterView,
    MyTokenObtainPairView,
    ParentAssessmentSubmitView,
    EducatorStudentView,
    SearchYouTube,
    AskTutor,
)

urlpatterns = [
    # Authentication
    path("register/", RegisterView.as_view(), name="register"),
    path("token/", MyTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),

    # Assessment
    path("assessment/submit/", ParentAssessmentSubmitView.as_view(), name="assessment_submit"),
    path("student/<int:student_id>/summary/", EducatorStudentView.as_view(), name="student_summary"),

    # ⭐ AI Tutor Endpoints ⭐
    path("ai-tutor/search/", SearchYouTube.as_view(), name="ai_tutor_search"),
    path("ai-tutor/ask/", AskTutor.as_view(), name="ai_tutor_ask"),
]
