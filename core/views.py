# core/views.py
from rest_framework import generics, status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken

from django.contrib.auth import get_user_model
from .serializers import (
    RegisterSerializer,
    MyTokenObtainPairSerializer,
    AssessmentSubmitSerializer,
    RiskResultSerializer,
    ConcernAnalysisSerializer,
    UserSerializer,
)
from .models import Assessment, RiskResult, ConcernAnalysis

import math, re
from collections import Counter

User = get_user_model()

# -------------------------------------------------
# Authentication + Registration
# -------------------------------------------------
class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        data = {
            "user": UserSerializer(user).data,
            "refresh": str(refresh),
            "access": str(refresh.access_token),
        }
        return Response(data, status=status.HTTP_201_CREATED)


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


# -------------------------------------------------
# Concern Text NLP (keywords + summary)
# -------------------------------------------------
def analyze_concern_text(text: str):
    if not text:
        return {"summary": "", "keywords": [], "sentiment": "neutral"}

    txt = text.lower()
    words = re.findall(r"\b[a-z]{4,}\b", txt)

    if not words:
        return {"summary": text[:200], "keywords": [], "sentiment": "neutral"}

    counts = Counter(words)
    keywords = [w for w, _ in counts.most_common(5)]

    positive_cues = sum(1 for w in words if w in {"good", "improve", "improved", "better"})
    negative_cues = sum(
        1 for w in words if w in
        {"struggl", "struggle", "difficul", "frustrat", "avoid", "problem", "worri"}
    )

    sentiment = (
        "negative" if negative_cues > positive_cues
        else "positive" if positive_cues > negative_cues
        else "neutral"
    )

    summary = " ".join(words[:20])

    return {"summary": summary, "keywords": keywords, "sentiment": sentiment}


# -------------------------------------------------
# Risk Probability (Logistic Demo Model)
# -------------------------------------------------
QUESTION_ORDER = [
    "reads_slowly","spelling_difficulty","memory_issues","phonological_awareness",
    "misreads_words","avoids_reading","letter_confusion","rapid_naming",
    "copying_difficulty","frustration_reading",
]

BASE_WEIGHTS = [0.85,0.7,0.5,0.75,0.7,0.35,0.9,0.6,0.45,0.5]


def compute_risk_probability(answer_dict):
    vec = []
    for key in QUESTION_ORDER:
        try:
            vec.append(float(answer_dict.get(key, 0)))
        except:
            vec.append(0.0)

    linear = sum([a*b for a,b in zip(vec, BASE_WEIGHTS)])
    denom = sum(BASE_WEIGHTS) * 3.0
    offset = sum(BASE_WEIGHTS) * 1.5

    linear_norm = (linear - offset) / denom
    prob = 1.0 / (1.0 + math.exp(-4.0 * linear_norm))
    return max(0.0, min(1.0, prob))


# -------------------------------------------------
# Parent Assessment Submit
# -------------------------------------------------
class ParentAssessmentSubmitView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data
        parent = request.user

        student_name = data.get("student_name") or "Unknown"
        answers = data.get("answers", {})

        if not isinstance(answers, dict):
            return Response({"detail": "answers must be a dict"}, status=400)

        assessment = Assessment.objects.create(
            questionnaire=None,
            student_name=student_name,
            parent=parent,
        )

        prob = compute_risk_probability(answers)
        risk_label = "At Risk" if prob >= 0.5 else "Low Risk"
        risk_level = "High" if prob >= 0.7 else ("Moderate" if prob >= 0.5 else "Low")

        RiskResult.objects.create(
            assessment=assessment,
            risk_score=prob,
            risk_level=risk_level,
            method="logistic_demo_v1"
        )

        parent_concern = data.get("parent_concern", "")
        nlp = analyze_concern_text(parent_concern)

        ConcernAnalysis.objects.create(
            assessment=assessment,
            summary=nlp.get("summary",""),
            keywords=",".join(nlp.get("keywords",[])),
            sentiment=nlp.get("sentiment","")
        )

        recommended_experts = [
            {"name": "Dr. Priya Sharma", "specialty": "Educational Psychologist", "contact": "example@clinic.org"},
            {"name": "Mr. R. Nair", "specialty": "Speech & Language Therapist", "contact": "example2@clinic.org"},
        ]

        return Response({
            "assessment_id": assessment.id,
            "student_name": student_name,
            "risk_probability": round(prob, 3),
            "risk_label": risk_label,
            "risk_level": risk_level,
            "nlp_summary": nlp["summary"],
            "nlp_keywords": nlp["keywords"],
            "nlp_sentiment": nlp["sentiment"],
            "recommended_experts": recommended_experts,
        }, status=201)


# -------------------------------------------------
# Educator View Assessment
# -------------------------------------------------
class EducatorStudentView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, student_id):
        assessments = Assessment.objects.filter(id=student_id)
        if not assessments.exists():
            return Response({"detail": "No assessment found"}, status=404)

        a = assessments.first()
        risks = RiskResultSerializer(a.risk_results.all().order_by("-created_at"), many=True)
        concerns = ConcernAnalysisSerializer(a.concern_analyses.all().order_by("-created_at"), many=True)

        return Response({
            "assessment": AssessmentSubmitSerializer(a).data,
            "risks": risks.data,
            "concerns": concerns.data,
        })


# =================================================
#                ⭐ AI TUTOR API ⭐
# =================================================

from ai_tutor.rag_youtube import YouTubeRAG
yt_rag = YouTubeRAG()


# --- 1) Search YouTube ---
class SearchYouTube(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        query = request.data.get("query", "")
        if not query:
            return Response({"error": "query is required"}, status=400)

        results = yt_rag.search_youtube(query)
        return Response({"results": results})


# --- 2) Ask Tutor ---
class AskTutor(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        video_id = request.data.get("video_id")
        question = request.data.get("question")

        if not video_id or not question:
            return Response({"error": "video_id and question are required"}, status=400)

        yt_rag.prepare_video(video_id)
        answer = yt_rag.ask_video(question, video_id)

        return Response({"answer": answer})
