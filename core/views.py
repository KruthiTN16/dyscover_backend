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
    negative_cues = sum(1 for w in words if w in {"struggl", "struggle", "difficul", "frustrat", "avoid", "problem", "worri"})
    sentiment = "negative" if negative_cues > positive_cues else ("positive" if positive_cues > negative_cues else "neutral")
    summary = " ".join(words[:20])
    return {"summary": summary, "keywords": keywords, "sentiment": sentiment}

QUESTION_ORDER = [
    "reads_slowly","spelling_difficulty","memory_issues","phonological_awareness",
    "misreads_words","avoids_reading","letter_confusion","rapid_naming",
    "copying_difficulty","frustration_reading",
]
BASE_WEIGHTS = [0.85,0.7,0.5,0.75,0.7,0.35,0.9,0.6,0.45,0.5]

def compute_risk_probability(answer_dict):
    vec = []
    for k in QUESTION_ORDER:
        v = answer_dict.get(k)
        try:
            vnum = float(v)
        except Exception:
            vnum = 0.0
        vec.append(vnum)
    linear = sum([a*b for a,b in zip(vec, BASE_WEIGHTS)])
    denom = sum(BASE_WEIGHTS) * 3.0
    offset = sum(BASE_WEIGHTS) * 1.5
    linear_norm = (linear - offset) / denom
    prob = 1.0 / (1.0 + math.exp(-4.0 * linear_norm))
    prob = max(0.0, min(1.0, prob))
    return prob

class ParentAssessmentSubmitView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        data = request.data
        parent = request.user
        student_name = data.get("student_name") or "Unknown"
        answers = data.get("answers", {})
        if not isinstance(answers, dict):
            return Response({"detail":"answers must be a dict"}, status=400)

        assessment = Assessment.objects.create(
            questionnaire=None,
            student_name=student_name,
            parent=parent,
        )

        prob = compute_risk_probability(answers)
        risk_label = 1 if prob >= 0.5 else 0
        risk_level = "High" if prob >= 0.7 else ("Moderate" if prob >= 0.5 else "Low")

        risk = RiskResult.objects.create(
            assessment=assessment,
            risk_score=prob,
            risk_level=risk_level,
            method="logistic_demo_v1"
        )

        parent_concern = data.get("parent_concern", "")
        nlp = analyze_concern_text(parent_concern)
        concern = ConcernAnalysis.objects.create(
            assessment=assessment,
            summary=nlp.get("summary",""),
            keywords=",".join(nlp.get("keywords",[])),
            sentiment=nlp.get("sentiment","")
        )

        recommended_experts = [
            {"name":"Dr. Priya Sharma","specialty":"Educational Psychologist","contact":"example@clinic.org"},
            {"name":"Mr. R. Nair","specialty":"Speech & Language Therapist","contact":"example2@clinic.org"},
        ]

        response = {
            "assessment_id": assessment.id,
            "student_name": assessment.student_name,
            "risk_probability": round(prob, 3),
            "risk_label": "At Risk" if risk_label else "Low Risk",
            "risk_level": risk_level,
            "nlp_summary": nlp.get("summary"),
            "nlp_keywords": nlp.get("keywords"),
            "nlp_sentiment": nlp.get("sentiment"),
            "recommended_experts": recommended_experts,
        }
        return Response(response, status=status.HTTP_201_CREATED)

class EducatorStudentView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, student_id):
        assessments = Assessment.objects.filter(id=student_id)
        if not assessments.exists():
            return Response({"detail":"No assessment found"}, status=404)
        a = assessments.first()
        risk_qs = a.risk_results.all().order_by("-created_at")
        concern_qs = a.concern_analyses.all().order_by("-created_at")
        risk_ser = RiskResultSerializer(risk_qs, many=True)
        concern_ser = ConcernAnalysisSerializer(concern_qs, many=True)
        return Response({
            "assessment": AssessmentSubmitSerializer(a).data,
            "risks": risk_ser.data,
            "concerns": concern_ser.data
        })
