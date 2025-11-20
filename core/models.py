from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings

# Custom user to support role field (safe early in project)
class User(AbstractUser):
    ROLE_PARENT = "parent"
    ROLE_STUDENT = "student"
    ROLE_EDUCATOR = "educator"
    ROLE_CHOICES = (
        (ROLE_PARENT, "Parent"),
        (ROLE_STUDENT, "Student"),
        (ROLE_EDUCATOR, "Educator"),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_PARENT)

    def __str__(self):
        return self.get_full_name() or self.username

# Child / Student linked to parent (or educator later)
class Child(models.Model):
    parent = models.ForeignKey("core.User", on_delete=models.CASCADE, related_name="children")
    name = models.CharField(max_length=200)
    age = models.PositiveIntegerField(null=True, blank=True)
    grade = models.CharField(max_length=50, blank=True)
    school = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.parent})"

# Questionnaire (standardized sets: DEST-2 / CLDQ)
class Questionnaire(models.Model):
    slug = models.SlugField(unique=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.title

# Assessment filled by parent
class Assessment(models.Model):
    questionnaire = models.ForeignKey(Questionnaire, on_delete=models.SET_NULL, null=True, blank=True)
    student_name = models.CharField(max_length=255)
    parent = models.ForeignKey("core.User", on_delete=models.CASCADE, related_name="assessments")
    # If you want to store raw answers, create a JSONField named 'answers' (requires Django/Postgres)
    # from django.contrib.postgres.fields import JSONField   # if using Postgres
    # answers = JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Assessment {self.id} - {self.student_name}"

# Risk result produced by model/rule
class RiskResult(models.Model):
    assessment = models.ForeignKey(Assessment, on_delete=models.CASCADE, related_name="risk_results")
    risk_score = models.FloatField(default=0.0)
    risk_level = models.CharField(max_length=50, blank=True)
    method = models.CharField(max_length=100, blank=True)  # e.g. "logistic_regression_v0"
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"RiskResult {self.assessment_id} ({self.risk_score})"

# NLP analysis of free-text concerns
class ConcernAnalysis(models.Model):
    assessment = models.ForeignKey(Assessment, on_delete=models.CASCADE, related_name="concern_analyses")
    summary = models.TextField(blank=True)
    keywords = models.TextField(blank=True)  # comma-separated simple store
    sentiment = models.CharField(max_length=32, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ConcernAnalysis for assessment {self.assessment_id}"
