from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from .models import User, Child, Questionnaire, Assessment, RiskResult, ConcernAnalysis

@admin.register(User)
class CustomUserAdmin(DjangoUserAdmin):
    list_display = ("username", "email", "role", "is_staff", "is_active")
    fieldsets = DjangoUserAdmin.fieldsets + (
        ("Role / Extra", {"fields": ("role",)}),
    )

@admin.register(Child)
class ChildAdmin(admin.ModelAdmin):
    list_display = ("name","parent","age","grade","school","created_at")

@admin.register(Questionnaire)
class QuestionnaireAdmin(admin.ModelAdmin):
    list_display = ("slug","title")

@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ("id","student_name","parent","created_at")

@admin.register(RiskResult)
class RiskResultAdmin(admin.ModelAdmin):
    list_display = ("assessment","risk_score","risk_level","method","created_at")

@admin.register(ConcernAnalysis)
class ConcernAnalysisAdmin(admin.ModelAdmin):
    list_display = ("assessment","sentiment","created_at")
