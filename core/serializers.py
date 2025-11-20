# core/serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import Child, Questionnaire, Assessment, RiskResult, ConcernAnalysis

User = get_user_model()

class ChildSerializer(serializers.ModelSerializer):
    class Meta:
        model = Child
        fields = ("id", "name", "age", "grade", "school")

class UserSerializer(serializers.ModelSerializer):
    children = ChildSerializer(many=True, read_only=True)
    class Meta:
        model = User
        fields = ("id", "username", "first_name", "last_name", "email", "role", "children")
        read_only_fields = ("id",)

class RegisterSerializer(serializers.ModelSerializer):
    child = ChildSerializer(required=False)
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ("id", "username", "first_name", "last_name", "email", "password", "role", "child")
        extra_kwargs = {"role": {"required": False}}

    def create(self, validated_data):
        child_data = validated_data.pop("child", None)
        password = validated_data.pop("password", None)
        user_fields = {k: v for k, v in validated_data.items() if k in [f.name for f in User._meta.fields]}
        user = User.objects.create(**user_fields)
        if password:
            user.set_password(password)
        if hasattr(user, "role") and validated_data.get("role"):
            try:
                user.role = validated_data.get("role")
            except Exception:
                pass
        user.save()
        if child_data:
            Child.objects.create(parent=user, **child_data)
        return user

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token["role"] = getattr(user, "role", "")
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data["user"] = UserSerializer(self.user).data
        return data

class QuestionnaireSerializer(serializers.ModelSerializer):
    class Meta:
        model = Questionnaire
        fields = ("id", "slug", "title", "description")

class AssessmentSubmitSerializer(serializers.ModelSerializer):
    class Meta:
        model = Assessment
        fields = ("id", "questionnaire", "student_name", "parent", "created_at")

class RiskResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = RiskResult
        fields = ("id", "assessment", "risk_score", "risk_level", "method", "created_at")

class ConcernAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConcernAnalysis
        fields = ("id", "assessment", "summary", "keywords", "sentiment", "created_at")
