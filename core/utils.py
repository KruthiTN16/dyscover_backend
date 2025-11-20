# core/utils.py
import os
import json
import joblib
from django.conf import settings
from pathlib import Path
import spacy
from collections import Counter
import re

# load spaCy model
SPACY_MODEL = "en_core_web_sm"
nlp = spacy.load(SPACY_MODEL)

MODEL_DIR = Path(settings.BASE_DIR) / "core" / "models_artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "risk_model.pkl"
VECT_PATH = MODEL_DIR / "vectorizer.pkl"

# ---- NLP helpers ----
def extract_keywords(text, top_k=8):
    """
    Basic keyword extraction using spaCy: nouns & proper nouns, plus frequent tokens.
    """
    if not text or not text.strip():
        return []
    doc = nlp(text.lower())
    # simple tokens: nouns, PROPN, and lemmas longer than 2 chars, excluding stopwords/punct
    tokens = [
        token.lemma_ for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop and token.is_alpha and len(token) > 2
    ]
    common = [t for t, _ in Counter(tokens).most_common(top_k)]
    # fallback: split words with regex if none found
    if not common:
        words = re.findall(r"\w{3,}", text.lower())
        common = list(dict.fromkeys(words))[:top_k]
    return common

def sentiment_analysis(text):
    """
    Very simple sentiment via spaCy's polarity? en_core_web_sm does not provide sentiment.
    For MVP we return neutral and rely on keyword signals. Could integrate vader or transformers later.
    """
    # Placeholder: use token polarity heuristics (extremely naive).
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0}
    lower = text.lower()
    negative = ["worried", "concern", "struggle", "difficult", "frustrat", "slow"]
    positive = ["improve", "good", "progress", "better"]
    score = 0
    for w in negative:
        if w in lower:
            score -= 1
    for w in positive:
        if w in lower:
            score += 1
    label = "neutral"
    if score <= -1:
        label = "negative"
    elif score >= 1:
        label = "positive"
    return {"label": label, "score": score}

# ---- Scoring helpers ----
def rule_based_score(answers, questionnaire=None):
    """
    Simple rule-based scoring: sum positive responses according to expected schema.
    For each answer value pick numeric mapping and normalize to [0,1].
    Expects answers dict: {q1: "never"/"sometimes"/"often"} or numeric.
    """
    if not answers:
        return 0.0
    score = 0.0
    total_weight = 0.0
    for qid, val in answers.items():
        # try to interpret numeric
        try:
            v = float(val)
        except Exception:
            s = str(val).lower()
            # mapping example; adjust to questionnaire specifics
            if s in ("never","no"):
                v = 0.0
            elif s in ("rarely","sometimes"):
                v = 0.5
            elif s in ("often","frequently","yes"):
                v = 1.0
            else:
                v = 0.0
        weight = 1.0
        score += v * weight
        total_weight += weight
    if total_weight == 0:
        return 0.0
    raw = score / total_weight  # in [0,1]
    return float(raw)

def load_model():
    """
    Attempt to load trained logistic model and vectorizer.
    Returns (model, vectorizer) or (None, None) if not available.
    """
    if MODEL_PATH.exists() and VECT_PATH.exists():
        model = joblib.load(str(MODEL_PATH))
        vect = joblib.load(str(VECT_PATH))
        return model, vect
    return None, None

def model_score(answers, questionnaire_text="", fallback=rule_based_score):
    """
    Combine answers into a text or feature vector to feed into vectorizer+model.
    For a simple approach, we convert answers dict => single text string of 'question:answer' pairs,
    vectorize via vectorizer (TF-IDF) and get model.predict_proba.
    """
    model, vect = load_model()
    if model and vect:
        # build pseudo-document
        doc = []
        for k, v in sorted(answers.items()):
            doc.append(f"{k} {v}")
        if questionnaire_text:
            doc.append(questionnaire_text)
        text = " . ".join(doc)
        X = vect.transform([text])
        proba = model.predict_proba(X)[0][1]  # probability of positive class (dyslexia)
        return float(proba), "logistic-regression"
    # fallback
    return fallback(answers), "rule-based"

def risk_level_from_score(score):
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    else:
        return "low"

# ---- Recommendation mapping ----
EXPERT_MAP = {
    "phonological": ["Speech-Language Pathologist", "Remedial Educator"],
    "reading": ["Special Educator", "Remedial Tutor"],
    "attention": ["Child Psychologist", "Educational Psychologist"],
    "general": ["Special Educator", "Speech-Language Pathologist"],
}

def recommend_experts(keywords, risk_level):
    # very naive mapping by keyword
    recs = set()
    for kw in (keywords or []):
        k = kw.lower()
        if "sound" in k or "phon" in k or "phono" in k:
            recs.update(EXPERT_MAP.get("phonological", []))
        if "read" in k or "spelling" in k or "word" in k:
            recs.update(EXPERT_MAP.get("reading", []))
        if "attent" in k or "focus" in k or "concen" in k:
            recs.update(EXPERT_MAP.get("attention", []))
    if not recs:
        recs.update(EXPERT_MAP.get("general", []))
    # consider risk level: if high, push psychologist + specialist
    if risk_level == "high":
        recs.add("Child Psychologist")
        recs.add("Comprehensive Diagnostic Evaluation Center")
    return list(recs)
