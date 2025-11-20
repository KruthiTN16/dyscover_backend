# core/management/commands/train_risk_model.py
from django.core.management.base import BaseCommand
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
from django.conf import settings
from core.utils import MODEL_DIR, MODEL_PATH, VECT_PATH

class Command(BaseCommand):
    help = "Train a demo logistic regression risk model (synthetic data)."

    def handle(self, *args, **options):
        # Synthetic dataset (replace with real labeled data later)
        texts = [
            "difficulty reading letters words slow reading poor spelling",
            "struggles to sound out words trouble with phonics reading aloud",
            "good reader enjoys books reads fluently",
            "can read okay minor spelling errors",
            "severe difficulty reading words avoiding reading",
            "mistakes reversing letters and difficulty with spelling"
        ]
        labels = [1,1,0,0,1,1]  # 1 -> likely dyslexia, 0 -> unlikely

        # Simple TF-IDF vectorizer
        vect = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
        X = vect.fit_transform(texts)
        y = np.array(labels)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Save artifacts
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(MODEL_PATH))
        joblib.dump(vect, str(VECT_PATH))

        self.stdout.write(self.style.SUCCESS(f"Trained demo model and saved to {MODEL_PATH} and {VECT_PATH}"))
