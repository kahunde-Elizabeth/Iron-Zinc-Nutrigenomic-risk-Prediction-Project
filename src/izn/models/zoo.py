"""The algorithms compared in the prototype (all scikit-learn, CPU, small data)."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_models(seed: int = 42) -> dict:
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(random_state=seed)),
    }
