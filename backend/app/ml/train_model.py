"""Entraînement du modèle de prédiction de prix immobilier.

DONNÉES DE DÉMONSTRATION
========================
Aucun jeu de données réel n'est disponible dans ce projet : ce script génère un
jeu SYNTHÉTIQUE, dont le processus de génération est DÉLIBÉRÉMENT non linéaire
(effet marginal décroissant de la superficie, décote accélérée par l'âge, prime
de quartier, bruit hétéroscédastique). Objectif : produire un problème où le
choix du modèle a un vrai impact, et non une régression linéaire qui récite une
formule.

Ce qui est réel et transférable dans ce script :
  - le split train/test ;
  - la validation croisée K-fold ;
  - la sélection de modèle sur des métriques hors échantillon ;
  - le Pipeline de prétraitement sérialisé avec le modèle ;
  - le calcul d'un intervalle d'incertitude à partir des résidus.

Pour passer en production, remplacer `make_dataset()` par un vrai jeu de
données : le reste du script n'a pas à changer.

Usage :
    python -m backend.app.ml.train_model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .constants import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    QUARTIERS,
    TARGET,
    UNITE,
)

# Prix moyen au m² par quartier (FCFA) : sert de base au générateur.
PRIX_M2_QUARTIER = {
    "Plateau": 900_000,
    "Cocody": 750_000,
    "Marcory": 500_000,
    "Yopougon": 220_000,
    "Bingerville": 150_000,
}

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_PATH = MODELS_DIR / "prix_maison.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def make_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Génère un jeu de données synthétique non linéaire et reproductible."""
    rng = np.random.default_rng(seed)

    superficie = rng.uniform(20, 400, size=n_samples)
    nb_pieces = rng.integers(1, 12, size=n_samples).astype(float)
    age = rng.uniform(0, 80, size=n_samples)
    quartier = rng.choice(QUARTIERS, size=n_samples)

    prix_m2 = np.array([PRIX_M2_QUARTIER[q] for q in quartier])

    # Superficie : rendement décroissant. Une grande surface se vend moins cher
    # au m² qu'une petite -> sqrt au lieu de linéaire.
    valeur_surface = prix_m2 * np.sqrt(superficie) * 6

    # Âge : décote qui s'accélère (quadratique) puis plancher à 45 %.
    decote_age = np.clip(1 - (age / 100) ** 2 * 0.55, 0.45, 1.0)

    # Pièces : effet plafonné, au-delà de 6 pièces l'apport marginal s'effondre.
    bonus_pieces = np.minimum(nb_pieces, 6) * 0.04

    # Bruit hétéroscédastique : les biens chers sont plus difficiles à estimer.
    ecart_relatif = 0.05 + (valeur_surface / valeur_surface.max()) * 0.10
    bruit = rng.normal(1.0, ecart_relatif)

    prix = valeur_surface * decote_age * (1 + bonus_pieces) * bruit

    return pd.DataFrame(
        {
            "superficie_m2": superficie,
            "nb_pieces": nb_pieces,
            "age_annees": age,
            "quartier": quartier,
            TARGET: prix,
        }
    )


def build_preprocessor() -> ColumnTransformer:
    """Prétraitement hétérogène : mise à l'échelle + encodage one-hot.

    Le ColumnTransformer est DANS le Pipeline final : il est donc sauvegardé
    avec le modèle. À la prédiction, il applique exactement les mêmes
    transformations (moyennes, écarts-types, catégories) qu'à l'entraînement.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    """Modèles candidats. Le choix final se fait par validation croisée."""
    return {
        "LinearRegression": Pipeline(
            [("preprocessor", build_preprocessor()), ("model", LinearRegression())]
        ),
        "RandomForestRegressor": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "HistGradientBoostingRegressor": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.08, max_iter=400, random_state=42
                    ),
                ),
            ]
        ),
    }


def train() -> Pipeline:
    data = make_dataset()
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Sélection de modèle par validation croisée 5-fold sur le TRAIN ------
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores: dict[str, float] = {}

    for nom, candidat in build_candidates().items():
        scores_cv = cross_val_score(
            candidat, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1
        )
        scores[nom] = float(scores_cv.mean())
        print(f"  {nom:<32} R² CV = {scores_cv.mean():.4f} (+/- {scores_cv.std():.4f})")

    meilleur_nom = max(scores, key=scores.__getitem__)
    modele = build_candidates()[meilleur_nom]
    print(f"Modèle retenu : {meilleur_nom}")

    # --- Entraînement final sur tout le train, évaluation sur le test --------
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Erreur relative médiane : plus robuste que la moyenne aux valeurs extrêmes.
    erreur_relative = (y_pred - y_test) / y_test
    q_bas, q_haut = np.quantile(erreur_relative, [0.10, 0.90])

    print(f"Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"R² (test)   = {r2:.4f}")
    print(f"MAE (test)  = {mae:,.0f} {UNITE}")
    print(f"RMSE (test) = {rmse:,.0f} {UNITE}")
    print(f"Intervalle 80 % : [{q_bas:+.2%}, {q_haut:+.2%}]")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, MODEL_PATH)

    metadata = {
        "algorithme": meilleur_nom,
        "features": FEATURES,
        "features_numeriques": NUMERIC_FEATURES,
        "features_categorielles": CATEGORICAL_FEATURES,
        "quartiers": QUARTIERS,
        "target": TARGET,
        "unite": UNITE,
        "jeu_de_donnees": "synthétique — démonstration, non fiable en production",
        "entraine_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "echantillons": {"train": len(X_train), "test": len(X_test)},
        "selection_modele": {
            nom: round(score, 4) for nom, score in scores.items()
        },
        "metriques_test": {
            "r2": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
        },
        "intervalle_80": {
            "quantile_bas": round(float(q_bas), 6),
            "quantile_haut": round(float(q_haut), 6),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modèle      : {MODEL_PATH}")
    print(f"Métadonnées : {METADATA_PATH}")
    return modele


if __name__ == "__main__":
    train()
