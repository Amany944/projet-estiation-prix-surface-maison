"""Schémas Pydantic : contrat d'entrée/sortie de l'API.

Ils servent de validation côté backend : si React envoie une valeur absente,
négative ou d'un mauvais type, FastAPI renvoie automatiquement une erreur 422
avant même de toucher au modèle.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ..ml.constants import QUARTIERS


class PredictionRequest(BaseModel):
    """Données envoyées par le formulaire React."""

    superficie_m2: float = Field(
        ..., gt=0, le=2000, description="Superficie du bien en mètres carrés"
    )
    nb_pieces: int = Field(..., ge=1, le=30, description="Nombre de pièces")
    age_annees: float = Field(..., ge=0, le=200, description="Âge du bien en années")
    # Variable catégorielle : le modèle l'encode en one-hot. Une valeur inconnue
    # serait silencieusement ignorée par l'encodeur -> on la rejette ici, en 422.
    quartier: str = Field(
        ..., min_length=1, max_length=50, description=f"Quartier parmi {QUARTIERS}"
    )

    @field_validator("quartier")
    @classmethod
    def quartier_connu(cls, valeur: str) -> str:
        if valeur not in QUARTIERS:
            raise ValueError(f"Quartier inconnu. Valeurs acceptées : {QUARTIERS}")
        return valeur


class PredictionResult(BaseModel):
    prix_estime: float
    unite: str
    # Intervalle à 80 % : absent si les métadonnées du modèle sont manquantes.
    intervalle_min: float | None = None
    intervalle_max: float | None = None


class PredictionResponse(BaseModel):
    """Réponse JSON normalisée consommée par le frontend."""

    success: bool
    prediction: PredictionResult


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Description du modèle servie au frontend (`GET /api/model-info`)."""

    algorithme: str
    features: list[str]
    quartiers: list[str] = Field(default_factory=list)
    unite: str
    jeu_de_donnees: str
    entraine_le: str | None = None
    metriques_test: dict[str, float] = Field(default_factory=dict)
    intervalle_80: dict[str, float] = Field(default_factory=dict)
    selection_modele: dict[str, float] = Field(default_factory=dict)
