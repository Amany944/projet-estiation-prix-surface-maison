"""Point d'entrée du backend FastAPI.

Routes :
    GET  /health       -> état du service et du modèle
    POST /api/predict  -> prédiction du prix estimé

Démarrage :
    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas.prediction import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from .services.model_service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement unique au démarrage, pas à chaque requête.
    model_service.load()
    yield


app = FastAPI(
    title="API de prédiction immobilière",
    version="1.0.0",
    lifespan=lifespan,
)

# Origines autorisées : configurables, avec les ports Vite par défaut.
origines = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origines],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def erreur_inattendue(request: Request, exc: Exception) -> JSONResponse:
    """Masque les traces techniques : le client reçoit un message générique."""
    logger.exception("Erreur non gérée sur %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Une erreur interne est survenue."},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Le modèle est indisponible. Entraînez-le puis redémarrez l'API.",
        )

    try:
        resultat = model_service.predict(
            superficie_m2=payload.superficie_m2,
            nb_pieces=payload.nb_pieces,
            age_annees=payload.age_annees,
            quartier=payload.quartier,
        )
    except Exception:
        logger.exception("Échec de la prédiction")
        raise HTTPException(
            status_code=500,
            detail="La prédiction a échoué.",
        ) from None

    intervalle_min = resultat.get("intervalle_min")
    intervalle_max = resultat.get("intervalle_max")

    return PredictionResponse(
        success=True,
        prediction=PredictionResult(
            prix_estime=round(resultat["prix_estime"], 2),
            unite=model_service.metadata.get("unite", "FCFA"),
            intervalle_min=(
                round(intervalle_min, 2) if intervalle_min is not None else None
            ),
            intervalle_max=(
                round(intervalle_max, 2) if intervalle_max is not None else None
            ),
        ),
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Décrit le modèle servi : variables attendues, métriques, fiabilité."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Le modèle est indisponible.")

    meta = model_service.metadata
    return ModelInfoResponse(
        algorithme=meta.get("algorithme", "inconnu"),
        features=meta.get("features", []),
        quartiers=meta.get("quartiers", []),
        unite=meta.get("unite", "FCFA"),
        jeu_de_donnees=meta.get("jeu_de_donnees", "inconnu"),
        entraine_le=meta.get("entraine_le"),
        metriques_test=meta.get("metriques_test", {}),
        intervalle_80=meta.get("intervalle_80", {}),
        selection_modele=meta.get("selection_modele", {}),
    )


# --- Service du frontend compilé (facultatif) -------------------------------
# Si `Api-python-js/dist` existe (`npm run build`), FastAPI sert l'application
# sur la même origine que l'API : un seul port, donc aucun problème de CORS et
# un seul lien à partager. En développement, on continue d'utiliser Vite (5173)
# pour le rechargement à chaud ; ce montage n'interfère pas avec lui.
DIST_DIR = Path(__file__).resolve().parents[2] / "Api-python-js" / "dist"

if DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="frontend")
    logger.info("Frontend compilé servi depuis %s", DIST_DIR)
