"""Point d'entrée Azure Functions (modèle de programmation Python v2).

Azure Functions s'attend à un fichier `function_app.py` à la racine du paquet
déployé. Il ne contient AUCUNE logique métier : il ne fait que brancher
l'application FastAPI existante (`backend/app/main.py`) sur le runtime Azure.

Conséquence : en local on lance `uvicorn backend.app.main:app`, sur Azure le
même objet `app` est servi par Functions. Une seule base de code, deux
hébergeurs possibles.
"""

from __future__ import annotations

import azure.functions as func

from backend.app.main import app as fastapi_app
from backend.app.main import model_service

# Sous Functions, il n'y a pas de protocole ASGI "lifespan" : l'événement de
# démarrage de FastAPI n'est pas déclenché. Sans cet appel explicite, le modèle
# ne serait jamais chargé et toutes les prédictions répondraient 503.
# Le chargement reste unique : il a lieu à l'initialisation du worker.
if not model_service.is_loaded:
    model_service.load()

# `{*route}` capture tous les chemins (voir host.json : routePrefix vide) pour
# que /health, /api/predict et les fichiers du frontend atteignent FastAPI.
app = func.AsgiFunctionApp(
    app=fastapi_app,
    http_auth_level=func.AuthLevel.ANONYMOUS,
)
