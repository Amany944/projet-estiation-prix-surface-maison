"""Service de chargement et d'utilisation du modèle.

Le modèle et ses métadonnées sont chargés UNE SEULE FOIS au démarrage du
backend, puis réutilisés pour chaque requête. On ne réentraîne jamais dans
une route.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# Racine du projet = 3 niveaux au-dessus de ce fichier (app/services/...).
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_PATH = MODELS_DIR / "prix_maison.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# L'ordre est critique : il doit être identique au script d'entraînement.
FEATURES = ["superficie_m2", "nb_pieces", "age_annees", "quartier"]


class ModelService:
    """Encapsule le modèle pour isoler FastAPI de scikit-learn."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model: Any | None = None
        self.metadata: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Charge l'artefact joblib et ses métadonnées."""
        if not self.model_path.exists():
            logger.error("Modèle introuvable : %s", self.model_path)
            self.model = None
            return

        # Sécurité : joblib/pickle peut exécuter du code arbitraire.
        # Ne charger que des fichiers produits par ce projet (voir README).
        self.model = joblib.load(self.model_path)
        logger.info("Modèle chargé depuis %s", self.model_path)

        # Les métadonnées sont optionnelles : sans elles, l'API fonctionne
        # mais ne peut pas fournir d'intervalle ni de métriques.
        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(
                    self.metadata_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                logger.exception("Métadonnées illisibles, on continue sans.")
                self.metadata = {}
        else:
            logger.warning("Métadonnées absentes : %s", self.metadata_path)
            self.metadata = {}

    def predict(
        self,
        superficie_m2: float,
        nb_pieces: int,
        age_annees: float,
        quartier: str,
    ) -> dict[str, float]:
        """Retourne le prix estimé et son intervalle à 80 %.

        On construit un DataFrame d'UNE ligne avec les noms de colonnes exacts
        vus à l'entraînement. C'est important : le préprocesseur est un
        ColumnTransformer qui s'appuie sur les noms (une colonne numérique à
        mettre à l'échelle, une colonne texte à encoder en one-hot). Un simple
        tableau numpy ferait perdre cette distinction.
        """
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé.")

        donnees = pd.DataFrame(
            [
                {
                    "superficie_m2": superficie_m2,
                    "nb_pieces": nb_pieces,
                    "age_annees": age_annees,
                    "quartier": quartier,
                }
            ],
            columns=FEATURES,
        )
        prediction = self.model.predict(donnees)

        # prediction est un ndarray de forme (1,) : on extrait le scalaire et on
        # le convertit en float pour garantir une sérialisation JSON correcte.
        prix = float(prediction[0])

        resultat = {"prix_estime": prix}

        # Intervalle calculé à partir des erreurs relatives observées sur le
        # jeu de test (voir train_model.py).
        intervalle = self.metadata.get("intervalle_80") or {}
        q_bas = intervalle.get("quantile_bas")
        q_haut = intervalle.get("quantile_haut")
        if isinstance(q_bas, (int, float)) and isinstance(q_haut, (int, float)):
            resultat["intervalle_min"] = prix * (1 + q_bas)
            resultat["intervalle_max"] = prix * (1 + q_haut)

        return resultat
