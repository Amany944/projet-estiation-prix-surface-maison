"""Tests du backend : santé, prédiction, intervalle, infos modèle, erreurs."""

from fastapi.testclient import TestClient

from backend.app.main import app

PAYLOAD_VALIDE = {
    "superficie_m2": 50,
    "nb_pieces": 3,
    "age_annees": 5,
    "quartier": "Cocody",
}


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valide():
    with TestClient(app) as client:
        response = client.post("/api/predict", json=PAYLOAD_VALIDE)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["prediction"]["unite"] == "FCFA"
    prix = body["prediction"]["prix_estime"]
    assert isinstance(prix, (int, float))
    assert prix > 0


def test_predict_intervalle_coherent():
    with TestClient(app) as client:
        response = client.post("/api/predict", json=PAYLOAD_VALIDE)

    prediction = response.json()["prediction"]
    assert prediction["intervalle_min"] < prediction["prix_estime"]
    assert prediction["intervalle_max"] > prediction["prix_estime"]


def test_predict_est_deterministe():
    """Deux appels identiques doivent renvoyer exactement la même valeur."""
    with TestClient(app) as client:
        premier = client.post("/api/predict", json=PAYLOAD_VALIDE).json()
        second = client.post("/api/predict", json=PAYLOAD_VALIDE).json()
    assert premier == second


def test_quartier_influence_le_prix():
    """Le quartier est une vraie variable du modèle, pas un décor.

    Même bien, deux quartiers différents : le prix doit changer nettement.
    """
    with TestClient(app) as client:
        plateau = client.post(
            "/api/predict", json={**PAYLOAD_VALIDE, "quartier": "Plateau"}
        ).json()["prediction"]["prix_estime"]
        yopougon = client.post(
            "/api/predict", json={**PAYLOAD_VALIDE, "quartier": "Yopougon"}
        ).json()["prediction"]["prix_estime"]

    assert plateau > yopougon * 1.5


def test_quartier_inconnu_rejete():
    with TestClient(app) as client:
        response = client.post(
            "/api/predict", json={**PAYLOAD_VALIDE, "quartier": "Atlantide"}
        )
    assert response.status_code == 422


def test_predict_champ_manquant():
    with TestClient(app) as client:
        response = client.post("/api/predict", json={"superficie_m2": 50})
    assert response.status_code == 422  # validation Pydantic


def test_predict_valeur_invalide():
    payload = {**PAYLOAD_VALIDE, "superficie_m2": -10}
    with TestClient(app) as client:
        response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_type_invalide():
    payload = {**PAYLOAD_VALIDE, "superficie_m2": "abc"}
    with TestClient(app) as client:
        response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_model_info():
    with TestClient(app) as client:
        response = client.get("/api/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["features"] == [
        "superficie_m2",
        "nb_pieces",
        "age_annees",
        "quartier",
    ]
    assert "Plateau" in body["quartiers"]
    assert body["unite"] == "FCFA"
    assert "r2" in body["metriques_test"]
    # La sélection de modèle doit être tracée (comparaison des candidats).
    assert len(body["selection_modele"]) >= 2
    # Le jeu de données de démonstration doit être annoncé comme tel.
    assert "synth" in body["jeu_de_donnees"].lower()
