# Prédiction immobilière — React + FastAPI + scikit-learn

Chaîne complète : formulaire React → requête HTTP → API FastAPI → modèle
scikit-learn → réponse JSON → état React → affichage.

## Architecture

```
.
├── Api-python-js/                 # Frontend React 19 + Vite
│   ├── src/
│   │   ├── App.jsx                # formulaire + états (données, résultat, chargement, erreur)
│   │   ├── services/
│   │   │   └── predictionService.js   # seul endroit qui connaît l'URL de l'API
│   │   └── App.css
│   └── .env.example               # VITE_API_URL
│
├── backend/                       # Backend FastAPI
│   ├── app/
│   │   ├── main.py                # routes API, CORS, gestion d'erreurs
│   │   ├── schemas/prediction.py  # validation Pydantic
│   │   ├── services/model_service.py  # chargement unique du modèle
│   │   └── ml/
│   │       ├── constants.py       # contrat de features, partagé API ↔ entraînement
│   │       └── train_model.py     # entraînement, sélection de modèle, métriques
│   ├── tests/test_api.py          # 10 tests
│   └── requirements.txt
│
├── models/
│   ├── prix_maison.joblib         # Pipeline entraîné (prétraitement + RandomForest)
│   └── model_metadata.json        # métriques, intervalle, date d'entraînement
│
├── Dockerfile                     # image du backend
├── docker-compose.yml             # backend + frontend en une commande
└── myenv/                         # environnement virtuel Python
```

> `essayons-python-&-js/` est un ancien clone du template Vite, laissé intact
> et non utilisé. `index.html` / `js/render.js` à la racine sont un brouillon
> statique, sans rapport avec l'application.

## ⚠️ À propos du modèle

**Aucun modèle ni jeu de données réel n'existait dans ce projet.** Le notebook
`app.ipynb` est un brouillon qui échoue (`linearRegression` au lieu de
`LinearRegression`) et ne produit aucun artefact.

`backend/app/ml/train_model.py` génère donc des **données synthétiques** dont le
processus de génération est délibérément **non linéaire** (rendement décroissant
de la superficie, décote accélérée par l'âge, prime de quartier, bruit
hétéroscédastique). Le modèle apprend ce processus : **ce n'est pas un modèle
prédictif fiable**, seulement un moyen de tester l'architecture de bout en bout.
Pour passer en production, remplacer `make_dataset()` par le chargement d'un
vrai jeu de données — le reste du script ne change pas.

Trois candidats sont comparés par **validation croisée 5-fold** sur le train,
le meilleur est retenu, puis évalué sur un **jeu de test mis de côté** (20 %)
jamais vu à l'entraînement :

| Candidat                      | R² (validation croisée) |
| ----------------------------- | ----------------------- |
| **RandomForestRegressor**     | **0.9491** (retenu)     |
| HistGradientBoostingRegressor | 0.9431                  |
| LinearRegression              | 0.8692                  |

Performances du modèle retenu sur le jeu de test :

| Métrique     | Valeur             |
| ------------ | ------------------ |
| R² (test)    | 0.9445             |
| MAE          | ≈ 4 108 000 FCFA   |
| RMSE         | ≈ 6 996 000 FCFA   |
| Intervalle 80 % | −11,3 % / +15,3 % |

L'intervalle à 80 % vient des quantiles 10 et 90 des erreurs relatives du
modèle sur le jeu de test — l'erreur est multiplicative, un écart fixe en FCFA
serait faux. C'est une **approximation honnête**, pas une garantie
statistique : elle ne vaut que pour des biens comparables à ceux vus à
l'entraînement.

Variables explicatives (`backend/app/ml/constants.py` est la source unique de
vérité, importée par l'API et par l'entraînement) :

| Champ           | Type  | Contrainte                  |
| --------------- | ----- | --------------------------- |
| `superficie_m2` | float | 0 < x ≤ 2000                |
| `nb_pieces`     | int   | 1 ≤ x ≤ 30                  |
| `age_annees`    | float | 0 ≤ x ≤ 200                 |
| `quartier`      | str   | Plateau, Cocody, Marcory, Yopougon, Bingerville |

Le `quartier` est encodé en one-hot **dans le Pipeline**, donc sauvegardé avec
le modèle : les catégories vues à l'entraînement sont exactement celles
appliquées à la prédiction. Un quartier inconnu est rejeté en `422`.

## Installation

```bash
# 1. Dépendances Python (le venv myenv existe déjà)
myenv/Scripts/python.exe -m pip install -r backend/requirements.txt

# 2. Dépendances frontend
cd Api-python-js && npm install && cd ..

# 3. (Re)générer le modèle — déjà présent dans models/
myenv/Scripts/python.exe -m backend.app.ml.train_model

# 4. Configuration frontend (facultatif : la valeur par défaut est 127.0.0.1:8000)
cp Api-python-js/.env.example Api-python-js/.env
```

## Lancement

```bash
# Terminal 1 — backend (port 8000)
myenv/Scripts/python.exe -m uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — frontend (port 5173)
cd Api-python-js && npm run dev
```

Ouvrir http://localhost:5173

## API

| Méthode | Route             | Description                                |
| ------- | ----------------- | ------------------------------------------ |
| GET     | `/health`         | `{"status":"ok","model_loaded":true}`     |
| POST    | `/api/predict`    | prédiction + intervalle                    |
| GET     | `/api/model-info` | algorithme, variables, métriques, fiabilité|

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"superficie_m2":50,"nb_pieces":3,"age_annees":5,"quartier":"Cocody"}'
# {"success":true,"prediction":{"prix_estime":34099207.96,"unite":"FCFA",
#  "intervalle_min":30253976.67,"intervalle_max":39309737.43}}
```

Même bien à Yopougon : `10193760.97` FCFA — le quartier pèse réellement dans
la prédiction.

Codes d'erreur : `422` données invalides, `503` modèle indisponible,
`500` erreur interne (message générique, aucune trace exposée).

## Docker (optionnel)

```bash
docker compose up --build
# frontend http://localhost:5173 — backend http://localhost:8000/docs
```

`VITE_API_URL` reste `http://127.0.0.1:8000` car l'appel est fait par le
**navigateur**, pas par le conteneur frontend : le nom de service `backend` ne
résoudrait pas côté client.

## Tests

```bash
myenv/Scripts/python.exe -m pytest backend/tests -q
cd Api-python-js && npm run lint && npm run build
```

## Erreurs fréquentes

| Symptôme                              | Cause / solution                                                    |
| ------------------------------------- | ------------------------------------------------------------------- |
| « Impossible de joindre l'API »       | backend non démarré, ou `VITE_API_URL` incorrecte                    |
| `503` Le modèle est indisponible      | `models/prix_maison.joblib` absent → lancer `train_model.py`         |
| Erreur CORS dans la console           | le port frontend n'est pas dans `FRONTEND_ORIGINS` (défaut : 5173)   |
| `ModuleNotFoundError: fastapi`        | utiliser `myenv/Scripts/python.exe`, pas le Python système           |
| Port 8000 déjà utilisé                | `--port 8001` + `VITE_API_URL=http://127.0.0.1:8001`                 |

## Sécurité

- CORS limité aux origines locales Vite, surchargeable via `FRONTEND_ORIGINS`.
- Validation stricte côté Pydantic (types + bornes) avant tout appel au modèle.
- Aucune donnée sensible dans les réponses d'erreur (handler global).
- `VITE_*` est public dans le navigateur : n'y jamais mettre de secret.
- `joblib`/`pickle` peut exécuter du code : ne charger que des artefacts
  produits par ce projet.
