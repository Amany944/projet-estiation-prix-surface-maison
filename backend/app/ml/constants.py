"""Contrat de features partagé entre l'entraînement et l'API.

Importé par `train_model.py` (entraînement) et `schemas/prediction.py`
(validation). Une seule source de vérité : si on ajoute un quartier ou une
variable, on la modifie ici et les deux côtés suivent.
"""

NUMERIC_FEATURES = ["superficie_m2", "nb_pieces", "age_annees"]
CATEGORICAL_FEATURES = ["quartier"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "prix_fcfa"
UNITE = "FCFA"

QUARTIERS = ["Plateau", "Cocody", "Marcory", "Yopougon", "Bingerville"]
