"""
Script 00 : voir quels modèles sont disponibles sur Albert.

La liste bouge de temps en temps (des modèles arrivent, d'autres disparaissent).
Si un autre script plante avec une erreur "model not found",
lancez celui-ci pour voir ce qui existe vraiment.

    python 00_lister_les_modeles.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# --- Initialisation du client Albert ---
load_dotenv()                        # lit le fichier .env (là où se trouve la clé)
API_KEY = os.getenv("ALBERT_API_KEY")
if not API_KEY:
    sys.exit("Il manque la clé API : copiez .env.example en .env et collez votre clé dedans.")

client = OpenAI(base_url="https://albert.api.etalab.gouv.fr/v1", api_key=API_KEY)

# On demande à l'API la liste de ses modèles
modeles = client.models.list().data

for m in modeles:
    # "type" dit à quoi sert le modèle : text-generation (chat),
    # text-embeddings-inference (RAG), automatic-speech-recognition (audio)...
    type_modele = getattr(m, "type", "?")
    print(f"{m.id:45s} {type_modele}")
