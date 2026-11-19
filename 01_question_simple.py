"""
Script 01 : poser une question au LLM et afficher la réponse.

C'est la base de tout. Si ce script marche, votre clé et votre
installation sont bonnes, et le reste ne sera que des variations.

    python 01_question_simple.py "Pourquoi le ciel est bleu ?"

Sans argument, il pose une question par défaut.
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
MODELE = "gpt-oss-120b"   


question = "Explique en deux phrases ce qu'est un trou noir."

print(question)

reponse = client.chat.completions.create(
    model=MODELE,
    messages=[
        # Le message "system" fixe le rôle du modèle. C'est le premier endroit à modifier si vous voulez changer le style des réponses.
        {"role": "system", "content": "Tu es un assistant scientifique. Tu réponds en français, de façon claire et concise."},
        # Le message "user", c'est la question de l'utilisateur.
        {"role": "user", "content": question},
    ],
    # 0 = réponses très stables, 1 = plus créatif. Pour de la science, on reste bas.
    temperature=1,
)

# La réponse est dans le premier "choix" renvoyé par l'API
print(reponse.choices[0].message.content)
