"""
Script 02 : un RAG minimal.

RAG = "Retrieval Augmented Generation". L'idée, c'est de ne pas demander
au LLM de répondre de mémoire, mais de lui donner d'abord les bons
passages de VOS documents, puis de lui demander de répondre avec.

Les étapes, dans l'ordre :
  1. on lit les fichiers .txt du dossier "documents"
  2. on les découpe en petits morceaux
  3. on transforme chaque morceau en vecteur (une liste de nombres qui
     représente son sens) avec un modèle d'embeddings
  4. on fait pareil avec la question
  5. on garde les morceaux dont le vecteur ressemble le plus à celui de la question
  6. on les donne au LLM avec la question

    python 02_rag.py 

Pour l'adapter : déposez vos propres .txt dans le dossier "documents".
"""

import glob
import os
import sys

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# --- Initialisation du client Albert ---
load_dotenv()                        # lit le fichier .env (là où se trouve la clé)
API_KEY = os.getenv("ALBERT_API_KEY")
if not API_KEY:
    sys.exit("Il manque la clé API : copiez .env.example en .env et collez votre clé dedans.")

client = OpenAI(base_url="https://albert.api.etalab.gouv.fr/v1", api_key=API_KEY)
MODELE = "gpt-oss-120b"              # le modèle qui rédige la réponse
MODELE_EMBEDDINGS = "bge-m3"         # le modèle qui transforme le texte en vecteurs

DOSSIER = "documents"
TAILLE_MORCEAU = 600    # nombre de caractères par morceau
CHEVAUCHEMENT = 100     # les morceaux se recouvrent un peu pour ne pas couper une idée en deux
NB_MORCEAUX = 3         # combien de morceaux on donne au LLM


def lire_et_decouper(dossier):
    """Lit tous les .txt du dossier et renvoie une liste de morceaux avec leur fichier d'origine."""
    morceaux = []
    for chemin in sorted(glob.glob(os.path.join(dossier, "*.txt"))):
        with open(chemin, encoding="utf-8") as f:
            texte = f.read()
        # On avance par pas de (taille - chevauchement)
        for debut in range(0, len(texte), TAILLE_MORCEAU - CHEVAUCHEMENT):
            morceau = texte[debut:debut + TAILLE_MORCEAU].strip()
            if morceau:
                morceaux.append({"source": os.path.basename(chemin), "texte": morceau})
    return morceaux


def vectoriser(client, textes):
    """Transforme une liste de textes en matrice de vecteurs normalisés."""
    vecteurs = []
    # On envoie par paquets pour ne pas faire une requête gigantesque
    for i in range(0, len(textes), 32):
        reponse = client.embeddings.create(model=MODELE_EMBEDDINGS, input=textes[i:i + 32])
        vecteurs.extend(d.embedding for d in reponse.data)
    matrice = np.array(vecteurs)
    # On normalise (longueur 1) : le produit scalaire donne alors directement
    # la similarité entre deux textes, entre -1 et 1.
    return matrice / np.linalg.norm(matrice, axis=1, keepdims=True)


question =  "Ou est le hackathon ?"

print("Question : ", question)
# Étapes 1 à 3 : lire, découper, vectoriser
morceaux = lire_et_decouper(DOSSIER)
if not morceaux:
    sys.exit(f"Aucun fichier .txt trouvé dans le dossier '{DOSSIER}'.")
print(f"{len(morceaux)} morceaux indexés.")
vecteurs_docs = vectoriser(client, [m["texte"] for m in morceaux])

# Étapes 4 et 5 : vectoriser la question, trouver les morceaux les plus proches
vecteur_question = vectoriser(client, [question])[0]
scores = vecteurs_docs @ vecteur_question                # un score par morceau
meilleurs = np.argsort(scores)[::-1][:NB_MORCEAUX]       # les indices des plus hauts scores

# Étape 6 : on construit le contexte et on interroge le LLM
contexte = ""
for i in meilleurs:
    contexte += f"[{morceaux[i]['source']}]\n{morceaux[i]['texte']}\n\n"
    print(f"  retenu : {morceaux[i]['source']} (score {scores[i]:.2f})")

reponse = client.chat.completions.create(
    model=MODELE,
    messages=[
        {
            "role": "system",
            "content": (
                "Tu réponds uniquement à partir du contexte fourni. "
                "Si la réponse n'y est pas, dis-le clairement, n'invente rien. "
                "Indique le nom du fichier source entre crochets."
            ),
        },
        {"role": "user", "content": f"Contexte :\n{contexte}\nQuestion : {question}"},
    ],
    temperature=0.1,
)

print("\n" + reponse.choices[0].message.content)
