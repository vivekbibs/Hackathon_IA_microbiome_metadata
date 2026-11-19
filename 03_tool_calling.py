"""
Script 03 : le tool calling (ou "function calling").

Un LLM ne sait pas faire de recherche internet, ni lire un fichier, ni calculer
de façon fiable. Mais on peut lui donner des "outils" : des fonctions Python
qu'on écrit nous-mêmes. Attention, le modèle n'exécute rien lui-même.
Il dit seulement "j'aimerais appeler tel outil avec tels arguments", et c'est
notre code qui le fait, puis qui lui renvoie le résultat.

Le script est en deux parties :
  1. la description des outils (ce que le modèle voit)
  2. la boucle qui exécute les outils demandés

Ici l'outil cherche des articles scientifiques sur PubMed.

    python 03_tool_calling.py
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from openai import OpenAI

# --- Initialisation du client Albert ---
load_dotenv()                        # lit le fichier .env (là où se trouve la clé)
API_KEY = os.getenv("ALBERT_API_KEY")
if not API_KEY:
    sys.exit("Il manque la clé API : copiez .env.example en .env et collez votre clé dedans.")

client = OpenAI(base_url="https://albert.api.etalab.gouv.fr/v1", api_key=API_KEY)
MODELE = "gpt-oss-120b"              # lancez 00_lister_les_modeles.py pour voir les autres


# ----------------------------------------------------------------------
# PARTIE 1 : les outils
# ----------------------------------------------------------------------

def pubmed_search(query, max_results=5):
    """La vraie fonction Python : cherche des articles sur PubMed et renvoie titres et résumés."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    try:
        # 1. Première requête : PubMed nous renvoie les identifiants (PMID) des articles les plus pertinents
        r = requests.get(base + "esearch.fcgi", timeout=20, params={
            "db": "pubmed", "term": query, "retmode": "json", "retmax": max_results, "sort": "relevance",
        })
        r.raise_for_status()                         # lève une erreur si le serveur a répondu autre chose que 200
        liste_pmid = r.json()["esearchresult"]["idlist"]
        if not liste_pmid:
            return f"Aucun article trouvé sur PubMed pour '{query}'."

        # 2. Deuxième requête : avec ces identifiants, on récupère les articles complets (en XML)
        r = requests.get(base + "efetch.fcgi", timeout=20, params={
            "db": "pubmed", "id": ",".join(liste_pmid), "retmode": "xml",
        })
        r.raise_for_status()

        # 3. On parcourt le XML pour en tirer le titre, l'année et le résumé de chaque article
        texte = ""
        for article in ET.fromstring(r.content).findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text
            # itertext() récupère tout le texte, même s'il y a de la mise en forme (italique, exposant...)
            titre = "".join(article.find(".//ArticleTitle").itertext())
            annee = article.find(".//PubDate/Year")
            # Un résumé peut être découpé en plusieurs parties (Background, Methods...), on les recolle
            resume = " ".join("".join(p.itertext()) for p in article.findall(".//Abstract/AbstractText"))

            texte += f"PMID {pmid} - {titre} ({annee.text if annee is not None else 'année inconnue'})\n"
            texte += f"{resume[:1500] or 'Pas de résumé disponible'}\n\n"    # on coupe pour ne pas surcharger le modèle
        return texte

    except Exception as erreur:
        # En cas de problème (réseau, XML inattendu...), on renvoie l'erreur au modèle sous forme
        # de texte plutôt que de planter : il pourra l'expliquer à l'utilisateur.
        return f"Erreur pendant la recherche PubMed : {erreur}"


# La description de l'outil, au format attendu par l'API.
# C'est le seul moyen pour le modèle de savoir que l'outil existe et à quoi il sert,
# donc soignez le champ "description" : un outil mal décrit sera mal utilisé.
OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Cherche des articles scientifiques sur PubMed et renvoie leurs titres, auteurs et résumés. "
                           "Utile pour trouver des études récentes, des études cliniques ou des informations "
                           "scientifiques précises sur des virus, des maladies ou des sujets de biologie.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La requête de recherche, de préférence en anglais "
                                       "(par exemple 'influenza vaccine efficacy' ou 'SARS-CoV-2 spike protein').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Nombre maximum d'articles à récupérer (5 maximum recommandé).",
                    },
                },
                "required": ["query"],    # max_results est facultatif
            },
        },
    }
]

# Un petit dictionnaire pour retrouver la fonction Python à partir du nom donné par le modèle.
# Si vous ajoutez un outil, ajoutez-le ici aussi, et dans OUTILS juste au-dessus.
FONCTIONS = {"pubmed_search": pubmed_search}


# ----------------------------------------------------------------------
# PARTIE 2 : la boucle
# ----------------------------------------------------------------------

question =  "Existe il un vaccin contre ebola ?"

messages = [
    {"role": "system", "content": "Tu es un assistant scientifique. Utilise tes outils quand tu as besoin d'informations vérifiées, "
                                  "et cite les articles (titre, année, PMID) sur lesquels tu t'appuies. Réponds en français."},
    {"role": "user", "content": question},
]

# On limite le nombre de tours, au cas où le modèle tournerait en rond
for tour in range(5):
    # 1. On interroge le LLM, en lui donnant la liste des outils
    reponse = client.chat.completions.create(
        model=MODELE,
        messages=messages,
        tools=OUTILS,
        tool_choice="auto",   # le modèle décide lui-même s'il a besoin d'un outil
    )
    message = reponse.choices[0].message

    # 2. Est-ce que le modèle demande un outil ?
    # Non : il a sa réponse finale, on l'affiche et on sort.
    if not message.tool_calls:
        print(message.content)
        break

    # Oui : il veut utiliser un ou plusieurs outils.
    # 3. On garde sa demande dans l'historique de la conversation...
    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": appel.id,
                "type": "function",
                "function": {"name": appel.function.name, "arguments": appel.function.arguments},
            }
            for appel in message.tool_calls
        ],
    })

    # 4. ... puis on exécute chaque outil demandé, et on ajoute le résultat à l'historique avec le rôle "tool".
    # (Le résultat ne va jamais directement à l'utilisateur : c'est le modèle qui le lit et le met en forme.)
    for appel in message.tool_calls:
        nom = appel.function.name
        arguments = json.loads(appel.function.arguments)   # les arguments arrivent sous forme de texte JSON
        print(f"-> le modèle appelle {nom}({arguments})")

        resultat = FONCTIONS[nom](**arguments)

        messages.append({"role": "tool", "tool_call_id": appel.id, "content": resultat})
    # 5. On repart au début de la boucle : le modèle voit maintenant le résultat de l'outil
    # et peut répondre, ou demander un autre outil.
else:
    print("Trop de tours sans réponse finale, on s'arrête.")
