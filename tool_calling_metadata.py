
import json
import os
import sys
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv
from openai import OpenAI


# ----------------------------------------------------------------------
# INITIALISATION
# ----------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("ALBERT_API_KEY")

if not API_KEY:
    sys.exit("Il manque la clé API Albert dans le fichier .env.")

client = OpenAI(
    base_url="https://albert.api.etalab.gouv.fr/v1",
    api_key=API_KEY
)

MODELE = "gpt-oss-120b"


# ----------------------------------------------------------------------
# ARTICLES DISPONIBLES
# ----------------------------------------------------------------------

ARTICLES = {
    "Art.1": "PMC5513205",
    "Art.2": "PMC7479317",
    "Art.3": "PMC5624284",
    "Art.4": "PMC6207063"
}


# ----------------------------------------------------------------------
# OUTIL 1 : RÉCUPÉRER LE TEXTE D'UN ARTICLE DEPUIS EUROPE PMC
# ----------------------------------------------------------------------

def fetch_article(pmcid):
    """
    Récupère le texte intégral d'un article à partir de son PMCID.
    Retourne le titre, le résumé et le corps de l'article.
    """

    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/"
        f"{pmcid}/fullTextXML"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Recherche des éléments XML sans dépendre des namespaces
        def find_element(element, tag):
            for child in element.iter():
                if child.tag.split("}")[-1] == tag:
                    return child
            return None

        def get_text(element):
            if element is None:
                return ""
            return " ".join("".join(element.itertext()).split())

        titre = get_text(find_element(root, "article-title"))
        abstract = get_text(find_element(root, "abstract"))
        body = get_text(find_element(root, "body"))

        if not titre and not body and not abstract:
            return {
                "error": "Aucun texte exploitable trouvé dans le XML.",
                "pmcid": pmcid
            }

        texte = f"""
        PMCID : {pmcid}

        TITRE :
        {titre}

        RÉSUMÉ :
        {abstract}

        TEXTE PRINCIPAL :
        {body}
        """

        return {
            "pmcid": pmcid,
            "title": titre,
            "text": texte
        }

    except requests.exceptions.HTTPError as erreur:
        return {
            "error": f"Article inaccessible ou texte intégral indisponible : {erreur}",
            "pmcid": pmcid
        }

    except Exception as erreur:
        return {
            "error": str(erreur),
            "pmcid": pmcid
        }


# ----------------------------------------------------------------------
# OUTIL 2 : EXTRAIRE LES MÉTADONNÉES
# ----------------------------------------------------------------------

def metadata_extraction(article_id, missing_fields):

    """
    Récupère automatiquement l'article puis demande au LLM
    d'extraire les métadonnées recherchées.
    """

    if article_id not in ARTICLES:
        return json.dumps({
            "error": "Identifiant d'article inconnu."
        }, ensure_ascii=False)

    pmcid = ARTICLES[article_id]

    # Étape 1 : récupération de l'article
    article = fetch_article(pmcid)

    if "error" in article:
        return json.dumps(article, ensure_ascii=False)

    texte_article = article["text"]

    # Limitation de sécurité : à remplacer ensuite par du RAG
    texte_article = texte_article[:60000]

    # Étape 2 : préparation du prompt
    prompt = f"""
Tu es un assistant scientifique spécialisé dans le microbiome
intestinal humain.

Tu dois extraire les métadonnées demandées à partir de l'article
scientifique suivant.

ARTICLE : {article_id}
PMCID : {pmcid}
TITRE : {article["title"]}

MÉTADONNÉES À RECHERCHER :
{json.dumps(missing_fields, ensure_ascii=False)}

TEXTE DE L'ARTICLE :
{texte_article}

CONSIGNES :

1. Extrais uniquement les informations explicitement présentes.
2. N'invente aucune valeur.
3. Utilise null si une information est absente.
4. Fournis une citation exacte du texte comme preuve.
5. Distingue les informations concernant :
   - un échantillon (sample),
   - un individu (individual),
   - un groupe (group),
   - l'ensemble de l'étude (study).
6. Ne transforme pas une moyenne ou une médiane en valeur
   individuelle.
7. Pour les antibiotiques, précise si possible la période,
   la durée et le contexte de l'exposition.
8. Si une information est ambiguë, explique pourquoi.

Réponds uniquement en JSON valide avec cette structure :

{{
  "article_id": "{article_id}",
  "pmcid": "{pmcid}",
  "title": "{article["title"]}",
  "extractions": [
    {{
      "sample_id": null,
      "metadata_field": "host_age",
      "value": null,
      "unit": null,
      "scope": "individual",
      "evidence": null,
      "status": "absent",
      "comment": null
    }}
  ]
}}
"""

    # Étape 3 : interrogation du LLM
    try:
        response = client.chat.completions.create(
            model=MODELE,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    except Exception as erreur:
        return json.dumps({
            "error": f"Erreur pendant l'extraction : {str(erreur)}"
        }, ensure_ascii=False)


# ----------------------------------------------------------------------
# DESCRIPTION DES OUTILS POUR LE LLM
# ----------------------------------------------------------------------

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "metadata_extraction",
            "description": (
                "Récupère automatiquement un article scientifique "
                "depuis Europe PMC grâce à son identifiant, puis "
                "extrait les métadonnées demandées. "
                "À utiliser pour analyser les articles sur le "
                "microbiome intestinal humain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "string",
                        "enum": list(ARTICLES.keys()),
                        "description": "Identifiant de l'article à analyser."
                    },
                    "missing_fields": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Métadonnées recherchées : host_age, "
                            "host_sex, antibiotic_exposure, etc."
                        )
                    }
                },
                "required": ["article_id", "missing_fields"]
            }
        }
    }
]


FONCTIONS = {
    "metadata_extraction": metadata_extraction
}


# ----------------------------------------------------------------------
# PARTIE 3 : BOUCLE DE TOOL CALLING
# ----------------------------------------------------------------------

question = input(
    "\nQue souhaites-tu extraire des articles ?\n> "
)

messages = [
    {
        "role": "system",
        "content": (
            "Tu es un assistant scientifique spécialisé dans "
            "le microbiome intestinal humain. "
            f"Les articles disponibles sont : {list(ARTICLES.keys())}. "
            "Lorsque l'utilisateur demande une extraction, utilise "
            "metadata_extraction pour analyser chaque article pertinent. "
            "Compare les résultats et présente les preuves textuelles. "
            "Ne fabrique jamais de données. Réponds en français."
        )
    },
    {
        "role": "user",
        "content": question
    }
]


for tour in range(8):

    reponse = client.chat.completions.create(
        model=MODELE,
        messages=messages,
        tools=OUTILS,
        tool_choice="auto",
        temperature=0
    )

    message = reponse.choices[0].message

    if not message.tool_calls:
        print("\n--- RÉSULTAT FINAL ---\n")
        print(message.content)
        break

    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": appel.id,
                "type": "function",
                "function": {
                    "name": appel.function.name,
                    "arguments": appel.function.arguments
                }
            }
            for appel in message.tool_calls
        ]
    })

    for appel in message.tool_calls:

        nom = appel.function.name
        arguments = json.loads(appel.function.arguments)

        print(f"\n-> Appel de {nom} : {arguments}")

        if nom not in FONCTIONS:
            resultat = json.dumps({"error": "Outil inconnu."})
        else:
            try:
                resultat = FONCTIONS[nom](**arguments)
            except Exception as erreur:
                resultat = json.dumps({
                    "error": str(erreur)
                }, ensure_ascii=False)

        messages.append({
            "role": "tool",
            "tool_call_id": appel.id,
            "content": str(resultat)
        })

else:
    print("Nombre maximal de tours atteint.")
