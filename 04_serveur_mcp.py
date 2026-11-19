"""
Script 04 : un serveur MCP minimal, sur le thème de la littérature scientifique.

MCP (Model Context Protocol) est un standard pour "brancher" des outils sur un LLM.
Dans le script 03, l'outil était écrit directement dans le script. Ici il vit dans
un serveur à part : n'importe quelle application compatible MCP peut s'y connecter
(Claude Desktop, VS Code, ou votre propre script, voir le 05).

Un serveur MCP peut proposer trois choses différentes :
  - des TOOLS : des fonctions que le modèle décide d'appeler (comme au script 03)
  - des RESOURCES : des données qu'on peut lire à partir d'une adresse (URI), sans rien exécuter
  - des PROMPTS : des modèles de consignes réutilisables, avec des paramètres

Ce serveur en propose deux de chaque :
  tools     : pubmed_search, europepmc_search
  resources : doc://syntaxe-pubmed, pubmed://article/{pmid}
  prompts   : revue_de_litterature, analyser_article

Ce script est un serveur : il se lance une fois, reste allumé, et attend qu'on
vienne l'interroger. On le laisse donc tourner dans son propre terminal :

    python 04_serveur_mcp.py

Sur une machine distante, ou pour qu'il continue à tourner quand on ferme
le terminal, on peut le lancer dans un "screen" :

    screen -S mcp               # ouvre un screen nommé "mcp"
    python 04_serveur_mcp.py    # on lance le serveur dedans
    # Ctrl+A puis D             # on "détache" le screen, le serveur continue
    screen -r mcp               # pour y revenir (Ctrl+C pour arrêter le serveur)

Le serveur écoute alors sur http://localhost:8000/mcp
Dans un autre terminal, on lance le client : 05_client_mcp.py

Pour le tester à la main, avec le serveur lancé, il y a un inspecteur dans le navigateur :
    fastmcp dev inspector 04_serveur_mcp.py
"""

import re
import xml.etree.ElementTree as ET
from typing import Literal

import requests
from pydantic import Field

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message

# On crée le serveur, avec un nom
mcp = FastMCP("litterature-scientifique")

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ENTETES = {"User-Agent": "HackathonIA/1.0"}


# ----------------------------------------------------------------------
# Fonctions utilitaires (pas exposées au modèle, juste là pour éviter de se répéter)
# ----------------------------------------------------------------------

def appeler(url, params):
    """Fait une requête GET. En cas de problème, lève une ToolError : le message revient au modèle."""
    try:
        reponse = requests.get(url, params=params, headers=ENTETES, timeout=20)
        reponse.raise_for_status()
        return reponse
    except requests.RequestException as erreur:
        raise ToolError(f"Impossible de contacter le service : {erreur}")


def lire_articles_pubmed(liste_pmid):
    """Récupère les articles PubMed correspondant à une liste de PMID, et les met en forme en texte."""
    reponse = appeler(NCBI + "efetch.fcgi", {"db": "pubmed", "id": ",".join(liste_pmid), "retmode": "xml"})

    texte = ""
    for article in ET.fromstring(reponse.content).findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text
        # itertext() récupère tout le texte, même s'il y a de la mise en forme (italique, exposant...)
        titre = "".join(article.find(".//ArticleTitle").itertext())
        annee = article.find(".//PubDate/Year")
        # Un résumé peut être découpé en plusieurs parties (Background, Methods...), on les recolle
        resume = " ".join("".join(p.itertext()) for p in article.findall(".//Abstract/AbstractText"))

        texte += f"PMID {pmid} - {titre} ({annee.text if annee is not None else 'année inconnue'})\n"
        texte += f"{resume[:1500] or 'Pas de résumé disponible'}\n\n"    # on coupe pour ne pas surcharger le modèle
    return texte


# ----------------------------------------------------------------------
# TOOLS : ce que le modèle peut appeler
# ----------------------------------------------------------------------
# Le décorateur @mcp.tool transforme la fonction en outil MCP.
# Le nom, les types des arguments et la docstring (le texte entre guillemets
# juste en dessous) sont envoyés au LLM : c'est tout ce qu'il voit de l'outil,
# donc il vaut mieux les soigner. Field(5, ge=1, le=10) veut dire "5 par défaut,
# entre 1 et 10" : FastMCP refuse automatiquement les valeurs hors limites.

@mcp.tool
def pubmed_search(query: str, max_results: int = Field(5, ge=1, le=10)) -> str:
    """Cherche des articles scientifiques sur PubMed (biologie, médecine) et renvoie titres et résumés.

    Args:
        query: La requête, de préférence en anglais (par exemple "influenza vaccine efficacy").
        max_results: Nombre maximum d'articles à récupérer.
    """
    # 1. PubMed nous renvoie d'abord les identifiants (PMID) des articles les plus pertinents
    reponse = appeler(NCBI + "esearch.fcgi", {
        "db": "pubmed", "term": query, "retmode": "json", "retmax": max_results, "sort": "relevance",
    })
    liste_pmid = reponse.json()["esearchresult"]["idlist"]
    if not liste_pmid:
        return f"Aucun article trouvé sur PubMed pour '{query}'."

    # 2. Puis on va chercher le contenu de ces articles
    return lire_articles_pubmed(liste_pmid)


@mcp.tool
def europepmc_search(query: str, max_results: int = Field(5, ge=1, le=10)) -> str:
    """Cherche dans Europe PMC, qui couvre PubMed mais aussi les preprints (bioRxiv, medRxiv...) et des brevets.

    À utiliser en complément de pubmed_search, notamment pour trouver des travaux très récents
    pas encore publiés dans une revue.

    Args:
        query: La requête, de préférence en anglais (par exemple "CRISPR base editing").
        max_results: Nombre maximum d'articles à récupérer.
    """
    # resultType=core : on demande la version complète, qui contient le résumé
    reponse = appeler(EUROPEPMC, {
        "query": query, "format": "json", "pageSize": max_results, "resultType": "core",
    })
    resultats = reponse.json()["resultList"]["result"]
    if not resultats:
        return f"Aucun article trouvé sur Europe PMC pour '{query}'."

    texte = ""
    for r in resultats:
        # Titres et résumés d'Europe PMC contiennent parfois des balises HTML, qu'on enlève
        titre = re.sub(r"<[^>]+>", "", r.get("title", "Sans titre"))
        resume = re.sub(r"<[^>]+>", "", r.get("abstractText", ""))
        identifiant = f"PMID {r['pmid']}" if "pmid" in r else f"DOI {r.get('doi', 'inconnu')}"
        texte += f"{identifiant} - {titre} ({r.get('pubYear', 'année inconnue')})\n"
        texte += f"Source : {r.get('journalTitle') or r.get('source', 'inconnue')}\n"
        texte += f"{resume[:1500] or 'Pas de résumé disponible'}\n\n"
    return texte


# ----------------------------------------------------------------------
# RESOURCES : des données qu'on lit à partir d'une adresse (URI)
# ----------------------------------------------------------------------
# Une ressource ne fait rien d'autre que donner du contenu. L'adresse est libre :
# ici "doc://..." et "pubmed://..." sont juste des noms que nous avons choisis.

@mcp.resource("doc://syntaxe-pubmed")
def syntaxe_pubmed() -> str:
    """Aide-mémoire pour écrire de bonnes requêtes PubMed."""
    return """Syntaxe des requêtes PubMed

- Opérateurs : AND, OR, NOT (en majuscules). Exemple : influenza AND vaccine NOT pediatric
- Expression exacte : entre guillemets. Exemple : "spike protein"
- Limiter à un champ : mot suivi du champ entre crochets
    [tiab]  titre ou résumé       exemple : vaccine[tiab]
    [mh]    terme MeSH (vocabulaire contrôlé)   exemple : "Influenza, Human"[mh]
    [au]    auteur                exemple : Pasteur L[au]
    [pt]    type de publication   exemple : review[pt]
- Dates : année ou intervalle avec [dp]. Exemple : 2020:2024[dp]
- Les requêtes en anglais donnent de bien meilleurs résultats qu'en français.
"""


# Une ressource avec {pmid} dans l'adresse est un "template" : l'argument de la fonction
# est pris dans l'adresse demandée. Lire pubmed://article/12345 appelle article_pubmed("12345").
@mcp.resource("pubmed://article/{pmid}")
def article_pubmed(pmid: str) -> str:
    """Titre et résumé d'un article PubMed à partir de son PMID."""
    return lire_articles_pubmed([pmid])


# ----------------------------------------------------------------------
# PROMPTS : des consignes réutilisables
# ----------------------------------------------------------------------
# Un prompt est un modèle de message, avec des paramètres. C'est utile pour ne pas
# réécrire à chaque fois les mêmes consignes, et pour garder une façon de répondre
# cohérente. Le client choisit un prompt, donne ses paramètres, et obtient le texte prêt à envoyer.

@mcp.prompt
def revue_de_litterature(sujet: str, niveau: Literal["grand public", "étudiant", "expert"] = "étudiant") -> str:
    """Demande une courte revue de littérature sur un sujet, adaptée à un niveau de lecteur."""
    return f"""Fais une courte revue de la littérature scientifique sur : {sujet}

Procède dans cet ordre :
1. Réfléchis d'abord à 2 ou 3 requêtes en anglais, avec des mots-clés différents.
2. Lance-les avec pubmed_search et europepmc_search.
3. Lis les résumés et garde les 3 à 5 articles les plus utiles.

Structure de ta réponse :
- Contexte (2 phrases)
- Ce que disent les articles (3 points clés)
- Limites et désaccords entre les études
- Références : titre, année, PMID ou DOI

Règles :
- Appuie-toi uniquement sur les articles trouvés. Si les résultats ne permettent pas de conclure, dis-le.
- Distingue ce qui est publié dans une revue de ce qui est un preprint.
- Adapte le vocabulaire à un lecteur de niveau : {niveau}.
- Réponds en français."""


# Ici le prompt est une vraie petite conversation : on peut même écrire le début
# de la réponse de l'assistant pour lui donner le ton.
@mcp.prompt
def analyser_article(pmid: str) -> list[Message]:
    """Demande une lecture critique d'un article PubMed à partir de son PMID."""
    return [
        Message(
            f"Fais une lecture critique de l'article PubMed {pmid} (utilise la ressource pubmed://article/{pmid}). "
            "Donne : la question posée, la méthode, les résultats principaux, puis les limites de l'étude. "
            "Si le résumé ne suffit pas pour juger un point, dis-le plutôt que de deviner."
        ),
        Message("D'accord, je commence par lire l'article.", role="assistant"),
    ]


if __name__ == "__main__":
    # Transport "http" : le serveur écoute sur un port, et n'importe quel client peut s'y connecter
    # en donnant l'adresse http://localhost:8000/mcp
    # Les deux options en plus servent à simplifier les échanges :
    #   json_response=True : les réponses sont du JSON classique (et pas un flux d'événements)
    #   stateless_http=True : pas de session à ouvrir, chaque requête est indépendante
    # Résultat : on peut interroger le serveur avec un simple requests.post (voir 05_client_mcp.py).
    # Attention : le serveur n'a aucune authentification. Il écoute seulement sur 127.0.0.1
    # (la machine elle-même). Ne mettez pas host="0.0.0.0" sur un réseau qui n'est pas de confiance.
    mcp.run(transport="http", host="127.0.0.1", port=8000, json_response=True, stateless_http=True)