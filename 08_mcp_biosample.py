"""
Script 08 : serveur MCP pour compléter les métadonnées BioSample avec l'article associé.

Outils exposés (FastMCP) :
  - lister_champs()                     : champs de métadonnées ciblés
  - get_biosample(accession)            : métadonnées d'un échantillon (API EBI BioSamples)
  - audit_project(project)              : complétude des champs ciblés pour un BioProject
  - extract_from_article(pmcid, champs) : extraction depuis l'article (RAG + Albert)
  - complete_project(project, pmcid)    : audit + extraction + CSV complété avec provenance

Principes :
  - Une valeur BioSample existante n'est JAMAIS écrasée.
  - Les valeurs venant de l'article sont des valeurs de COHORTE (par exemple « âge médian 39 »),
    écrites dans des colonnes séparées (`<champ>_article`). Elles ne sont pas rattachées à un
    échantillon précis : les données individuelles sont anonymisées.
  - Chaque extraction garde la phrase source et un indicateur « phrase retrouvée dans le texte ».

Lancement (transport stdio, utilisé par OpenCode) :
    python 08_mcp_biosample.py

Prérequis : variable ALBERT_API_KEY, paquets openai, numpy, requests, python-dotenv, fastmcp.
Les journaux vont sur stderr : stdout est réservé au protocole MCP.
"""

import csv
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import OpenAI

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("mcp-biosample")

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("ALBERT_API_KEY")
if not API_KEY:
    sys.exit("Il manque la variable d'environnement ALBERT_API_KEY.")

client = OpenAI(base_url="https://albert.api.etalab.gouv.fr/v1", api_key=API_KEY)
MODELE = os.getenv("ALBERT_MODELE", "gpt-oss-120b")
MODELE_EMBEDDINGS = os.getenv("ALBERT_EMBEDDINGS", "bge-m3")

ENA = "https://www.ebi.ac.uk/ena/portal/api"
BIOSAMPLES = "https://www.ebi.ac.uk/biosamples/samples"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

TAILLE_MORCEAU = 800
CHEVAUCHEMENT = 150
NB_MORCEAUX = 5
MAX_ECHANTILLONS = 50

DATA_DIR = Path(os.getenv("MCP_DATA_DIR", "data_mcp"))
CACHE_DIR = DATA_DIR / "cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Champs ciblés : question posée à l'article + noms d'attributs BioSample équivalents.
CHAMPS = {
    "taille_cohorte": {
        "question": "Quelle est la taille de la cohorte étudiée (nombre de sujets) ?",
        "attributs": [],
    },
    "sexe_cohorte": {
        "question": "Quelle est la répartition par sexe de la cohorte ?",
        "attributs": ["sex", "host sex", "gender"],
    },
    "age_cohorte": {
        "question": "Quel est l'âge des participants (plage, moyenne ou médiane) ?",
        "attributs": ["age", "host age"],
    },
    "pays": {
        "question": "Dans quel pays ou quelle région les échantillons ont-ils été collectés ?",
        "attributs": ["geo loc name", "country"],
    },
    "pathologie": {
        "question": "Quel est l'état de santé ou la pathologie des participants ?",
        "attributs": ["disease", "host disease", "health state", "host health state"],
    },
    "traitement_antibiotique": {
        "question": "Les participants ont-ils pris des antibiotiques, et dans quelles conditions ?",
        "attributs": ["antibiotic", "antibiotics", "antibiotic usage", "antimicrobial"],
    },
    "regime": {
        "question": "Quel est le régime alimentaire des participants ?",
        "attributs": ["diet", "host diet"],
    },
}

VALEURS_VIDES = {
    "", "na", "n/a", "nan", "none", "null", "missing", "unknown", "not provided",
    "not applicable", "not collected", "not available", "restricted access", "-", "?",
}

RE_PROJET = re.compile(r"^PRJ(EB|NA|DB)\d+$")
RE_PMCID = re.compile(r"^PMC\d+$")
RE_SAMPLE = re.compile(r"^SAM[END][A-Z]?\d+$")

mcp = FastMCP("biosample-completion")


# ---------- Outils réseau ----------
def http_get(url, params=None, headers=None, timeout=120, essais=3):
    """GET avec nouvelles tentatives sur les erreurs temporaires (429, 5xx, réseau)."""
    derniere = None
    for essai in range(essais):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                derniere = f"HTTP {r.status_code}"
            else:
                return r
        except requests.exceptions.RequestException as e:
            derniere = str(e)
        log.warning("Échec (%s) pour %s, nouvel essai", derniere, url)
        time.sleep(2 ** essai)
    raise RuntimeError(f"Requête impossible : {url} ({derniere})")


def normaliser_nom(nom):
    return re.sub(r"[\s_\-]+", " ", nom.lower()).strip()


def valeur_renseignee(valeur):
    return valeur is not None and str(valeur).strip().lower() not in VALEURS_VIDES


def verifier_projet(project):
    project = project.strip().upper()
    if not RE_PROJET.match(project):
        raise ValueError(f"Accession de projet invalide : {project} (attendu : PRJEB…, PRJNA… ou PRJDB…)")
    return project


def verifier_pmcid(pmcid):
    pmcid = pmcid.strip().upper()
    if not RE_PMCID.match(pmcid):
        raise ValueError(f"PMCID invalide : {pmcid} (attendu : PMC suivi de chiffres)")
    return pmcid


# ---------- BioSample / ENA ----------
def lister_echantillons(project):
    """Accessions des échantillons d'un projet, via l'API ENA Portal."""
    r = http_get(
        f"{ENA}/search",
        params={
            "result": "sample",
            "query": f'study_accession="{project}"',
            "fields": "sample_accession,sample_alias",
            "format": "tsv",
            "limit": 0,
        },
    )
    if not r.ok:
        raise RuntimeError(f"Erreur ENA {r.status_code} : {r.text[:300]}")
    lignes = r.text.strip().splitlines()
    if len(lignes) < 2:
        return []
    entete = lignes[0].split("\t")
    return [dict(zip(entete, ligne.split("\t"))) for ligne in lignes[1:]]


def lire_biosample(accession):
    """Métadonnées d'un échantillon, aplaties : {nom_attribut: valeur}."""
    r = http_get(f"{BIOSAMPLES}/{accession}", headers={"Accept": "application/json"})
    if r.status_code == 404:
        raise ValueError(f"Échantillon introuvable dans BioSamples : {accession}")
    if not r.ok:
        raise RuntimeError(f"Erreur BioSamples {r.status_code} pour {accession}")
    js = r.json()
    attributs = {}
    for nom, valeurs in (js.get("characteristics") or {}).items():
        textes = []
        for v in valeurs:
            t = str(v.get("text", "")).strip()
            if v.get("unit"):
                t += f" {v['unit']}"
            if t:
                textes.append(t)
        attributs[nom] = "; ".join(textes)
    return {"accession": js.get("accession", accession), "name": js.get("name"), "attributs": attributs}


def valeur_du_champ(attributs, champ):
    """Première valeur renseignée d'un échantillon pour un champ ciblé (ou None)."""
    noms = {normaliser_nom(n): v for n, v in attributs.items()}
    for nom in CHAMPS[champ]["attributs"]:
        v = noms.get(nom)
        if valeur_renseignee(v):
            return v
    return None


def collecter_echantillons(project, max_echantillons):
    liste = lister_echantillons(project)
    total = len(liste)
    resultats, erreurs = [], []
    for item in liste[:max_echantillons]:
        acc = item.get("sample_accession", "")
        if not RE_SAMPLE.match(acc):
            continue
        try:
            resultats.append(lire_biosample(acc))
        except Exception as e:  # une erreur sur un échantillon ne bloque pas le reste
            erreurs.append({"accession": acc, "erreur": str(e)})
        time.sleep(0.1)
    return total, resultats, erreurs


def calculer_audit(echantillons):
    audit = {}
    n = len(echantillons)
    for champ, cfg in CHAMPS.items():
        if not cfg["attributs"]:
            continue
        remplis = sum(1 for e in echantillons if valeur_du_champ(e["attributs"], champ) is not None)
        audit[champ] = {
            "renseignes": remplis,
            "total": n,
            "pct_complet": round(100 * remplis / n, 1) if n else 0.0,
        }
    return audit


# ---------- Article (Europe PMC + RAG) ----------
INDEX_ARTICLES = {}  # pmcid -> (passages, vecteurs)


def nettoyer(texte):
    return re.sub(r"\s+", " ", texte).strip()


def texte_article(pmcid):
    """Texte complet de l'article (XML Europe PMC, avec cache local)."""
    chemin = CACHE_DIR / f"{pmcid}.xml"
    if chemin.exists():
        xml = chemin.read_bytes()
    else:
        r = http_get(f"{EPMC}/{pmcid}/fullTextXML", headers={"User-Agent": "mcp-biosample/1.0"})
        if not r.ok:
            raise RuntimeError(f"Europe PMC a répondu {r.status_code} pour {pmcid} (article open access ?)")
        xml = r.content
        chemin.write_bytes(xml)
    racine = ET.fromstring(xml)
    parties = []
    titre = racine.find(".//article-title")
    if titre is not None:
        parties.append(nettoyer("".join(titre.itertext())))
    for p in racine.iter("p"):
        t = nettoyer(" ".join(p.itertext()))
        if t:
            parties.append(t)
    for tw in racine.iter("table-wrap"):
        for tr in tw.iter("tr"):
            cellules = [nettoyer(" ".join(c.itertext())) for c in tr if c.tag in ("td", "th")]
            if any(cellules):
                parties.append(" | ".join(cellules))
    return "\n".join(parties)


def decouper(texte):
    pas = TAILLE_MORCEAU - CHEVAUCHEMENT
    return [m for m in (texte[i:i + TAILLE_MORCEAU].strip() for i in range(0, len(texte), pas)) if m]


def encoder(textes):
    vecteurs = []
    for i in range(0, len(textes), 32):
        rep = client.embeddings.create(model=MODELE_EMBEDDINGS, input=textes[i:i + 32])
        vecteurs.extend(d.embedding for d in rep.data)
    m = np.array(vecteurs)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def indexer_article(pmcid):
    if pmcid not in INDEX_ARTICLES:
        passages = decouper(texte_article(pmcid))
        if not passages:
            raise RuntimeError("Aucun texte exploitable dans l'article.")
        INDEX_ARTICLES[pmcid] = (passages, encoder(passages))
    return INDEX_ARTICLES[pmcid]


def comparable(texte):
    return re.sub(r"\s+", " ", texte).lower().strip()


def lire_json(texte):
    m = re.search(r"\{.*\}", texte or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


CONSIGNE = (
    "Tu es un assistant d'extraction de métadonnées scientifiques. "
    "Réponds UNIQUEMENT avec un objet JSON de la forme "
    '{"valeur": <texte court ou null>, "phrase_source": <phrase copiée mot pour mot du contexte ou null>, '
    '"niveau": "cohorte" | "echantillon" | "absent"}. '
    "Utilise uniquement le contexte fourni, sans rien déduire. "
    "Si l'information n'y figure pas : valeur null, phrase_source null, niveau \"absent\". "
    "niveau = \"cohorte\" si la valeur décrit le groupe entier (effectif, médiane, répartition), "
    "\"echantillon\" si elle décrit un échantillon ou un sujet précis."
)


def extraire_champ(pmcid, champ):
    passages, vecteurs = indexer_article(pmcid)
    question = CHAMPS[champ]["question"]
    q = encoder([question])[0]
    meilleurs = np.argsort(vecteurs @ q)[::-1][:NB_MORCEAUX]
    contexte = "\n\n".join(f"Passage {i}:\n{passages[i]}" for i in meilleurs)
    rep = client.chat.completions.create(
        model=MODELE,
        temperature=0,
        messages=[
            {"role": "system", "content": CONSIGNE},
            {"role": "user", "content": f"Contexte :\n{contexte}\n\nQuestion : {question}"},
        ],
    )
    brut = rep.choices[0].message.content
    donnees = lire_json(brut) or {}
    phrase = donnees.get("phrase_source")
    return {
        "champ": champ,
        "valeur": donnees.get("valeur"),
        "niveau": donnees.get("niveau", "absent") if donnees else "illisible",
        "phrase_source": phrase,
        "phrase_retrouvee": bool(phrase) and comparable(phrase) in comparable(contexte),
        "passages": [int(i) for i in meilleurs],
    }


def choisir_champs(champs):
    if not champs:
        return list(CHAMPS)
    inconnus = [c for c in champs if c not in CHAMPS]
    if inconnus:
        raise ValueError(f"Champs inconnus : {inconnus}. Disponibles : {list(CHAMPS)}")
    return list(champs)


# ---------- Outils MCP ----------
@mcp.tool
def lister_champs() -> dict:
    """Liste les champs de métadonnées ciblés et les attributs BioSample équivalents."""
    return {c: {"question": cfg["question"], "attributs_biosample": cfg["attributs"]} for c, cfg in CHAMPS.items()}


@mcp.tool
def get_biosample(accession: str) -> dict:
    """Métadonnées d'un échantillon BioSample (ex. SAMEA53404168), via l'API EBI BioSamples."""
    accession = accession.strip().upper()
    if not RE_SAMPLE.match(accession):
        raise ValueError(f"Accession BioSample invalide : {accession}")
    return lire_biosample(accession)


@mcp.tool
def audit_project(project: str, max_samples: int = 20) -> dict:
    """Complétude des champs ciblés (âge, sexe, pays, pathologie, antibiotiques, régime)
    pour un BioProject (ex. PRJEB19090). Lit au plus `max_samples` échantillons."""
    project = verifier_projet(project)
    max_samples = max(1, min(max_samples, MAX_ECHANTILLONS))
    total, echantillons, erreurs = collecter_echantillons(project, max_samples)
    return {
        "projet": project,
        "echantillons_dans_projet": total,
        "echantillons_lus": len(echantillons),
        "audit": calculer_audit(echantillons),
        "erreurs": erreurs,
    }


@mcp.tool
def extract_from_article(pmcid: str, champs: list[str] | None = None) -> list[dict]:
    """Extrait des métadonnées d'un article Europe PMC (ex. PMC5513205) avec le RAG Albert.
    Chaque résultat contient la valeur, le niveau (cohorte, echantillon ou absent) et la phrase source."""
    pmcid = verifier_pmcid(pmcid)
    return [extraire_champ(pmcid, c) for c in choisir_champs(champs)]


@mcp.tool
def complete_project(project: str, pmcid: str, max_samples: int = 20) -> dict:
    """Audite un BioProject, extrait de l'article les champs vides dans BioSample, puis écrit un CSV.
    Les valeurs BioSample existantes ne sont jamais écrasées. Les valeurs de l'article sont de niveau
    cohorte, dans des colonnes `<champ>_article`, avec la provenance dans un fichier JSON."""
    project = verifier_projet(project)
    pmcid = verifier_pmcid(pmcid)
    max_samples = max(1, min(max_samples, MAX_ECHANTILLONS))

    total, echantillons, erreurs = collecter_echantillons(project, max_samples)
    if not echantillons:
        raise RuntimeError(f"Aucun échantillon lu pour {project}. Erreurs : {erreurs[:3]}")
    audit = calculer_audit(echantillons)

    # Champs à compléter : ceux qui ne sont renseignés pour aucun échantillon, et la taille de cohorte.
    a_completer = [c for c, cfg in CHAMPS.items() if not cfg["attributs"] or audit[c]["renseignes"] == 0]
    extractions = {}
    for champ in a_completer:
        try:
            extractions[champ] = extraire_champ(pmcid, champ)
        except Exception as e:
            extractions[champ] = {"champ": champ, "valeur": None, "niveau": "erreur", "erreur": str(e)}

    colonnes = ["accession", "name"]
    colonnes += [c for c, cfg in CHAMPS.items() if cfg["attributs"]]
    colonnes += [f"{c}_article" for c in a_completer] + [f"{c}_niveau" for c in a_completer]
    chemin_csv = DATA_DIR / f"completed_{project}_{pmcid}.csv"
    with open(chemin_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        for e in echantillons:
            ligne = {"accession": e["accession"], "name": e["name"]}
            for c, cfg in CHAMPS.items():
                if cfg["attributs"]:
                    ligne[c] = valeur_du_champ(e["attributs"], c) or ""
            for c in a_completer:
                x = extractions[c]
                ligne[f"{c}_article"] = x.get("valeur") or ""
                ligne[f"{c}_niveau"] = x.get("niveau", "")
            w.writerow(ligne)

    chemin_prov = DATA_DIR / f"provenance_{project}_{pmcid}.json"
    chemin_prov.write_text(json.dumps(extractions, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "projet": project,
        "article": pmcid,
        "echantillons_dans_projet": total,
        "echantillons_lus": len(echantillons),
        "audit_avant": audit,
        "champs_completes": {c: x.get("valeur") for c, x in extractions.items()},
        "phrases_non_retrouvees": [c for c, x in extractions.items() if x.get("valeur") and not x.get("phrase_retrouvee")],
        "fichier_csv": str(chemin_csv),
        "fichier_provenance": str(chemin_prov),
        "erreurs": erreurs,
        "remarque": "Valeurs de l'article = niveau cohorte, non rattachées à un échantillon précis.",
    }


if __name__ == "__main__":
    mcp.run()
