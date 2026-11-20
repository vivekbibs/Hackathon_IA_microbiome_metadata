"""Extract cohort-level disease and drug metadata from an Europe PMC article.

Usage:
    python 08_europepmc_cohort_metadata.py PMC5513205

When a cohort contains different subgroups, the global field is "mixed" and
the subgroup-specific values are listed in the "subgroups" array.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
API_KEY = os.getenv("ALBERT_API_KEY")
if not API_KEY:
    sys.exit("Il manque la clé API Albert dans le fichier .env.")

client = OpenAI(
    base_url="https://albert.api.etalab.gouv.fr/v1",
    api_key=API_KEY,
)
MODELE = "gpt-oss-120b"
MODELE_EMBEDDINGS = "bge-m3"
TAILLE_MORCEAU = 800
CHEVAUCHEMENT = 150
NB_MORCEAUX = 8


def local_name(element):
    return element.tag.split("}")[-1]


def get_element_text(root, tag):
    for element in root.iter():
        if local_name(element) == tag:
            return " ".join("".join(element.itertext()).split())
    return ""


def fetch_article(pmcid):
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/" + pmcid + "/fullTextXML"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    title = get_element_text(root, "article-title")
    abstract = get_element_text(root, "abstract")
    body = get_element_text(root, "body")
    article_text = "\n\n".join(
        part for part in (
            f"TITLE: {title}" if title else "",
            f"ABSTRACT: {abstract}" if abstract else "",
            f"BODY: {body}" if body else "",
        ) if part
    )
    if not article_text:
        raise ValueError("Aucun texte exploitable trouvé dans le XML.")

    return title, article_text


def retrieve_relevant_passages(article_text):
    chunks = [
        article_text[start:start + TAILLE_MORCEAU].strip()
        for start in range(0, len(article_text), TAILLE_MORCEAU - CHEVAUCHEMENT)
    ]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        raise ValueError("Aucun passage n'a pu être préparé pour le RAG.")

    vectors = []
    for start in range(0, len(chunks), 32):
        response = client.embeddings.create(
            model=MODELE_EMBEDDINGS,
            input=chunks[start:start + 32],
        )
        vectors.extend(item.embedding for item in response.data)

    matrix = np.array(vectors)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    query = (
        "Study cohort population, diagnoses and disease status, healthy controls, "
        "patient subgroups, and medications or drugs administered as treatment."
    )
    query_vector = client.embeddings.create(
        model=MODELE_EMBEDDINGS,
        input=[query],
    ).data[0].embedding
    query_vector = np.array(query_vector)
    query_vector = query_vector / np.linalg.norm(query_vector)

    scores = matrix @ query_vector
    best_chunks = np.argsort(scores)[::-1][:NB_MORCEAUX]
    return "\n\n".join(
        f"Passage {index + 1}:\n{chunks[index]}" for index in best_chunks
    )


def extract_metadata(pmcid, title, article_text):
    relevant_passages = retrieve_relevant_passages(article_text)
    prompt = f"""Extract metadata for the STUDY COHORT as a whole from this scientific article.

PMCID: {pmcid}
Title: {title}

Return only valid JSON with exactly this structure:
{{
  "pmcid": "{pmcid}",
  "title": "article title",
  "disease_status": "healthy | exact disease name | mixed | unknown",
  "drug": "none | exact active drug name | mixed | unknown",
  "subgroups": [
    {{
      "name": "group name as described in the article",
      "disease_status": "healthy | exact disease name | unknown",
      "drug": "none | exact active drug name | unknown"
    }}
  ]
}}

Rules:
- Do not extract patient-by-patient data. The two top-level fields describe the whole
  study cohort.
- If the cohort has subgroups with different disease statuses or drugs, use "mixed"
  for the corresponding top-level field and list the relevant groups in "subgroups".
- For a uniform cohort, use its value at the top level; "subgroups" may be empty.
- Use "healthy" only for a healthy/control group. Otherwise use the exact disease name
  stated in the paper. Use "unknown" when it cannot be determined.
- For drug, return the exact active medication name, not a class or a guessed ingredient.
  Use "none" for an explicitly untreated/no-active-drug group (including placebo
  controls), and "unknown" if the article does not provide enough information.
- Include only subgroups relevant to disease status or drug exposure. Do not invent
  subgroup names or values.
- Do not confuse a drug mentioned as background, an exclusion criterion, or a proposed
  treatment with a drug administered to the study cohort.
- Use only the evidence in the passages below. If they do not provide enough evidence,
    use "unknown" rather than guessing.

RELEVANT ARTICLE PASSAGES:
{relevant_passages}
"""

    response = client.chat.completions.create(
        model=MODELE,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    data = json.loads(content)
    validate_result(data, pmcid, title)
    return data


def validate_result(data, pmcid, title):
    if not isinstance(data, dict):
        raise ValueError("La réponse du modèle n'est pas un objet JSON.")

    data["pmcid"] = pmcid
    data["title"] = title
    for field in ("disease_status", "drug"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError(f"Champ manquant ou invalide dans le JSON : {field}")

    subgroups = data.get("subgroups", [])
    if not isinstance(subgroups, list):
        raise ValueError("Le champ 'subgroups' doit être une liste.")
    for subgroup in subgroups:
        if not isinstance(subgroup, dict) or not all(
            isinstance(subgroup.get(field), str) and subgroup[field].strip()
            for field in ("name", "disease_status", "drug")
        ):
            raise ValueError("Un sous-groupe ne respecte pas le format JSON attendu.")
    data["subgroups"] = subgroups


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage : python 08_europepmc_cohort_metadata.py <PMCID>")

    pmcid = sys.argv[1]
    try:
        title, article_text = fetch_article(pmcid)
        result = extract_metadata(pmcid, title, article_text)
        filename = f"results_{pmcid}_cohort.json"
        with open(filename, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nRésultat sauvegardé dans : {filename}")
    except (requests.RequestException, ET.ParseError, json.JSONDecodeError, ValueError) as error:
        sys.exit(f"Erreur : {error}")
    except Exception as error:
        sys.exit(f"Erreur pendant l'extraction : {error}")


if __name__ == "__main__":
    main()