"""
Script 07 : RAG à partir d'un article EuropePMC avec extraction de données de cohorte.

Ce script permet de :
1. Récupérer un article scientifique au format XML via l'API EuropePMC.
2. Extraire les métadonnées (titre, journal, date, auteurs).
3. Extraire le texte complet pour l'utiliser dans un système de RAG.
4. Extraire automatiquement des informations de cohorte et les sauvegarder en CSV.
5. Poser des questions sur l'article.

Usage:
    python 07_europepmc_rag.py PMC5513205
"""

import os
import sys
import requests
import numpy as np
import xml.etree.ElementTree as ET
import csv
import re
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("ALBERT_API_KEY")
if not API_KEY:
    sys.exit("Il manque la clé API : copiez .env.example en .env et collez votre clé dedans.")

client = OpenAI(base_url="https://albert.api.etalab.gouv.fr/v1", api_key=API_KEY)
MODELE = "gpt-oss-120b"
MODELE_EMBEDDINGS = "bge-m3"

TAILLE_MORCEAU = 800
CHEVAUCHEMENT = 150
NB_MORCEAUX = 4

class EuropePMCRAG:
    def __init__(self, pmcid):
        self.pmcid = pmcid
        self.metadata = {}
        self.full_text = ""
        self.chunks = []
        self.embeddings = None

    def fetch_xml(self):
        print(f"Récupération de l'article {self.pmcid}...")
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{self.pmcid}/fullTextXML"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=120)
            
            if response.status_code == 503:
                raise Exception("Le serveur EuropePMC est temporairement indisponible (Erreur 503).")
            elif response.status_code != 200:
                raise Exception(f"Erreur lors de la récupération : {response.status_code}")
            
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"Une erreur réseau est survenue : {e}")

    def parse_xml(self, xml_content):
        print("Analyse du XML...")
        root = ET.fromstring(xml_content)
        
        title = root.find(".//article-title")
        self.metadata['title'] = title.text if title is not None else "Titre inconnu"
        
        journal = root.find(".//journal-title")
        self.metadata['journal'] = journal.text if journal is not None else "Journal inconnu"
        
        pub_date = root.find(".//pub-date")
        self.metadata['date'] = pub_date.text if pub_date is not None else "Date inconnue"
        
        authors = []
        for contrib in root.findall(".//contrib"):
            name = contrib.find(".//surname")
            if name is not None:
                authors.append(name.text)
        self.metadata['authors'] = ", ".join(authors) if authors else "Auteurs inconnus"

        text_parts = []
        abstract = root.find(".//abstract")
        if abstract is not None:
            text_parts.append("ABSTRACT:\n" + "".join(abstract.itertext()).strip())
        
        body = root.find(".//body")
        if body is not None:
            body_text = "".join(body.itertext()).strip()
            body_text = re.sub(r'\s+', ' ', body_text)
            text_parts.append("BODY:\n" + body_text)
        
        self.full_text = "\n\n".join(text_parts)

    def prepare_rag(self):
        print("Préparation du RAG...")
        text = self.full_text
        for debut in range(0, len(text), TAILLE_MORCEAU - CHEVAUCHEMENT):
            morceau = text[debut:debut + TAILLE_MORCEAU].strip()
            if morceau:
                self.chunks.append(morceau)
        
        if not self.chunks:
            raise Exception("Aucun texte n'a pu être extrait.")
            
        vecteurs = []
        for i in range(0, len(self.chunks), 32):
            reponse = client.embeddings.create(model=MODELE_EMBEDDINGS, input=self.chunks[i:i + 32])
            vecteurs.extend(d.embedding for d in reponse.data)
        
        matrice = np.array(vecteurs)
        self.embeddings = matrice / np.linalg.norm(matrice, axis=1, keepdims=True)

    def query(self, question, system_prompt=None):
        vecteur_question = client.embeddings.create(model=MODELE_EMBEDDINGS, input=[question]).data[0].embedding
        vecteur_question = np.array(vecteur_question) / np.linalg.norm(vecteur_question)
        
        scores = self.embeddings @ vecteur_question
        meilleurs = np.argsort(scores)[::-1][:NB_MORCEAUX]
        
        contexte = ""
        for i in meilleurs:
            contexte += f"Passage {i}:\n{self.chunks[i]}\n\n"
            
        if system_prompt is None:
            system_prompt = f"Tu es un expert scientifique. Tu réponds à partir du contexte fourni pour l'article suivant : {self.metadata['title']}. Si la réponse n'est pas dans le texte, dis-le. Cite les passages."

        reponse = client.chat.completions.create(
            model=MODELE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexte :\n{contexte}\nQuestion : {question}"},
            ],
            temperature=0.1,
        )
        return reponse.choices[0].message.content

    def extract_cohort_data(self):
        """Extrait les informations de cohorte via le RAG et les renvoie sous forme de dictionnaire."""
        print("\nExtraction des données de cohorte via RAG...")
        
        questions = {
            "taille_cohorte": "Quelle est la taille de la cohorte étudiée ? (Réponds uniquement par le nombre ou 'Non mentionné')",
            "sexe_cohorte": "Quel est le sexe de la cohorte ? (Ex: Hommes, Femmes, Mixte. Réponds brièvement)",
            "age_cohorte": "Quelle est la plage d'âge de la cohorte ? (Ex: 18-65 ans. Réponds brièvement)",
            "pays_etude": "Dans quel pays l'étude a-t-elle été réalisée ? (Réponds uniquement par le nom du pays ou 'Non mentionné')",
        }
        
        results = {"pmcid": self.pmcid}
        system_prompt = "Tu es un assistant d'extraction de données. Réponds de manière extrêmement concise, sans phrase complète."
        
        for key, q in questions.items():
            print(f"  Recherche de {key}...")
            res = self.query(q, system_prompt=system_prompt)
            results[key] = res.strip()
            
        return results

    def save_to_csv(self, data):
        """Sauvegarde les données extraites dans un fichier CSV."""
        filename = f"results_{self.pmcid}.csv"
        fieldnames = ["pmcid", "taille_cohorte", "sexe_cohorte", "age_cohorte", "pays_etude"]
        
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(data)
        
        print(f"\nDonnées sauvegardées dans : {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python 07_europepmc_rag.py <PMCID>")
        print("Exemple: python 07_europepmc_rag.py PMC5513205")
        sys.exit(1)
        
    pmcid = sys.argv[1]
    
    try:
        rag = EuropePMCRAG(pmcid)
        xml_data = rag.fetch_xml()
        rag.parse_xml(xml_data)
        
        print("\n--- Métadonnées de l'article ---")
        print(f"Titre   : {rag.metadata['title']}")
        print(f"Journal : {rag.metadata['journal']}")
        print(f"Date    : {rag.metadata['date']}")
        print(f"Auteurs : {rag.metadata['authors']}")
        print("-------------------------------\n")
        
        rag.prepare_rag()
        
        # --- Nouvelle fonctionnalité : Extraction CSV ---
        cohort_data = rag.extract_cohort_data()
        rag.save_to_csv(cohort_data)
        
        print("\n--- Résultats de l'extraction ---")
        for k, v in cohort_data.items():
            print(f"{k}: {v}")
        print("--------------------------------\n")
        
        print("\nL'article est prêt ! Posez vos questions (tapez 'quit' pour sortir).")
        while True:
            question = input("\nQuestion : ")
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            print("Réflexion...")
            reponse = rag.query(question)
            print(f"\nRéponse :\n{reponse}")
            
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()
