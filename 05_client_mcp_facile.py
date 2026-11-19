"""
Script 05 : un client qui utilise le serveur MCP du script 04 avec Albert.

C'est le même principe que le script 03, avec une différence : les outils ne
sont plus écrits à la main dans le script. Ils sont sur le serveur MCP, et le client :
  1. demande au serveur la liste de ses outils, et la donne au LLM
  2. quand le LLM veut utiliser un outil, transmet la demande au serveur

Le serveur doit déjà tourner : lancez d'abord 04_serveur_mcp.py dans un autre
terminal (ou dans un screen), puis ce script dans celui-ci.

    python 05_client_mcp_facile.py "Que sait-on du virus de la rage ?"

Sous le capot, MCP n'est que du JSON envoyé en HTTP : on envoie le nom d'une
action (par exemple "tools/list" pour lister les outils) et on reçoit la réponse.
C'est ce que fait la fonction appeler_mcp() ci-dessous.

Pour aller plus loin (ressources et prompts du serveur) : 06_client_mcp_avance.py
"""

import json
import os
import sys

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

# L'adresse du serveur MCP (celle qu'il affiche au démarrage)
URL_MCP = "http://localhost:8000/mcp"


def appeler_mcp(action, parametres=None):
    """Envoie une demande au serveur MCP et renvoie sa réponse."""
    try:
        reponse = requests.post(
            URL_MCP,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            # Le format de la demande : (jsonrpc et id sont imposés par le protocole, on ne s'en occupe pas)
            json={"jsonrpc": "2.0", "id": 1, "method": action, "params": parametres or {}},
            timeout=60,
        )
    except requests.ConnectionError:
        # Cas le plus courant : on a oublié de lancer le serveur
        sys.exit(f"Impossible de joindre le serveur MCP sur {URL_MCP}.\n"
                 "Est-il lancé ? (python 04_serveur_mcp.py dans un autre terminal)")

    donnees = reponse.json()
    if "error" in donnees:
        raise RuntimeError(f"Le serveur MCP a refusé '{action}' : {donnees['error']['message']}")
    return donnees["result"]


question = " ".join(sys.argv[1:]) or "Que sait-on du virus de la rage ?"

# 1. On demande au serveur ce qu'il sait faire...
liste = appeler_mcp("tools/list")["tools"]

# ... et on traduit ça au format attendu par l'API (le même qu'au script 03).
# Tout vient du serveur : le nom, la description (la docstring) et les arguments.
outils = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["inputSchema"],
        },
    }
    for t in liste
]
print("Outils trouvés sur le serveur :", [t["name"] for t in liste])

messages = [
    {"role": "system", "content": "Tu es un assistant scientifique. Utilise tes outils pour trouver des articles, "
                                  "ne réponds pas de mémoire. Réponds en français."},
    {"role": "user", "content": question},
]

# 2. Même boucle qu'au script 03
reponse_finale = False
for tour in range(5):
    reponse = client.chat.completions.create(
        model=MODELE, messages=messages, tools=outils, tool_choice="auto"
    )
    message = reponse.choices[0].message

    # Pas de demande d'outil : le modèle a sa réponse finale
    if not message.tool_calls:
        print("\n" + message.content)
        reponse_finale = True
        break

    # Sinon, on garde sa demande dans l'historique...
    messages.append({
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": a.id,
                "type": "function",
                "function": {"name": a.function.name, "arguments": a.function.arguments},
            }
            for a in message.tool_calls
        ],
    })

    # ... puis on exécute chaque outil demandé et on renvoie le résultat au modèle
    for appel in message.tool_calls:
        arguments = json.loads(appel.function.arguments)
        print(f"-> le modèle appelle {appel.function.name}({arguments})")

        # La seule vraie différence avec le script 03 :
        # au lieu d'appeler une fonction Python locale, on demande au serveur MCP.
        resultat = appeler_mcp("tools/call", {"name": appel.function.name, "arguments": arguments})
        # Si l'outil a échoué, le message d'erreur est dans le contenu : on le donne
        # au modèle comme n'importe quel résultat, il pourra se corriger.
        texte_outil = "\n".join(c["text"] for c in resultat["content"] if c["type"] == "text")

        messages.append({"role": "tool", "tool_call_id": appel.id, "content": texte_outil})

if not reponse_finale:
    print("Trop de tours sans réponse finale, on s'arrête.")