"""
Script 06 : un client qui utilise le serveur MCP du script 04 avec Albert mode avancé

C'est le même principe que le script 03, avec une différence : les outils ne
sont plus écrits à la main dans le script. Le client :
  1. interroge le serveur MCP (04_serveur_mcp.py) pour connaître ses outils, et les donne au LLM
  2. lit une ressource du serveur (l'aide-mémoire PubMed) pour la donner au LLM
  3. optionnel : utilise un prompt du serveur pour écrire la consigne
  4. quand le LLM veut un outil, il transmet la demande au serveur

Le serveur doit déjà tourner : lancez d'abord 04_serveur_mcp.py dans un autre
terminal (ou dans un screen), puis ce script dans celui-ci.

Ici on parle au serveur avec un simple requests.post. Sous le capot, MCP n'est que du JSON envoyé en HTTP, au format "JSON-RPC" : on envoie une "méthode"
(tools/list, tools/call, resources/read, prompts/get...) avec ses paramètres et on reçoit un résultat. La fonction appeler_mcp() plus bas fait exactement ça.

Deux façons de l'utiliser :

    # une question libre, le modèle se sert des outils s'il en a besoin
    python 06_client_mcp.py "Que sait-on du virus de la rage ?"

    # on laisse le serveur écrire la consigne, avec son prompt "revue_de_litterature"
    python 06_client_mcp.py --revue "efficacité du vaccin contre la grippe"
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


def appeler_mcp(methode, parametres=None):
    """Envoie une requête au serveur MCP et renvoie son résultat."""
    try:
        reponse = requests.post(
            URL_MCP,
            # Le serveur veut qu'on annonce qu'on sait lire du JSON
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            # Le format JSON-RPC : une version, un numéro de requête, une méthode et ses paramètres
            json={"jsonrpc": "2.0", "id": 1, "method": methode, "params": parametres or {}},
            timeout=60,
        )
    except requests.ConnectionError:
        # Cas le plus courant : on a oublié de lancer le serveur
        sys.exit(f"Impossible de joindre le serveur MCP sur {URL_MCP}.\n"
                 "Est-il lancé ? (python 04_serveur_mcp.py dans un autre terminal)")

    donnees = reponse.json()
    if "error" in donnees:
        raise RuntimeError(f"Le serveur MCP a refusé {methode} : {donnees['error']['message']}")
    return donnees["result"]


# On sépare l'option --revue du reste des arguments
arguments_cli = [a for a in sys.argv[1:] if a != "--revue"]
mode_revue = "--revue" in sys.argv[1:]

if mode_revue:
    texte = " ".join(arguments_cli) or "efficacité du vaccin contre la grippe"
else:
    texte = " ".join(arguments_cli) or "Que sait-on du virus de la rage ?"

# 1. TOOLS : on demande au serveur ce qu'il sait faire...
liste = appeler_mcp("tools/list")["tools"]

# ... et on traduit ça au format attendu par l'API (le même qu'au script 03).
# Tout vient du serveur : nom, description (la docstring) et arguments.
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
print("Prompts disponibles           :", [p["name"] for p in appeler_mcp("prompts/list")["prompts"]])

# 2. RESOURCES : on lit l'aide-mémoire, et on le donne au modèle dans sa consigne
# pour qu'il écrive de meilleures requêtes PubMed.
ressource = appeler_mcp("resources/read", {"uri": "doc://syntaxe-pubmed"})
aide_memoire = ressource["contents"][0]["text"]

# 3. PROMPTS : en mode --revue, c'est le serveur qui écrit la question.
# On lui donne les paramètres, il renvoie des messages tout prêts.
if mode_revue:
    prompt = appeler_mcp("prompts/get", {"name": "revue_de_litterature", "arguments": {"sujet": texte}})
    question = prompt["messages"][0]["content"]["text"]
else:
    question = texte

messages = [
    {"role": "system", "content": "Tu es un assistant scientifique. Utilise tes outils pour trouver des articles, "
                                  "ne réponds pas de mémoire. Réponds en français.\n\n" + aide_memoire},
    {"role": "user", "content": question},
]

# 4. Même boucle qu'au script 03 (un peu plus de tours : une revue demande plusieurs recherches)
for tour in range(8):
    reponse = client.chat.completions.create(
        model=MODELE, messages=messages, tools=outils, tool_choice="auto"
    )
    message = reponse.choices[0].message

    if not message.tool_calls:
        print("\n" + message.content)
        break

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

    for appel in message.tool_calls:
        arguments = json.loads(appel.function.arguments)
        print(f"-> le modèle appelle {appel.function.name}({arguments})")

        # La seule vraie différence avec le script 03 :
        # au lieu d'appeler une fonction Python locale, on demande au serveur MCP.
        resultat = appeler_mcp("tools/call", {"name": appel.function.name, "arguments": arguments})
        # Si l'outil a échoué (ToolError, paramètre hors limites...), le message d'erreur est dans le
        # contenu : on le donne au modèle comme n'importe quel résultat, il pourra se corriger.
        texte_outil = "\n".join(c["text"] for c in resultat["content"] if c["type"] == "text")

        messages.append({"role": "tool", "tool_call_id": appel.id, "content": texte_outil})
else:
    print("Trop de tours sans réponse finale, on s'arrête.")