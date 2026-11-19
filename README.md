# Scripts de démarrage pour le hackathon IA

Quelques petits scripts pour partir d'une base qui marche. L'idée : on les lance,
on regarde ce qu'ils font, puis on les modifie pour son projet. Tout est
commenté en français.

On utilise l'API Albert (DINUM), qui fonctionne comme celle d'OpenAI.

Tous les fichiers sont dans le dépôt GitLab : [tutoriel/Hackathon_IA](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/tree/master/tutoriel/Hackathon_IA?ref_type=heads).

## Installation

Il faut **Python 3.11 ou plus** (FastMCP nécessite cette version).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Les paquets à installer sont listés dans [requirements.txt](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/requirements.txt?ref_type=heads).

### Renseigner la clé ALBERT

```bash
cp .env.example .env
```

Ouvrez ensuite le fichier `.env` (modèle : [.env.example](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/.env.example?ref_type=heads)) et collez votre clé Albert à la place de `votre_cle_ici`.
La clé se crée sur https://albert.playground.etalab.gouv.fr/keys

À chaque nouveau terminal, il faut réactiver l'environnement, sinon Python ne trouvera pas les paquets :

```bash
cd le_dossier_du_projet
source .venv/bin/activate
```

Quand c'est actif, `(.venv)` ou le nom du dossier apparaît au début de la ligne.
Utilisez alors `python` (et pas `python3`).

## Les scripts

| Script | Ce qu'il fait |
|---|---|
| [`00_lister_les_modeles.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/00_lister_les_modeles.py?ref_type=heads) | Affiche les modèles disponibles sur Albert |
| [`01_question_simple.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/01_question_simple.py?ref_type=heads) | Pose une question, affiche la réponse |
| [`02_rag.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/02_rag.py?ref_type=heads) | Répond à une question à partir de vos fichiers `.txt` (dossier [`documents`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/tree/master/tutoriel/Hackathon_IA/documents?ref_type=heads)) |
| [`03_tool_calling.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/03_tool_calling.py?ref_type=heads) | Le LLM utilise un outil (recherche d'articles sur PubMed) |
| [`04_serveur_mcp.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/04_serveur_mcp.py?ref_type=heads) | Un serveur MCP : 2 outils (PubMed, Europe PMC), 2 ressources, 2 prompts |
| [`05_client_mcp_facile.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/05_client_mcp_facile.py?ref_type=heads) | Un client qui utilise les outils du serveur MCP avec Albert |
| [`06_client_mcp.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/06_client_mcp.py?ref_type=heads) | Pareil, avec en plus une ressource et un prompt du serveur |

Les scripts 00 à 03 se lancent seuls, il est possible de modifier la question dans le script

```bash
python 00_lister_les_modeles.py
python 01_question_simple.py 
python 02_rag.py  
python 03_tool_calling.py
```

## Le serveur MCP (scripts 04, 05 et 06)

Ici il y a deux programmes qui tournent en même temps : le serveur (04), qui reste allumé puisque c'est un serveur,
et un client (05 ou 06) qui vient l'interroger. Il faut donc **deux terminaux** ou bien lancer le serveur dans un screen, avec l'environnement activé dans chacun.

Terminal 1, le serveur (il écoute sur http://localhost:8000/mcp) :

```bash
python 04_serveur_mcp.py
```

Terminal 2, le client :

```bash
python 05_client_mcp_facile.py "Que sait-on du virus de la rage ?"

# version avancée : une ressource en plus, et la consigne écrite par le serveur
python 06_client_mcp.py --revue "efficacité du vaccin contre la grippe"
```

Pour que le serveur continue de tourner quand on ferme le terminal, on peut le lancer dans un `screen` :

```bash
screen -S mcp               # ouvre un screen nommé "mcp"
python 04_serveur_mcp.py    # on lance le serveur dedans
# Ctrl+A puis D             # on "détache" le screen, le serveur continue
screen -r mcp               # pour y revenir (Ctrl+C pour arrêter le serveur)
```

Le serveur peut aussi être branché sur d'autres applications compatibles MCP
(Claude Desktop, VS Code...) en leur donnant son adresse. Il n'a aucune authentification :
il n'écoute que sur la machine elle-même, ne le mettez pas en accès libre sur un réseau.

## Si ça ne marche pas

- **`No module named 'xxx'`** : l'environnement n'est pas activé dans ce terminal (`source .venv/bin/activate`), ou l'installation des paquets n'est pas allée au bout.
- **`Requires-Python >=3.10`** à l'installation : votre Python est trop ancien, voir la partie Installation.
- **`Failed to build cryptography`** (Mac Intel) : une version trop récente de `cryptography` n'a pas de version pré-compilée. Le fichier `requirements.txt` fixe déjà une version qui marche ; si l'erreur revient, lancez `uv pip install "cryptography<49"` puis refaites l'installation.
- **`zsh: unknown sort specifier`** : en copiant des commandes, vous avez collé une ligne de commentaire qui commence par `#`. Ne copiez que les commandes.
- **"Il manque la clé API"** : le fichier `.env` n'existe pas, ou il n'est pas dans le même dossier que les scripts.
- **"Impossible de joindre le serveur MCP"** : le serveur n'est pas lancé. Démarrez `04_serveur_mcp.py` dans un autre terminal.
- **Erreur "model not found"** : la liste des modèles change. Lancez le `00`, puis changez la ligne `MODELE = ...` en haut du script.
- **Erreur 401** : la clé est mauvaise ou expirée.
- Autre erreur : demandez à un encadrant, à vos collègues ou à une IA ;)

## Pour aller plus loin

- RAG : mettez vos propres `.txt` dans `documents/`. Pour des PDF, voir la doc Albert.
- Tool calling : ajoutez une fonction, décrivez-la dans `OUTILS`, ajoutez-la dans `FONCTIONS`.
- MCP : ajoutez une fonction avec `@mcp.tool` dans le `04` (n'oubliez pas de relancer le serveur), les clients la détecteront tout seuls. Le serveur propose aussi des ressources (`@mcp.resource`) et des prompts (`@mcp.prompt`), voir les exemples dedans.

Doc de l'API Albert : https://guides.ia.numerique.gouv.fr/albert-api