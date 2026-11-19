# Starter scripts for the AI hackathon

A few small scripts to start from a working base. The idea: run them, look at what they do,
then modify them for your own project. Everything is commented (in French).

We use the Albert API (DINUM), which works like the OpenAI one.

All the files are in the GitLab repository: [tutoriel/Hackathon_IA](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/tree/master/tutoriel/Hackathon_IA?ref_type=heads).

## Installation

You need **Python 3.11 or later** (FastMCP requires this version).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The packages to install are listed in [requirements.txt](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/requirements.txt?ref_type=heads).

### Setting the ALBERT key

```bash
cp .env.example .env
```

Then open the `.env` file (template: [.env.example](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/.env.example?ref_type=heads)) and paste your Albert key in place of `votre_cle_ici`.
A key can be created at https://albert.playground.etalab.gouv.fr/keys

In every new terminal, you need to reactivate the environment, otherwise Python will not find the packages:

```bash
cd the_project_folder
source .venv/bin/activate
```

When it is active, `(.venv)` or the folder name appears at the start of the line.
Then use `python` (and not `python3`).

## The scripts

| Script | What it does |
|---|---|
| [`00_lister_les_modeles.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/00_lister_les_modeles.py?ref_type=heads) | Lists the models available on Albert |
| [`01_question_simple.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/01_question_simple.py?ref_type=heads) | Asks a question and prints the answer |
| [`02_rag.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/02_rag.py?ref_type=heads) | Answers a question from your `.txt` files (folder [`documents`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/tree/master/tutoriel/Hackathon_IA/documents?ref_type=heads)) |
| [`03_tool_calling.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/03_tool_calling.py?ref_type=heads) | The LLM uses a tool (search for articles on PubMed) |
| [`04_serveur_mcp.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/04_serveur_mcp.py?ref_type=heads) | An MCP server: 2 tools (PubMed, Europe PMC), 2 resources, 2 prompts |
| [`05_client_mcp_facile.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/05_client_mcp_facile.py?ref_type=heads) | A client that uses the MCP server's tools with Albert |
| [`06_client_mcp.py`](https://gitlab.in2p3.fr/ifb-biosphere/mudis4ls-wp4/-/blob/master/tutoriel/Hackathon_IA/06_client_mcp.py?ref_type=heads) | Same, with a resource and a prompt from the server in addition |

Scripts 00 to 03 run on their own; you can change the question inside the script.

```bash
python 00_lister_les_modeles.py
python 01_question_simple.py 
python 02_rag.py  
python 03_tool_calling.py
```

## The MCP server (scripts 04, 05 and 06)

Here two programs run at the same time: the server (04), which stays on since it is a server,
and a client (05 or 06) that queries it. You therefore need **two terminals**, or you can run the server in a screen, with the environment activated in each.

Terminal 1, the server (it listens on http://localhost:8000/mcp):

```bash
python 04_serveur_mcp.py
```

Terminal 2, the client:

```bash
python 05_client_mcp_facile.py "What do we know about the rabies virus?"

# advanced version: one more resource, and the prompt is written by the server
python 06_client_mcp.py --revue "influenza vaccine efficacy"
```

To keep the server running when you close the terminal, you can start it in a `screen`:

```bash
screen -S mcp               # opens a screen named "mcp"
python 04_serveur_mcp.py    # start the server inside it
# Ctrl+A then D             # "detach" the screen, the server keeps running
screen -r mcp               # to go back to it (Ctrl+C to stop the server)
```

The server can also be plugged into other MCP-compatible applications
(Claude Desktop, VS Code...) by giving them its address. It has no authentication:
it only listens on the machine itself, so do not expose it openly on a network.

## If it does not work

- **`No module named 'xxx'`**: the environment is not activated in this terminal (`source .venv/bin/activate`), or the package installation did not complete.
- **`Requires-Python >=3.10`** during installation: your Python is too old, see the Installation section.
- **`Failed to build cryptography`** (Intel Mac): a version of `cryptography` that is too recent has no pre-built version. The `requirements.txt` file already pins a working version; if the error comes back, run `uv pip install "cryptography<49"` and redo the installation.
- **`zsh: unknown sort specifier`**: when copying commands, you pasted a comment line starting with `#`. Copy only the commands.
- **"Il manque la clé API"** (API key missing): the `.env` file does not exist, or it is not in the same folder as the scripts.
- **"Impossible de joindre le serveur MCP"** (cannot reach the MCP server): the server is not running. Start `04_serveur_mcp.py` in another terminal.
- **"model not found" error**: the list of models changes. Run `00`, then change the `MODELE = ...` line at the top of the script.
- **401 error**: the key is wrong or has expired.
- Any other error: ask a mentor, your colleagues, or an AI ;)

## Going further

- RAG: put your own `.txt` files in `documents/`. For PDFs, see the Albert documentation.
- Tool calling: add a function, describe it in `OUTILS`, and add it to `FONCTIONS`.
- MCP: add a function with `@mcp.tool` in `04` (remember to restart the server); the clients will detect it on their own. The server also offers resources (`@mcp.resource`) and prompts (`@mcp.prompt`), see the examples inside.

Albert API documentation: https://guides.ia.numerique.gouv.fr/albert-api