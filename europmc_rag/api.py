
import re
import threading
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from europepmc_rag import EuropePMCRAG


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Ensure the static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Europe PMC RAG API",
    description="API for querying scientific articles using Europe PMC and RAG.",
    version="1.0.0",
)

# Serve frontend assets
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


# ============================================================
# Cache
# ============================================================

rag_cache: Dict[str, EuropePMCRAG] = {}
cache_lock = threading.Lock()


# ============================================================
# Request and response models
# ============================================================

class QueryRequest(BaseModel):
    pmcid: str
    question: str = Field(min_length=1)


class CohortRequest(BaseModel):
    pmcid: str
    save_csv: bool = False


class QueryResponse(BaseModel):
    pmcid: str
    question: str
    answer: str


# ============================================================
# Helper functions
# ============================================================

def validate_pmcid(pmcid: str) -> str:
    """
    Validate and normalize a Europe PMC identifier.
    Expected format: PMC1234567
    """
    pmcid = pmcid.strip().upper()

    if not re.fullmatch(r"PMC\d+", pmcid):
        raise HTTPException(
            status_code=400,
            detail="Invalid PMCID. Expected format: PMC1234567",
        )

    return pmcid


def get_or_load_rag(pmcid: str) -> EuropePMCRAG:
    """
    Return a cached RAG instance or initialize one.

    Initialization includes:
    1. Downloading the article XML.
    2. Parsing article metadata and full text.
    3. Splitting the text into chunks.
    4. Generating embeddings.
    """

    pmcid = validate_pmcid(pmcid)

    # Check cache first
    if pmcid in rag_cache:
        return rag_cache[pmcid]

    # Prevent multiple simultaneous initializations
    with cache_lock:

        if pmcid in rag_cache:
            return rag_cache[pmcid]

        try:
            rag = EuropePMCRAG(pmcid)

            xml_content = rag.fetch_xml()

            if not xml_content:
                raise ValueError("No XML content returned by Europe PMC.")

            rag.parse_xml(xml_content)

            if not rag.full_text:
                raise ValueError("No full text could be extracted from the article.")

            rag.prepare_rag()

            rag_cache[pmcid] = rag

            return rag

        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Could not load article {pmcid}: {str(e)}",
            )


# ============================================================
# Frontend
# ============================================================

@app.get("/", include_in_schema=False)
def home():
    """
    Serve the chatbot frontend.
    """

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Create static/index.html.",
        )

    return FileResponse(index_file)


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "loaded_articles": len(rag_cache),
        "loaded_pmcids": list(rag_cache.keys()),
    }


# ============================================================
# Load article
# ============================================================

@app.post("/articles/{pmcid}/load")
def load_article(pmcid: str):
    """
    Load an article and prepare its RAG index.
    """

    pmcid = validate_pmcid(pmcid)

    rag = get_or_load_rag(pmcid)

    return {
        "status": "success",
        "pmcid": pmcid,
        "metadata": rag.metadata,
        "chunks": len(rag.chunks),
        "embedding_dimension": (
            int(rag.embeddings.shape[1])
            if rag.embeddings is not None and rag.embeddings.ndim == 2
            else None
        ),
    }


# ============================================================
# Get article metadata
# ============================================================

@app.get("/articles/{pmcid}")
def get_article(pmcid: str):
    """
    Return metadata for an article.

    The article must already be loaded.
    """

    pmcid = validate_pmcid(pmcid)

    if pmcid not in rag_cache:
        raise HTTPException(
            status_code=404,
            detail="Article not loaded. Call POST /articles/{pmcid}/load first.",
        )

    rag = rag_cache[pmcid]

    return {
        "pmcid": pmcid,
        "metadata": rag.metadata,
        "chunks": len(rag.chunks),
    }


# ============================================================
# Ask a question about an article
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query_article(request: QueryRequest):
    """
    Ask a question about a loaded scientific article.
    """

    pmcid = validate_pmcid(request.pmcid)
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    rag = get_or_load_rag(pmcid)

    try:
        answer = rag.query(question)

        return QueryResponse(
            pmcid=pmcid,
            question=question,
            answer=str(answer),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while querying the article: {str(e)}",
        )


# ============================================================
# Extract cohort information
# ============================================================

@app.post("/cohort")
def extract_cohort(request: CohortRequest):
    """
    Extract cohort information from a scientific article.

    Expected information:
    - Cohort size
    - Sex distribution
    - Age range
    - Country

    Optionally saves the result to CSV.
    """

    pmcid = validate_pmcid(request.pmcid)

    rag = get_or_load_rag(pmcid)

    try:
        cohort_data = rag.extract_cohort_data()

        csv_file = None

        if request.save_csv:
            rag.save_to_csv(cohort_data)
            csv_file = f"results_{pmcid}.csv"

        return {
            "status": "success",
            "pmcid": pmcid,
            "cohort_data": cohort_data,
            "csv_file": csv_file,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting cohort information: {str(e)}",
        )
