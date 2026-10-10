import json
from pathlib import Path

import faiss
import numpy as np


VECTORSTORE_DIR = Path("vectorstore")

FAISS_INDEX_PATH = VECTORSTORE_DIR / "jobs.index"
METADATA_PATH = VECTORSTORE_DIR / "jobs_metadata.json"


def build_faiss_index(
    embeddings: list[list[float]] | np.ndarray,
) -> faiss.Index:
    """
    Build a FAISS index from embedding vectors.

    Embeddings are normalized so that inner product
    corresponds to cosine similarity.
    """

    if embeddings is None:
        raise ValueError("Embeddings cannot be None.")

    vectors = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    if vectors.shape[0] == 0:
        raise ValueError(
            "Embeddings cannot be empty."
        )

    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


def save_faiss_index(
    index: faiss.Index,
    metadata: list[dict],
) -> None:
    """
    Save the FAISS index and its metadata.
    """

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match metadata size."
        )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"FAISS index saved: {FAISS_INDEX_PATH}"
    )

    print(
        f"Metadata saved: {METADATA_PATH}"
    )


def load_faiss_index() -> tuple[faiss.Index, list[dict]]:
    """
    Load the FAISS index and metadata from disk.
    """

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found: {METADATA_PATH}"
        )

    index = faiss.read_index(
        str(FAISS_INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match metadata size."
        )

    return index, metadata


def search_faiss(
    index: faiss.Index,
    query_embedding: list[float],
    metadata: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Search the FAISS index using a query embedding.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(
        [query_embedding],
        dtype=np.float32,
    )

    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match "
            "the FAISS index dimension."
        )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        if idx == -1:
            continue

        result = metadata[idx].copy()

        result["similarity_score"] = float(
            score
        )

        results.append(result)

    return results
def build_candidate_profile_text(profile: dict) -> str:
    """Convert a parsed resume profile into searchable text."""

    if not isinstance(profile, dict):
        raise ValueError("Candidate profile must be a dictionary.")

    required_fields = {
        "name",
        "skills",
        "experience",
        "education",
        "target_role",
    }

    if set(profile.keys()) != required_fields:
        raise ValueError(
            "Candidate profile does not contain the required fields."
        )

    skills = ", ".join(profile["skills"])
    experience = "\n".join(profile["experience"])
    education = "\n".join(profile["education"])

    return f"""
Target Role:
{profile["target_role"]}

Skills:
{skills}

Experience:
{experience}

Education:
{education}
""".strip()