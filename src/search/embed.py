
from pathlib import Path

import numpy as np


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DEFAULT_BATCH_SIZE = 32

CACHE_DIR = Path("vectorstore")
CACHE_FILE = CACHE_DIR / "job_embeddings.npy"

# Load the embedding model only when it is actually needed.
model = None


def get_model():
    """Load the embedding model on demand."""
    global model

    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate normalized embeddings for multiple texts."""
    if not texts:
        raise ValueError("Text list cannot be empty.")

    cleaned_texts = []

    for text in texts:
        if not isinstance(text, str):
            raise ValueError("Each text must be a string.")

        text = text.strip()

        if not text:
            raise ValueError("Text cannot be empty.")

        cleaned_texts.append(text)

    embeddings = get_model().encode(
        cleaned_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32).tolist()


def embed_text(text: str) -> list[float]:
    """Generate an embedding for a single text."""
    return embed_texts([text])[0]


def embed_in_batches(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Generate embeddings in smaller local batches."""
    if not texts:
        raise ValueError("Text list cannot be empty.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    all_embeddings = []
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        print(f"Embedding jobs {start + 1}-{end} of {total}...")

        all_embeddings.extend(embed_texts(texts[start:end]))

    return all_embeddings


def save_embeddings(
    embeddings: list[list[float]],
    path: Path = CACHE_FILE,
) -> None:
    """Save embeddings to a NumPy file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(embeddings, dtype=np.float32)

    np.save(path, array)

    print(f"Saved {len(array)} embeddings to {path}")


def load_embeddings(
    path: Path = CACHE_FILE,
) -> np.ndarray:
    """Load cached embeddings."""
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {path}"
        )

    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError("Cached embeddings must be a 2D matrix.")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EMBEDDING_DIMENSION}, "
            f"but found {embeddings.shape[1]}."
        )

    print(f"Loaded {len(embeddings)} embeddings from {path}")

    return embeddings
