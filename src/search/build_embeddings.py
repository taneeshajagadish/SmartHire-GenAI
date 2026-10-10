import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.search.embed import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    embed_texts,
)


JOBS_FILE = Path("data/jobs/cleaned_jobs.csv")
CACHE_DIR = Path("vectorstore")

EMBEDDINGS_FILE = CACHE_DIR / "job_embeddings.npy"
METADATA_FILE = CACHE_DIR / "job_embedding_metadata.json"

BATCH_SIZE = 32


def load_jobs() -> pd.DataFrame:
    """
    Load the cleaned job dataset.
    """

    if not JOBS_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned jobs file not found: {JOBS_FILE}"
        )

    df = pd.read_csv(JOBS_FILE).fillna("")

    if "job_document" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'job_document' column."
        )

    return df


def load_existing_progress():
    """
    Load previously generated embeddings and metadata.
    """

    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        return [], []

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if embeddings.ndim != 2:
        raise ValueError("Cached embeddings must be a 2D matrix.")

    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Cached embeddings and metadata are out of sync."
        )

    return embeddings.tolist(), metadata


def save_progress(
    embeddings: list[list[float]],
    metadata: list[dict],
) -> None:
    """
    Save all successful progress to disk.
    """

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        EMBEDDINGS_FILE,
        np.asarray(embeddings, dtype=np.float32),
    )

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    """
    Generate embeddings in batches and resume automatically
    from the last successfully saved batch.
    """

    df = load_jobs()

    total_jobs = len(df)

    embeddings, metadata = load_existing_progress()

    completed = len(embeddings)

    print(f"Total jobs: {total_jobs}")
    print(f"Already embedded: {completed}")
    print(f"Remaining: {total_jobs - completed}")

    if completed > total_jobs:
        raise ValueError(
            "Cached embeddings contain more records than the dataset."
        )

    if completed == total_jobs:
        print("\nAll jobs are already embedded.")
        return

    for start in range(
        completed,
        total_jobs,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            total_jobs,
        )

        batch_df = df.iloc[start:end]

        print(
            f"\nEmbedding jobs {start + 1}-{end} "
            f"of {total_jobs}..."
        )

        texts = batch_df["job_document"].tolist()

        try:
            batch_embeddings = embed_texts(texts)

        except Exception as exc:
            print("\nEmbedding request failed.")
            print(f"Reason: {exc}")
            print(
                f"Successfully cached: "
                f"{len(embeddings)}/{total_jobs}"
            )
            print(
                "\nRun this script again after the quota "
                "becomes available. It will resume automatically."
            )
            raise

        batch_metadata = (
            batch_df[
                [
                    "jobid",
                    "jobtitle",
                    "company",
                    "joblocation_address",
                    "job_document",
                ]
            ]
            .to_dict("records")
        )

        embeddings.extend(batch_embeddings)
        metadata.extend(batch_metadata)

        save_progress(
            embeddings,
            metadata,
        )

        print(
            f"Progress saved: "
            f"{len(embeddings)}/{total_jobs}"
        )

    print("\nEmbedding generation completed successfully.")

    print(
        f"Embeddings: {EMBEDDINGS_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()