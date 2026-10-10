import json

import faiss
import numpy as np

from src.search.embed import embed_text


INDEX_FILE = "vectorstore/career_guidance.index"
METADATA_FILE = "vectorstore/career_guidance_metadata.json"

question = "What skills should I develop to become a software engineer?"


# Load the career-guidance FAISS index
index = faiss.read_index(INDEX_FILE)

# Load career-guidance metadata
with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)


# Convert the question into an embedding
query_embedding = embed_text(question)

query_vector = np.asarray(
    [query_embedding],
    dtype=np.float32,
)

# Retrieve relevant career guidance
scores, indices = index.search(
    query_vector,
    min(2, index.ntotal),
)


print("\nRetrieved Career Guidance")
print("=" * 80)

for rank, (score, index_id) in enumerate(
    zip(scores[0], indices[0]),
    start=1,
):
    document = metadata[index_id]

    print(f"\nRank: {rank}")
    print(f"Score: {score:.4f}")
    print(f"Source: {document['source']}")
    print(f"Chunk: {document['chunk']}")
    print("\nContent:")
    print(document["content"])
    print("-" * 80)