from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

texts = [
    "Software Engineer with Python, Java and SQL experience.",
    "Machine Learning Engineer with Python and machine learning experience.",
    "Frontend Developer skilled in React, JavaScript and HTML.",
]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
)

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", embeddings.shape[1])
print("Embedding shape:", embeddings.shape)
print("First embedding (first 5 values):", embeddings[0][:5])