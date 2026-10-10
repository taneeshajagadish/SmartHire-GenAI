import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

texts = [
    "Software Engineer with Python, Java and SQL experience.",
    "Machine Learning Engineer with experience in Python and machine learning.",
    "Frontend Developer skilled in React, JavaScript and HTML.",
]

batch = client.batches.create_embeddings(
    model="gemini-embedding-001",
    src={
        "inlined_requests": {
            "contents": texts
        }
    },
    config={
        "display_name": "SmartHire embedding test"
    },
)

print("Batch created successfully!")
print("Batch name:", batch.name)
print("Batch state:", batch.state)