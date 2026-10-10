from src.parsing.loader import load_resume
from src.parsing.resume_parser import parse_resume
from src.search.embed import embed_text
from src.search.job_search import (
    build_candidate_profile_text,
    load_faiss_index,
    search_faiss,
)


RESUME_FILE = "data/resumes/sample_resume.docx"


# 1. Load resume
resume_text = load_resume(RESUME_FILE)

# 2. Parse resume into structured profile
profile = parse_resume(resume_text)

print("\nParsed Resume Profile")
print("=" * 80)
print(profile)


# 3. Convert profile into searchable text
candidate_text = build_candidate_profile_text(profile)

print("\nCandidate Search Text")
print("=" * 80)
print(candidate_text)


# 4. Embed candidate profile
query_embedding = embed_text(candidate_text)


# 5. Load FAISS index
index, metadata = load_faiss_index()


# 6. Search for matching jobs
results = search_faiss(
    index,
    query_embedding,
    metadata,
    top_k=10,
)


# 7. Display ranked jobs
print("\nTop 10 Matching Jobs")
print("=" * 80)

for rank, job in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Job Title: {job.get('jobtitle', '')}")
    print(f"Company: {job.get('company', '')}")
    print(f"Location: {job.get('joblocation_address', '')}")
    print(f"Similarity Score: {job.get('similarity_score', 0):.4f}")
    print("-" * 80)