from src.parsing.loader import load_resume
from src.parsing.resume_parser import parse_resume
from src.search.embed import embed_text
from src.search.job_search import (
    build_candidate_profile_text,
    load_faiss_index,
    search_faiss,
)
from src.generate.cv_suggestions import generate_cv_suggestions


RESUME_FILE = "data/resumes/sample_resume.docx"


# 1. Load and parse the resume
resume_text = load_resume(RESUME_FILE)
candidate_profile = parse_resume(resume_text)


# 2. Convert candidate profile into searchable text
candidate_text = build_candidate_profile_text(candidate_profile)


# 3. Find the best matching job
query_embedding = embed_text(candidate_text)

index, metadata = load_faiss_index()

matches = search_faiss(
    index,
    query_embedding,
    metadata,
    top_k=1,
)

if not matches:
    raise ValueError("No matching job was found.")


target_job = matches[0]


# 4. Generate CV improvement suggestions
suggestions = generate_cv_suggestions(
    candidate_profile,
    target_job,
)


print("\nTarget Job")
print("=" * 80)
print(f"Job Title: {target_job.get('jobtitle', '')}")
print(f"Company: {target_job.get('company', '')}")
print(
    f"Similarity Score: "
    f"{target_job.get('similarity_score', 0):.4f}"
)


print("\nCV Improvement Suggestions")
print("=" * 80)

print("\nMissing Skills:")
for skill in suggestions["missing_skills"]:
    print(f"- {skill}")

print("\nWeak Bullet Points:")
for bullet in suggestions["weak_bullet_points"]:
    print(f"- {bullet}")

print("\nImproved Bullet Points:")
for bullet in suggestions["improved_bullet_points"]:
    print(f"- {bullet}")

print("\nRewritten Summary:")
print(suggestions["rewritten_summary"])