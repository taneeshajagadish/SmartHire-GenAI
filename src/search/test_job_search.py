from src.search.embed import embed_text
from src.search.job_search import load_faiss_index, search_faiss


candidate_profile = """
Target Role: Software Engineer

Skills:
Python, Java, SQL, Data Structures and Algorithms, Git, MySQL,
JDBC, Java Swing, Machine Learning

Education:
B.E. Computer Science Engineering

Experience:
Software development projects involving Java, Python, databases,
and machine learning.
"""


# Load the FAISS index and job metadata
index, metadata = load_faiss_index()

# Convert the candidate profile into an embedding
query_embedding = embed_text(candidate_profile)

# Search for the top 10 matching jobs
results = search_faiss(
    index,
    query_embedding,
    metadata,
    top_k=10,
)

print("\nTop 10 Matching Jobs")
print("=" * 80)

for rank, job in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Job Title: {job.get('jobtitle', '')}")
    print(f"Company: {job.get('company', '')}")
    print(f"Location: {job.get('joblocation_address', '')}")
    print(f"Similarity Score: {job.get('similarity_score', 0):.4f}")
    print("-" * 80)