CV_IMPROVEMENT_PROMPT = """
You are an AI career assistant helping a candidate improve their resume
for a specific target job.

Use ONLY the information provided about the candidate and target job.

Do not invent:
- skills the candidate does not have
- work experience
- education
- projects
- achievements
- certifications

Analyze the candidate against the target job and provide:

1. Missing Skills
   - Skills required or strongly relevant to the target job that are not
     clearly present in the candidate profile.

2. Weak Bullet Points
   - Identify experience/project statements that could be made clearer,
     more specific, or more impact-oriented.
   - Do not invent achievements or numbers.

3. Improved Bullet Points
   - Rewrite the weak bullet points using only the information already
     provided by the candidate.

4. Rewritten Summary
   - Write a concise professional resume summary targeted to the role.
   - Use only information supported by the candidate profile.

Return ONLY valid JSON in this format:

{
    "missing_skills": [],
    "weak_bullet_points": [],
    "improved_bullet_points": [],
    "rewritten_summary": ""
}
"""


def build_cv_improvement_prompt(
    candidate_profile: dict,
    target_job: dict,
) -> str:
    """Build the prompt used for CV improvement generation."""

    if not isinstance(candidate_profile, dict):
        raise ValueError("Candidate profile must be a dictionary.")

    if not isinstance(target_job, dict):
        raise ValueError("Target job must be a dictionary.")

    return f"""
{CV_IMPROVEMENT_PROMPT}

CANDIDATE PROFILE
-----------------
Name:
{candidate_profile.get("name", "")}

Target Role:
{candidate_profile.get("target_role", "")}

Skills:
{", ".join(candidate_profile.get("skills", []))}

Experience:
{" ".join(candidate_profile.get("experience", []))}

Education:
{" ".join(candidate_profile.get("education", []))}


TARGET JOB
----------
Job Title:
{target_job.get("jobtitle", "")}

Company:
{target_job.get("company", "")}

Job Description:
{target_job.get("job_document", "")}
""".strip()