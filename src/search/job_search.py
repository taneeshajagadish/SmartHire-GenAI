"""Hybrid FAISS retrieval and explainable job reranking for SmartHire GenAI."""
from __future__ import annotations

from streamlit import title

from src.parsing.resume_parser import detect_job_level

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np

VECTORSTORE_DIR = Path("vectorstore")
FAISS_INDEX_PATH = VECTORSTORE_DIR / "jobs.index"
METADATA_PATH = VECTORSTORE_DIR / "jobs_metadata.json"

STOP_WORDS = {
    "and", "or", "the", "a", "an", "for", "to", "of", "in", "with",
    "on", "at", "by", "role", "job", "position", "experience",
    "experienced", "required", "skills", "candidate", "professional",
    "looking", "seeking", "responsibilities", "responsibility", "work",
}

CLINICAL_TERMS = {
    "physiotherapist", "physiotherapy", "physical therapist",
    "clinical physiotherapist",
}

CLINICAL_QUALIFICATION_TERMS = {
    "bpt", "mpt", "bachelor of physiotherapy",
    "master of physiotherapy", "physiotherapy degree",
    "registered physiotherapist", "clinical physiotherapy",
}

# Domain-specific aliases. Generic "coach" is deliberately excluded.
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    # Software and IT
    "python": (r"\bpython\b",),
    "java": (r"\bjava\b",),
    "kotlin": (r"\bkotlin\b",),
    "javascript": (r"\bjavascript\b", r"\bjs\b"),
    "typescript": (r"\btypescript\b",),
    "sql": (r"\bsql\b",),
    "mysql": (r"\bmysql\b",),
    "postgresql": (r"\bpostgres(?:ql)?\b",),
    "react": (r"\breact(?:\.js|js)?\b",),
    "node.js": (r"\bnode(?:\.js|js)?\b",),
    "express": (r"\bexpress(?:\.js|js)?\b",),
    "html": (r"\bhtml5?\b",),
    "css": (r"\bcss3?\b",),
    "rest api": (r"\brest(?:ful)?\s+apis?\b",),
    "data structures": (r"\bdata structures?\b", r"\bdsa\b"),
    "git": (r"\bgit\b",),
    "docker": (r"\bdocker\b",),
    "aws": (r"\baws\b", r"\bamazon web services\b"),
    "android": (r"\bandroid\b",),
    "jetpack compose": (r"\bjetpack compose\b",),
    "flutter": (r"\bflutter\b",),
    "postman": (r"\bpostman\b",),
    "selenium": (r"\bselenium\b",),
    "machine learning": (r"\bmachine learning\b",),
    "api integration": (
        r"\bapi integration\b", r"\bintegrating apis\b",
    ),
    "testing": (
        r"\bsoftware testing\b", r"\btest automation\b", r"\btesting\b",
    ),
    "debugging": (r"\bdebugging\b",),
    "cloud computing": (r"\bcloud computing\b",),

    # Sports and fitness
    "strength and conditioning": (
        r"\bstrength training\b",
        r"\bstrength\b",
        r"\bconditioning\b",
        r"\bathletic performance\b",
        r"\bathletic development\b",
        r"\bphysical conditioning\b",
    ),
    
    "strength training": (
        r"\bstrength training\b", r"\bresistance training\b",
        r"\bweight training\b",
    ),
    "movement screening": (
        r"\bmovement screening\b",
        r"\bfunctional movement screening\b",
    ),
    "client communication": (
        r"\bclient communication\b",
        r"\bcommunication with clients\b",
    ),
    "client retention": (r"\bclient retention\b",),
    "athletic development": (
        r"\bathletic development\b", r"\bathlete development\b",
    ),
    "biomechanical assessment": (
        r"\bbiomechanical assessment\b", r"\bbiomechanics\b",
    ),
    "cricket performance": (
        r"\bcricket athletic performance\b", r"\bcricket performance\b",
    ),
    "hypertrophy": (r"\bhypertrophy\b",),
    "endurance training": (r"\bendurance(?: training)?\b",),
    "corrective exercise": (r"\bcorrective exercises?\b",),
    "sport-specific conditioning": (
        r"\bsports?[- ]specific conditioning\b",
    ),
    "personal coaching": (
        r"\b1\s*[-–]?\s*on\s*[-–]?\s*1 coaching\b",
        r"\bone[- ]on[- ]one coaching\b",
        r"\bpersonal coaching\b",
    ),
    "progress tracking": (
        r"\bprogress tracking\b", r"\bperformance tracking\b",
    ),
    "equipment optimization": (
        r"\bequipment optimization\b",
        r"\bequipment optimisation\b",
    ),
    
    "sports coaching": (
        r"\bsports? coaching\b", r"\bphysical education\b",
        r"\bsports? coach\b",
    ),
    "athletic performance": (
        r"\bathletic performance\b", r"\bsports performance\b",
        r"\bsport performance\b",
    ),
    "fitness assessment": (
        r"\bfitness assessment\b", r"\bphysical assessment\b",
        r"\bfitness testing\b",
    ),
    "exercise programming": (
        r"\bexercise programming\b", r"\btraining programming\b",
        r"\bprogram design\b",
    ),
    "injury prevention": (r"\binjury prevention\b",),
    "nutrition": (r"\bnutrition\b",),
    "personal training": (
        r"\bpersonal training\b", r"\bpersonal trainer\b",
    ),
    "athlete monitoring": (
        r"\bathlete monitoring\b", r"\btraining load monitoring\b",
    ),
    "sports science": (r"\bsports science\b",),
    "swimming coaching": (r"\bswimming coach(?:ing)?\b",),
}


def tokenize(text: str) -> set[str]:
    """Normalize text into searchable words."""
    words = re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
    return {
        word for word in words
        if len(word) > 1 and word not in STOP_WORDS
    }


def get_job_text(job: dict) -> str:
    """Combine the job fields used for matching."""
    fields = (
        "jobtitle", "title", "skills", "jobdescription",
        "job_description", "job_document", "content", "company",
        "industry", "education",
    )
    return " ".join(
        str(job.get(field, "") or "") for field in fields
    )

GENDER_RESTRICTION_PATTERNS = (
    r"\bonly\s+(?:female|females|woman|women|male|males|man|men)\s+(?:candidates?\s+)?can\s+apply\b",
    r"\b(?:female|females|male|males)\s+(?:candidates?|applicants?)\s+only\b",
    r"\b(?:women|men)\s+only\b",
    r"\b(?:female|females|male|males)\s+only\b",
    r"\b(?:female|male)\b\s*[-–]\s*(?:only|candidates only)\b",
    r"\b(?:female|male)\b\s+(?:candidates|applicants)\s+only\b",
)

def is_gender_restricted_job(job: dict) -> bool:
    text = get_job_text(job).lower()
    return any(
        re.search(pattern, text)
        for pattern in GENDER_RESTRICTION_PATTERNS
    )



def extract_section(profile_text: str, section: str) -> str:
    """Extract a labelled section until the next labelled section."""
    text = str(profile_text or "")
    pattern = (
        rf"(?ims)^\s*{re.escape(section)}\s*:\s*(.*?)"
        rf"(?=^\s*[A-Za-z][A-Za-z /&_-]*\s*:\s*|\Z)"
    )
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def calculate_overlap(
    candidate_words: set[str],
    job_words: set[str],
) -> float:
    """Measure candidate vocabulary overlap with a job."""
    if not candidate_words:
        return 0.0
    return len(candidate_words & job_words) / len(candidate_words)


def _role_family(text: str) -> str | None:
    """Classify a role into a broad career domain."""
    tokens = tokenize(text)

    if tokens & {
        "physiotherapist", "physiotherapy", "clinical",
        "rehabilitation",
    }:
        return "clinical"

    if tokens & {"agile", "scrum"}:
        return "agile_business"

    if tokens & {
        "sports", "sport", "fitness", "athletic", "athlete",
        "strength", "conditioning", "swimming", "cricket",
        "trainer", "training", "exercise", "gym", "wellness",
    }:
        return "sports_fitness"

    if tokens & {
        "software", "developer", "engineer", "programmer",
        "frontend", "backend", "android", "ios", "web",
        "programming",
    }:
        return "software"

    if tokens & {
        "teacher", "teaching", "school", "education",
        "instructor", "faculty",
    }:
        return "teaching"

    if tokens & {"sales", "retail", "account", "executive"}:
        return "sales"

    return None


def _classify_role_specialty(text: str) -> str:
    """Classify a job by its actual specialty, not isolated shared words."""
    text = re.sub(r"[-_/]+", " ", str(text or "").lower())
    text = " ".join(text.split())

    # IMPORTANT: classify sales before sports.
    # "Sports Supplement Sales Executive" is a sales role.
    if re.search(
        r"\b(sales|retail|business development|"
        r"account executive|supplement sales)\b",
        text,
    ):
        return "sales"

    if re.search(r"\b(agile|scrum)\b", text):
        return "agile"

    if re.search(
        r"\b(physiotherapist|physiotherapy|physical therapist|"
        r"sports therapist|sports therapy|rehabilitation therapist)\b",
        text,
    ):
        return "clinical"

    if re.search(
        r"\bstrength\s*(?:and|&)\s*conditioning\b",
        text,
    ):
        return "strength_conditioning"

    if re.search(r"\bswimming\b", text):
        return "swimming"

    if re.search(
        r"\b(physical education|sports teacher|"
        r"sport teacher|physical education teacher)\b",
        text,
    ):
        return "sports_education"

    if re.search(
        r"\b(sports coach|sport coach|sports coaching|"
        r"sport coaching|athletic development|sports performance)\b",
        text,
    ):
        return "sports_coaching"

    if re.search(
        r"\b(fitness|personal trainer|fitness trainer|"
        r"gym instructor|exercise instructor)\b",
        text,
    ):
        return "fitness"

    if re.search(
        r"\b(software|developer|programmer|programming|"
        r"frontend|backend|android|ios|web development|"
        r"full stack|data engineer)\b",
        text,
    ):
        return "software"

    if re.search(
        r"\b(teacher|teaching|school instructor|faculty)\b",
        text,
    ):
        return "education"

    if re.search(r"\b(trainer|coach|coaching)\b", text):
        return "general_coaching"

    return "other"


def calculate_role_match(
    target_role: str,
    job_title: str,
) -> float:
    """Calculate role fit using specialty-specific compatibility."""
    target = str(target_role or "").strip()
    title = str(job_title or "").strip()

    if not target or not title:
        return 0.0

    target_normalized = " ".join(
        re.sub(r"[-_/]+", " ", target.lower()).split()
    )
    title_normalized = " ".join(
        re.sub(r"[-_/]+", " ", title.lower()).split()
    )

    if target_normalized == title_normalized:
        return 1.0

    target_family = _classify_role_specialty(target)
    job_family = _classify_role_specialty(title)

    # Never award a strong match just for sharing "coach" or "trainer".
    if target_family != job_family:
        compatibility = {
            "strength_conditioning": {
                "sports_coaching": 0.72,
                "sports_education": 0.62,
                "fitness": 0.68,
                "swimming": 0.45,
                "general_coaching": 0.35,
                "clinical": 0.20,
            },
            "sports_coaching": {
                "strength_conditioning": 0.78,
                "sports_education": 0.75,
                "fitness": 0.68,
                "swimming": 0.60,
                "general_coaching": 0.35,
            },
            "sports_education": {
                "strength_conditioning": 0.62,
                "sports_coaching": 0.75,
                "swimming": 0.65,
                "fitness": 0.55,
            },
            "swimming": {
                "sports_coaching": 0.60,
                "sports_education": 0.65,
                "strength_conditioning": 0.45,
            },
            "software": {
                "general_coaching": 0.0,
                "agile": 0.0,
                "sales": 0.0,
            },
        }

        score = compatibility.get(target_family, {}).get(
            job_family, 0.0
        )

        return score

    # Same specialty: compare meaningful title words.
    target_tokens = tokenize(target_normalized)
    title_tokens = tokenize(title_normalized)

    generic_words = {
        "coach", "coaching", "trainer", "training",
        "manager", "lead", "senior", "junior",
        "assistant", "executive", "specialist",
    }

    meaningful_target = target_tokens - generic_words
    meaningful_title = title_tokens - generic_words

    if not meaningful_target:
        return 0.65

    overlap = (
        len(meaningful_target & meaningful_title)
        / len(meaningful_target)
    )

    if overlap >= 1.0:
        return 1.0

    if overlap >= 0.5:
        return 0.80

    return 0.55


def has_clinical_qualification_evidence(
    candidate_text: str,
) -> bool:
    """Check for explicit evidence of clinical qualifications."""
    text = str(candidate_text or "").lower()
    return any(
        term in text for term in CLINICAL_QUALIFICATION_TERMS
    )


def build_faiss_index(
    embeddings: list[list[float]] | np.ndarray,
) -> faiss.Index:
    """Build a normalized FAISS index."""
    if embeddings is None:
        raise ValueError("Embeddings cannot be None.")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if (
        vectors.ndim != 2
        or vectors.shape[0] == 0
        or vectors.shape[1] == 0
    ):
        raise ValueError(
            "Embeddings must be a non-empty 2D matrix."
        )

    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_faiss_index(
    index: faiss.Index,
    metadata: list[dict],
) -> None:
    """Save the FAISS index and its metadata."""
    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match metadata size."
        )

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(
            metadata, file, indent=2, ensure_ascii=False
        )

    print(f"FAISS index saved: {FAISS_INDEX_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


def load_faiss_index() -> tuple[faiss.Index, list[dict]]:
    """Load the index and enrich job experience from the cleaned CSV."""
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"FAISS metadata not found: {METADATA_PATH}"
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match metadata size."
        )

    csv_path = Path("data/jobs/cleaned_jobs.csv")

    if csv_path.exists():
        import pandas as pd

        df = pd.read_csv(csv_path).fillna("")

        if "jobid" in df.columns and "experience" in df.columns:
            experience_by_id = {
                str(row["jobid"]): str(row["experience"]).strip()
                for _, row in df.iterrows()
            }

            for job in metadata:
                exp = experience_by_id.get(
                    str(job.get("jobid", "")), ""
                )
                if exp:
                    job["experience"] = exp

        elif (
            "jobtitle" in df.columns
            and "experience" in df.columns
        ):
            by_title: dict[str, str] = {}

            for _, row in df.iterrows():
                title = str(row["jobtitle"]).strip().lower()
                by_title.setdefault(
                    title, str(row["experience"]).strip()
                )

            for job in metadata:
                title = str(
                    job.get("jobtitle", job.get("title", ""))
                ).strip().lower()

                if by_title.get(title):
                    job["experience"] = by_title[title]

    return index, metadata



    """Estimate level only when explicit experience evidence exists."""
    experience = extract_section(profile_text, "Experience")
    combined = f"{experience}\n{profile_text}".lower()

    if re.search(
        r"\b(intern|internship|fresher|graduate trainee|"
        r"entry.level|new graduate)\b",
        combined,
    ):
        return "entry"

    total = re.search(
        r"\b(\d{1,2})\s*\+?\s*years?\s+(?:of\s+)?experience\b",
        experience.lower(),
    )

    if total:
        years = int(total.group(1))
        if years < 2:
            return "entry"
        if years < 5:
            return "mid"
        return "senior"

    spans = re.findall(
        r"\b(20\d{2})\s*(?:-|–|—|to)\s*"
        r"(20\d{2}|present|current)\b",
        experience.lower(),
    )

    if spans:
        durations = []

        for start, end in spans:
            end_year = (
                2026 if end in {"present", "current"} else int(end)
            )
            durations.append(max(0, end_year - int(start)))

        years = max(durations, default=0)

        if years < 2:
            return "entry"
        if years < 5:
            return "mid"
        return "senior"

    return "unknown"

def detect_candidate_level(profile_text: str) -> str:
    """Estimate experience level from explicit resume evidence."""
    experience = extract_section(profile_text, "Experience")
    combined = f"{experience}\n{profile_text}".lower()

    if re.search(
        r"\b(intern|internship|fresher|graduate trainee|"
        r"entry.level|new graduate)\b",
        combined,
    ):
        return "entry"

    # Explicit total experience, e.g. "3 years of experience".
    total = re.search(
        r"\b(\d{1,2})\s*\+?\s*years?\s+"
        r"(?:of\s+)?experience\b",
        experience,
    )
    if total:
        years = int(total.group(1))
        if years < 2:
            return "entry"
        if years < 5:
            return "mid"
        return "senior"

    # Recognize common resume date formats.
    spans = re.findall(
        r"\b(20\d{2})\s*(?:-|–|—|to)\s*"
        r"(20\d{2}|present|current|now)\b",
        experience,
    )

    if spans:
        durations = []
        for start, end in spans:
            end_year = (
                2026 if end in {"present", "current", "now"}
                else int(end)
            )
            durations.append(max(0, end_year - int(start)))

        years = max(durations, default=0)
        if years < 2:
            return "entry"
        if years < 5:
            return "mid"
        return "senior"

    return "unknown"



def classify_experience_requirement(
    job: dict,
    title: str,
) -> str:
    """Describe a job's requirement without inventing missing data."""
    title_lower = str(title or "").lower()

    if re.search(
        r"\b(lead|principal|staff|director|head of|architect|"
        r"sr\.?|senior|manager|team leader)\b",
        title_lower,
    ):
        return "Senior/lead role"

    if re.search(
        r"\b(intern|internship|trainee|fresher|"
        r"graduate engineer trainee|junior)\b",
        title_lower,
    ):
        return "Fresher-friendly title"

    for field in (
        "experience",
        "min_experience",
        "minimum_experience",
        "required_experience",
    ):
        value = str(job.get(field, "") or "").strip().lower()

        if not value:
            continue

        match = re.search(
            r"\b(\d{1,2})\s*(?:-|–|—|to)\s*"
            r"(\d{1,2})\s*(?:yrs?|years?)\b",
            value,
        )

        if match:
            minimum = int(match.group(1))
            maximum = int(match.group(2))

            if maximum <= 1:
                return (
                    f"Fresher-friendly ({minimum}–{maximum} years)"
                )
            if minimum >= 5:
                return (
                    f"Senior experience required "
                    f"({minimum}–{maximum} years)"
                )
            if minimum >= 2:
                return (
                    f"Prior experience required "
                    f"({minimum}–{maximum} years)"
                )

            return f"Experience range: {minimum}–{maximum} years"

        single = re.search(
            r"\b(\d{1,2})\s*(?:yrs?|years?)\b",
            value,
        )

        if single:
            years = int(single.group(1))

            if years <= 1:
                return f"Fresher-friendly ({years} year)"
            if years < 5:
                return f"Prior experience required ({years} years)"

            return f"Senior experience required ({years} years)"

    return "Not specified in job data"


def calculate_experience_match(
    candidate_level: str,
    requirement: str,
    job_level: str,
) -> float:
    """Estimate seniority fit; unknown candidate experience stays neutral."""
    candidate_level = str(
        candidate_level or "unknown"
    ).lower()
    job_level = str(job_level or "unknown").lower()
    requirement = str(requirement or "").lower()

    if candidate_level == "unknown":
        return 0.50

    if "senior/lead role" in requirement or job_level == "senior":
        return {
            "entry": 0.10,
            "mid": 0.45,
            "senior": 0.90,
        }.get(candidate_level, 0.50)

    match = re.search(
        r"(\d{1,2})\s*(?:-|–|—|to)\s*"
        r"(\d{1,2})\s*years?",
        requirement,
    )

    estimated_years = {
        "entry": 0,
        "mid": 2,
        "senior": 5,
    }.get(candidate_level)

    if estimated_years is None:
        return 0.50

    if match:
        minimum = int(match.group(1))
        maximum = int(match.group(2))

        if minimum <= estimated_years <= maximum:
            return 1.0

        if estimated_years < minimum:
            return max(
                0.10,
                1.0 - 0.20 * (minimum - estimated_years),
            )

        return max(
            0.50,
            1.0 - 0.10 * (estimated_years - maximum),
        )

    fallback_scores = {
        "entry": 0.75 if job_level == "entry" else 0.55,
        "mid": 0.75,
        "senior": 0.65,
    }

    return fallback_scores.get(candidate_level, 0.50)


def extract_canonical_skills(text: str) -> set[str]:
    """Return recognized, normalized skills from text."""
    text = str(text or "").lower()

    return {
        skill
        for skill, patterns in SKILL_ALIASES.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def get_matched_skills(
    candidate_skills_text: str,
    job_text: str,
) -> list[str]:
    """Find exact and carefully related sports/fitness skill evidence."""
    candidate_skills = extract_canonical_skills(
        candidate_skills_text
    )
    job_skills = extract_canonical_skills(job_text)
    job_lower = str(job_text or "").lower()

    matched = candidate_skills & job_skills

    # Related-skill inference is restricted to relevant sports roles.
    family = _classify_role_specialty(job_text)

    sports_families = {
        "strength_conditioning",
        "sports_coaching",
        "sports_education",
        "swimming",
        "fitness",
    }

    related_evidence = {
        "strength and conditioning": (
            r"\bstrength training\b",
            r"\bconditioning\b",
            r"\bathletic performance\b",
            r"\bathletic development\b",
        ),
        "movement screening": (
            r"\bmovement assessment\b",
            r"\bfunctional movement\b",
            r"\bbiomechanical assessment\b",
        ),
        "athletic development": (
            r"\bathlete development\b",
            r"\bathletic performance\b",
            r"\bsports performance\b",
        ),
        "biomechanical assessment": (
            r"\bbiomechanics\b",
            r"\bmovement assessment\b",
        ),
        "cricket performance": (
            r"\bcricket\b",
        ),
        "hypertrophy": (
            r"\bhypertrophy\b",
            r"\bbodybuilding\b",
            r"\bresistance training\b",
        ),
        "endurance training": (
            r"\bendurance\b",
            r"\bstamina\b",
            r"\bendurance training\b",
        ),
        "corrective exercise": (
            r"\bcorrective exercise\b",
            r"\bcorrective movement\b",
        ),
        "sport-specific conditioning": (
            r"\bsport specific conditioning\b",
            r"\bsports specific conditioning\b",
            r"\bathletic conditioning\b",
        ),
        "personal coaching": (
            r"\bpersonal training\b",
            r"\bpersonal trainer\b",
        ),
        "progress tracking": (
            r"\bperformance tracking\b",
            r"\bathlete monitoring\b",
            r"\btraining load monitoring\b",
        ),
        "equipment optimization": (
            r"\bequipment optimization\b",
            r"\bequipment optimisation\b",
            r"\bgym equipment\b",
        ),
    }

    if family in sports_families:
        for skill, patterns in related_evidence.items():
            if skill in candidate_skills and any(
                re.search(pattern, job_lower)
                for pattern in patterns
            ):
                matched.add(skill)

    return sorted(matched)


def calculate_skill_match(
    candidate_skills_text: str,
    job_text: str,
) -> float:
    """Calculate the share of candidate skills supported by job evidence."""
    candidate_skills = extract_canonical_skills(
        candidate_skills_text
    )

    if not candidate_skills:
        return 0.0

    matched_skills = get_matched_skills(
        candidate_skills_text,
        job_text,
    )

    return len(matched_skills) / len(candidate_skills)
def build_candidate_profile_text(profile) -> str:
    """Convert the extracted candidate profile into searchable text."""

    if isinstance(profile, str):
        return profile.strip()

    if not isinstance(profile, dict):
        return str(profile or "").strip()

    lines = [
        f"Candidate: {profile.get('name', '')}",
        f"Target Role: {profile.get('target_role', '')}",
    ]

    skills = profile.get("skills", [])
    if isinstance(skills, list):
        skills = ", ".join(str(skill) for skill in skills)
    lines.append(f"Skills: {skills}")

    experience = profile.get("experience", [])
    if isinstance(experience, list):
        experience = "\n".join(str(item) for item in experience)
    lines.append(f"Experience: {experience}")

    education = profile.get("education", [])
    if isinstance(education, list):
        education = "\n".join(str(item) for item in education)
    lines.append(f"Education: {education}")

    return "\n".join(
        line for line in lines
        if line.split(":", 1)[-1].strip()
    )

def search_faiss(
    index,
    candidate_embedding,
    metadata,
    top_k=10,
    candidate_profile_text="",
):
    """Retrieve jobs with FAISS and rerank them using explainable signals."""

    if index is None or not metadata or not candidate_profile_text:
        return []

    query = np.asarray(candidate_embedding, dtype=np.float32).reshape(1, -1)

    if query.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: query={query.shape[1]}, "
            f"index={index.d}"
        )

    faiss.normalize_L2(query)

    candidate_text = str(candidate_profile_text)
    target_match = re.search(
        r"(?im)^\s*Target Role\s*:\s*(.+)$", candidate_text
    )
    target_role = target_match.group(1).strip() if target_match else ""

    skills_text = extract_section(candidate_text, "Skills")
    candidate_level = detect_candidate_level(candidate_text)

    search_count = min(index.ntotal, max(top_k * 50, 500))
    similarities, indices = index.search(query, search_count)

    results = []

    for similarity, idx in zip(similarities[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        job = metadata[int(idx)]

        if is_gender_restricted_job(job):
            continue

        title = str(job.get("jobtitle", job.get("title", "")) or "")
        job_text = get_job_text(job)

        role_score = calculate_role_match(target_role, title)
        matched_skills = get_matched_skills(skills_text, job_text)

        candidate_skills = extract_canonical_skills(skills_text)
        skill_score = (
            len(matched_skills) / len(candidate_skills)
            if candidate_skills else 0.0
        )

        
        job_level = detect_job_level(f"{title} {job}")

        requirement = classify_experience_requirement(job, title)
        experience_score = calculate_experience_match(
            candidate_level, requirement, job_level
        )

        similarity_score = float(similarity)
        ranking_score = (
            0.30 * max(0.0, similarity_score)
            + 0.35 * role_score
            + 0.25 * skill_score
            + 0.10 * experience_score
        )

        target_family = _classify_role_specialty(target_role)
        job_family = _classify_role_specialty(title)

        if (
            target_family not in {"other", "general_coaching"}
            and job_family not in {"other", "general_coaching"}
            and role_score == 0.0
        ):
            ranking_score *= 0.5

        if (
            target_family == "clinical"
            and not has_clinical_qualification_evidence(candidate_text)
        ):
            ranking_score *= 0.5

        results.append({
            **job,
            "similarity_score": similarity_score,
            "ranking_score": float(ranking_score),
            "role_match_score": float(role_score),
            "skill_match_score": float(skill_score),
            "skill_overlap_score": float(skill_score),
            "experience_match_score": float(experience_score),
            "candidate_experience_level": candidate_level,
            "job_experience_level": job_level,
            "experience_requirement": requirement,
            "experience_requirement_available": bool(
                str(job.get("experience", "") or "").strip()
            ),
            "matched_skills": matched_skills,
            "profile_overlap_score": float(skill_score),
        })

    results.sort(key=lambda item: item["ranking_score"], reverse=True)
    return results[:top_k]
