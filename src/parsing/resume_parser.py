
import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME ="gemini-3.1-flash-lite"

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# RESPONSE SCHEMA
# --------------------------------------------------

RESUME_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "name": {"type": "STRING"},
        "skills": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "experience": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "education": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "target_role": {"type": "STRING"},
    },
    "required": [
        "name",
        "skills",
        "experience",
        "education",
        "target_role",
    ],
}


# --------------------------------------------------
# JOB LEVEL DETECTION
# --------------------------------------------------

def detect_job_level(resume_text):
    """Estimate a candidate's experience level."""
    text = (resume_text or "").lower()

    if any(term in text for term in [
        "senior engineer",
        "senior developer",
        "principal engineer",
        "technical lead",
        "team lead",
    ]):
        return "Senior"

    year_matches = re.findall(
        r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience",
        text,
    )

    if year_matches:
        years = max(int(value) for value in year_matches)

        if years >= 7:
            return "Senior"
        if years >= 3:
            return "Mid-Level"
        return "Entry-Level"

    if any(term in text for term in [
        "internship",
        "intern",
        "fresher",
        "entry-level",
        "entry level",
        "recent graduate",
    ]):
        return "Entry-Level"

    return "Entry-Level"


# --------------------------------------------------
# CLEAN GEMINI RESPONSE
# --------------------------------------------------

def _clean_response(response_text):
    """Extract a JSON object from Gemini's response."""
    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    text = response_text.strip()

    # Remove Markdown code fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    # Extract the JSON object if surrounding text exists.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "Gemini did not return a JSON object. Please retry."
        )

    return text[start:end + 1]


# --------------------------------------------------
# VALIDATE PROFILE
# --------------------------------------------------

def _validate_profile(data):
    """Ensure the extracted profile has consistent fields."""
    if not isinstance(data, dict):
        raise ValueError("Gemini returned an invalid profile.")

    profile = {
        "name": data.get("name") or "Unknown",
        "skills": data.get("skills") or [],
        "experience": data.get("experience") or [],
        "education": data.get("education") or [],
        "target_role": data.get("target_role") or "",
    }

    for field in ("name", "target_role"):
        if not isinstance(profile[field], str):
            profile[field] = str(profile[field])

    for field in ("skills", "experience", "education"):
        value = profile[field]

        if isinstance(value, str):
            profile[field] = [value] if value.strip() else []
        elif isinstance(value, list):
            profile[field] = [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]
        else:
            profile[field] = []

    return profile


# --------------------------------------------------
# PARSE RESUME
# --------------------------------------------------

def parse_resume(resume_text):
    """Use Gemini to extract structured information from a resume."""
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError(
            "Resume text is empty. Please upload a readable resume."
        )

    prompt = f"""
You are an expert resume parser for a job-matching application.

Extract the candidate's information ONLY from the resume below.
Never invent skills, qualifications, experience, or education.

Return:
- name: candidate's full name, or Unknown if unavailable
- skills: list of technical and professional skills
- experience: list of work experience, internships, and projects
- education: list of qualifications and institutions
- target_role: most suitable job role based on the resume

Use empty lists for unavailable list fields.
Return only JSON matching the supplied schema.

RESUME:
{resume_text}
"""

    # Retry only transient service errors, with short backoff.
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=2000,
                    response_mime_type="application/json",
                    response_schema=RESUME_SCHEMA,
                ),
            )

            response_text = response.text

            if not response_text:
                raise ValueError(
                    "Gemini returned an empty response. Please retry."
                )

            cleaned_response = _clean_response(response_text)

            try:
                parsed_data = json.loads(cleaned_response)
            except json.JSONDecodeError as exc:
                print(
                    "GEMINI RAW RESPONSE:",
                    repr(response_text),
                    flush=True,
                )
                print(
                    "JSON ERROR:",
                    str(exc),
                    flush=True,
                )
                raise ValueError(
                    f"Gemini returned malformed JSON: {exc}"
                ) from exc

            return _validate_profile(parsed_data)

        except ValueError:
            raise

        except Exception as exc:
            error_text = str(exc)
            error_lower = error_text.lower()

            # Diagnostic information; never print the API key.
            print(
                f"GEMINI DEBUG: {type(exc).__name__}: {error_text}",
                flush=True,
            )

            is_transient = any(term in error_lower for term in [
                "503",
                "unavailable",
                "high demand",
                "429",
                "resource_exhausted",
                "internal server error",
                "500",
                "temporarily",
            ])

            if is_transient and attempt < max_attempts - 1:
                time.sleep(2)
                continue

            if "429" in error_text or "resource_exhausted" in error_lower:
                raise RuntimeError(
                    "Gemini rate limit or quota reached. "
                    f"Actual API error: {error_text}"
                ) from exc

            if (
                "503" in error_text
                or "unavailable" in error_lower
                or "high demand" in error_lower
            ):
                raise RuntimeError(
                    "Gemini service unavailable after retry. "
                    f"Actual API error: {error_text}"
                ) from exc

            if (
                "api key" in error_lower
                or "401" in error_text
                or "403" in error_text
            ):
                raise RuntimeError(
                    "Gemini authentication failed. Check GEMINI_API_KEY "
                    "in your .env file. Details: " + error_text
                ) from exc

            if "timeout" in error_lower:
                raise RuntimeError(
                    f"Gemini request timed out: {error_text}"
                ) from exc

            raise RuntimeError(
                f"Resume parsing failed ({type(exc).__name__}): "
                f"{error_text}"
            ) from exc

    raise RuntimeError("Gemini request failed after retry.")
