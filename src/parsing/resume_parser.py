
import json
import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Check that your .env file exists and contains your API key."
    )

client = genai.Client(api_key=API_KEY)

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


def _clean_response(text: str) -> str:
    """Remove Markdown code fences if the model returns them."""
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def _validate_profile(data: dict) -> dict:
    """Validate and normalize the extracted resume profile."""
    required_keys = {
        "name",
        "skills",
        "experience",
        "education",
        "target_role",
    }

    if not isinstance(data, dict):
        raise ValueError("The model did not return a JSON object.")

    if set(data.keys()) != required_keys:
        raise ValueError(
            "The model returned an unexpected profile structure."
        )

    for field in ("name", "target_role"):
        if not isinstance(data[field], str):
            raise ValueError(
                f"The '{field}' field must be a string."
            )

        data[field] = data[field].strip()

    for field in ("skills", "experience", "education"):
        if not isinstance(data[field], list):
            raise ValueError(
                f"The '{field}' field must be a list."
            )

        if not all(isinstance(item, str) for item in data[field]):
            raise ValueError(
                f"Every item in '{field}' must be a string."
            )

        data[field] = [
            item.strip()
            for item in data[field]
            if item.strip()
        ]

    return data


def parse_resume(resume_text: str) -> dict:
    """
    Extract a structured profile from resume text using Gemini.

    Returns:
        A validated dictionary containing:
        name, skills, experience, education, target_role.
    """
    if not isinstance(resume_text, str):
        raise ValueError("Resume content must be text.")

    resume_text = resume_text.strip()

    if not resume_text:
        raise ValueError(
            "The uploaded resume is empty or contains no readable text."
        )

    if len(resume_text) > 50000:
        resume_text = resume_text[:50000]

    prompt = f"""
You are a professional resume information extraction system.

Extract information ONLY from the supplied resume.

Rules:
1. Never invent personal information, skills, qualifications,
   employment history, or achievements.
2. Keep the person's name empty if it cannot be identified.
3. Extract skills as concise individual skill names.
4. Summarize relevant work experience, internships, and projects.
5. Extract education and qualifications.
6. For target_role, prioritize an explicitly stated target,
   objective, headline, or professional profile.
7. Infer a target role only when the resume strongly supports it.
   Otherwise return an empty string.
8. Return only the requested structured JSON.

RESUME:
--- BEGIN RESUME ---
{resume_text}
--- END RESUME ---
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1500,
                response_mime_type="application/json",
                response_schema=RESUME_SCHEMA,
            ),
        )

        response_text = response.text

        if not response_text:
            raise ValueError(
                "Gemini returned an empty response. Try again."
            )

        parsed_data = json.loads(_clean_response(response_text))

        return _validate_profile(parsed_data)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON while parsing the resume."
        ) from exc

    except Exception as exc:
        error_text = str(exc)
        error_lower = error_text.lower()

        if any(
            term in error_lower
            for term in ("429", "resource_exhausted", "rate limit")
        ):
            raise RuntimeError(
                "Gemini API rate limit reached. Please wait and retry."
            ) from exc

        if any(
            term in error_lower
            for term in ("503", "unavailable", "overloaded")
        ):
            raise RuntimeError(
                "Gemini is temporarily unavailable. Please retry shortly."
            ) from exc

        if any(
            term in error_lower
            for term in ("timeout", "timed out")
        ):
            raise RuntimeError(
                "The request timed out. Please try again."
            ) from exc

        raise RuntimeError(
            f"Resume parsing failed: {type(exc).__name__}: {exc}"
        ) from exc
