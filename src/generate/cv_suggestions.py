import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.generate.prompts import build_cv_improvement_prompt


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# REQUIRED FIELDS
# ============================================================

REQUIRED_FIELDS = {
    "missing_skills",
    "weak_bullet_points",
    "improved_bullet_points",
    "rewritten_summary",
}


# ============================================================
# VALIDATION
# ============================================================

def validate_cv_suggestions(result: dict) -> dict:
    """Validate the structured CV improvement response."""

    if not isinstance(result, dict):
        raise ValueError(
            "CV suggestions must be a dictionary."
        )

    if set(result.keys()) != REQUIRED_FIELDS:
        raise ValueError(
            "CV suggestions do not contain the required fields."
        )

    if not isinstance(result["missing_skills"], list):
        raise ValueError(
            "'missing_skills' must be a list."
        )

    if not isinstance(result["weak_bullet_points"], list):
        raise ValueError(
            "'weak_bullet_points' must be a list."
        )

    if not isinstance(result["improved_bullet_points"], list):
        raise ValueError(
            "'improved_bullet_points' must be a list."
        )

    if not isinstance(result["rewritten_summary"], str):
        raise ValueError(
            "'rewritten_summary' must be a string."
        )

    return result


# ============================================================
# GENERATE CV SUGGESTIONS
# ============================================================

def generate_cv_suggestions(
    candidate_profile: dict,
    target_job: dict,
) -> dict:
    """Generate structured CV improvement suggestions."""

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = build_cv_improvement_prompt(
        candidate_profile,
        target_job,
    )

    # --------------------------------------------------------
    # GEMINI API CALL
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",

            contents=prompt,

            config=types.GenerateContentConfig(
                temperature=0,

                response_mime_type="application/json",

                response_schema={
                    "type": "OBJECT",

                    "properties": {
                        "missing_skills": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING"
                            },
                        },

                        "weak_bullet_points": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING"
                            },
                        },

                        "improved_bullet_points": {
                            "type": "ARRAY",
                            "items": {
                                "type": "STRING"
                            },
                        },

                        "rewritten_summary": {
                            "type": "STRING"
                        },
                    },

                    "required": [
                        "missing_skills",
                        "weak_bullet_points",
                        "improved_bullet_points",
                        "rewritten_summary",
                    ],
                },

                max_output_tokens=1200,
            ),
        )

    except Exception as error:

        error_text = str(error)

        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if (
            "429" in error_text
            or "Too Many Requests" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):
            raise RuntimeError(
                "Gemini API rate limit reached. "
                "Please wait before generating CV improvements."
            ) from error

        # ----------------------------------------------------
        # TEMPORARY SERVICE ERROR
        # ----------------------------------------------------

        if (
            "503" in error_text
            or "service_unavailable" in error_text
            or "high demand" in error_text
            or "UNAVAILABLE" in error_text
        ):
            raise RuntimeError(
                "Gemini is temporarily unavailable. "
                "Please try again in a moment."
            ) from error

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        if (
            "timeout" in error_text.lower()
            or "timed out" in error_text.lower()
        ):
            raise RuntimeError(
                "Gemini took too long to generate CV improvements. "
                "Please try again."
            ) from error

        # ----------------------------------------------------
        # OTHER ERRORS
        # ----------------------------------------------------

        raise


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    if not response or not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )


    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        result = json.loads(
            response.text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "The LLM returned invalid JSON."
        ) from exc


    # ========================================================
    # VALIDATE
    # ========================================================

    return validate_cv_suggestions(
        result
    )