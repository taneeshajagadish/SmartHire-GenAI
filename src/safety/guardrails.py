import re


# ============================================================
# SMART HIRE GUARDRAILS
# ============================================================

# Questions that are clearly outside the purpose of
# the SmartHire Career Mentor.
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|recipes|cooking)\b",
    r"\b(movie|movies|song|songs|lyrics)\b",
    r"\b(sports?|cricket|football|basketball)\b",
    r"\b(weather|temperature)\b",
    r"\b(politics?|election|president|prime minister)\b",
]


# Requests that should not be processed by the career mentor.
UNSAFE_PATTERNS = [
    r"\b(hack|hacking)\b",
    r"\b(password|steal credentials)\b",
    r"\b(malware|ransomware|virus)\b",
    r"\b(phishing)\b",
    r"\b(exploit|exploitation)\b",
]


def validate_user_input(question: str) -> str:
    """
    Validate and clean user input before it reaches an LLM.
    """

    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if len(question) > 2000:
        raise ValueError(
            "Question is too long. Please keep it below 2000 characters."
        )

    return question


def is_off_topic(question: str) -> bool:
    """
    Check whether the question is outside the career domain.
    """

    question_lower = question.lower()

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, question_lower):
            return True

    return False


def is_unsafe(question: str) -> bool:
    """
    Check for clearly unsafe or malicious requests.
    """

    question_lower = question.lower()

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, question_lower):
            return True

    return False


def check_user_input(question: str) -> tuple[bool, str]:
    """
    Main guardrail check.

    Returns:
        (True, cleaned_question)
        OR
        (False, rejection_message)
    """

    try:
        question = validate_user_input(question)

    except ValueError as error:
        return False, str(error)

    if is_unsafe(question):
        return (
            False,
            "I can't help with unsafe or malicious requests."
        )

    if is_off_topic(question):
        return (
            False,
            "I'm SmartHire AI Career Mentor. "
            "I can help with resumes, job matching, "
            "career skills, interviews, projects, and "
            "career development."
        )

    return True, question