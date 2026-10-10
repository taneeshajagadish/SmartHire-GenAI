from pathlib import Path
import re
import pandas as pd


RAW_FILE = Path("data/jobs/naukri_com-job_sample.csv")
CLEAN_FILE = Path("data/jobs/cleaned_jobs.csv")


REQUIRED_COLUMNS = [
    "company",
    "education",
    "experience",
    "industry",
    "jobdescription",
    "jobid",
    "joblocation_address",
    "jobtitle",
    "skills",
    "uniq_id",
]


def clean_text(value):
    """Clean text and remove obvious contact information."""

    if pd.isna(value):
        return ""

    text = str(value)

    # Remove email addresses.
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        " ",
        text,
    )

    # Remove phone-number-like sequences.
    text = re.sub(
        r"\b(?:\+?\d[\d\s().-]{8,}\d)\b",
        " ",
        text,
    )

    # Normalize repeated whitespace.
    text = " ".join(text.split())

    return text.strip()


def clean_jobs():
    """Load the raw job dataset and create a cleaned copy."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    # Remove rows where the essential job information is unavailable.
    df = df.dropna(subset=["jobtitle", "jobdescription"])

    # Clean text fields.
    text_columns = [
        "company",
        "education",
        "experience",
        "industry",
        "jobdescription",
        "joblocation_address",
        "jobtitle",
        "skills",
    ]

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    # Remove records with empty job descriptions after cleaning.
    df = df[df["jobdescription"].str.len() > 0]

    # Remove exact duplicate jobs based on the unique job identifier.
    df = df.drop_duplicates(subset=["uniq_id"])

    # Create a standardized text document for semantic search.
    df["job_document"] = (
        "Job Title: " + df["jobtitle"] +
        "\nCompany: " + df["company"] +
        "\nIndustry: " + df["industry"] +
        "\nLocation: " + df["joblocation_address"] +
        "\nExperience: " + df["experience"] +
        "\nEducation: " + df["education"] +
        "\nSkills: " + df["skills"] +
        "\nJob Description: " + df["jobdescription"]
    )

    df.to_csv(CLEAN_FILE, index=False)

    print("Job cleaning completed.")
    print(f"Cleaned records: {len(df)}")
    print(f"Output file: {CLEAN_FILE}")


if __name__ == "__main__":
    clean_jobs()