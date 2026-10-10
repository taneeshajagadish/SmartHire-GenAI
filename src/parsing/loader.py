from pathlib import Path

from docx import Document
from pypdf import PdfReader


def load_resume(file_path: str) -> str:
    """
    Load a PDF or DOCX resume and return its text.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Resume not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    elif suffix == ".docx":
        document = Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")

    text = text.strip()

    if not text:
        raise ValueError("Could not extract any text from the resume.")

    return text