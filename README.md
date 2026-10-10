# SmartHire GenAI — Resume Matching & AI Career Mentor

SmartHire GenAI is an AI-powered career portal that helps users understand their resumes, discover relevant job opportunities, improve their CVs for target roles, and receive grounded career guidance through an AI Career Mentor.

The system combines Large Language Models (LLMs), structured output, embeddings, FAISS vector search, Retrieval-Augmented Generation (RAG), LangChain, guardrails, and Streamlit.

## Project Overview

SmartHire GenAI provides an end-to-end career assistance workflow:

1. Upload a resume in PDF or DOCX format.
2. Extract a structured candidate profile using an LLM.
3. Match the candidate profile against a local job corpus using semantic search.
4. Display ranked job matches with similarity scores.
5. Generate CV improvement suggestions for a target role.
6. Provide an AI Career Mentor grounded in the available job corpus and career knowledge base.
7. Apply guardrails to validate inputs and reject irrelevant or unsafe requests.
8. Deliver the experience through a Streamlit web application.

## Key Features

### 1. Resume Parser
- Supports PDF and DOCX resumes.
- Extracts resume text.
- Uses an LLM for structured information extraction.
- Produces a validated profile containing name, skills, experience, education, and target role.

### 2. Semantic Job Search
- Represents job descriptions using embeddings.
- Stores job vectors in FAISS.
- Embeds candidate information for similarity search.
- Returns ranked job recommendations.

### 3. CV Improvement Generator
- Suggests missing skills.
- Identifies weak resume bullet points.
- Generates improved bullet points.
- Rewrites the professional summary.

### 4. AI Career Mentor
Uses RAG to answer career-related questions using the project's available knowledge sources. It is designed to avoid unsupported claims.

### 5. Guardrails
- Validates inputs.
- Rejects irrelevant and unsafe requests.
- Helps reduce unnecessary LLM calls.
- Supports grounded responses.

### 6. Streamlit Career Portal
Provides a unified interface for resume upload, candidate profiles, job matching, CV improvement, and career mentoring.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Gemini API | LLM capabilities |
| LangChain | RAG workflow |
| Sentence Transformers | Local text embeddings |
| FAISS | Vector similarity search |
| Streamlit | Web application |
| Pandas | Dataset processing |
| PyPDF | PDF loading |
| python-docx | DOCX loading |
| python-dotenv | Environment configuration |
| Git and GitHub | Version control |

## System Architecture

```text
Resume Upload (PDF/DOCX)
          |
          v
Resume Loader and Text Extraction
          |
          v
LLM Resume Parser
          |
          v
Structured Candidate Profile
          |
          v
Embedding Generation
          |
          v
FAISS Job Index
          |
          v
Ranked Job Matches


Career Question
      |
      v
Guardrails
      |
      v
RAG Retrieval <--- Job Corpus + Career Notes
      |
      v
Gemini LLM
      |
      v
Grounded Career Answer


All features are presented through Streamlit.
```

## Project Structure

```text
SmartHire-GenAI/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── jobs/              # Local dataset; not included in GitHub
│   ├── resumes/           # Private local resumes
│   └── career_notes/
├── vectorstore/           # Generated locally; not included in GitHub
├── notebooks/
│   ├── 01_embeddings_explore.ipynb
│   ├── 02_build_faiss.ipynb
│   └── 03_rag_prototype.ipynb
├── src/
│   ├── config.py
│   ├── evaluate.py
│   ├── data/
│   ├── parsing/
│   ├── search/
│   ├── generate/
│   ├── mentor/
│   └── safety/
├── app/
│   └── streamlit_app.py
└── reports/
    ├── answer_quality.md
    ├── dataset_quality.md
    └── final_report.pdf
```

## Setup

1. Install Python and Git.
2. Clone the repository.
3. Create and activate a virtual environment.
4. Install dependencies from `requirements.txt`.
5. Create a local `.env` file using `.env.example` as a template.
6. Add your Gemini API key to `.env`.
7. Place the permitted job dataset in `data/jobs/`.
8. Build the local embeddings and FAISS index using the project's embedding scripts or notebooks.
9. Start the application:

```bash
python -m streamlit run app/streamlit_app.py
```

## Configuration

Create a `.env` file containing:

```text
GEMINI_API_KEY=your_api_key_here
```

Never commit the actual `.env` file or expose API keys publicly.

## Dataset and Generated Files

The job dataset, private resumes, generated embeddings, and FAISS index are intentionally excluded from this repository. Obtain and configure the dataset locally before running features that depend on it.

## Evaluation

The project includes reports and test files for assessing job retrieval, mentor answers, and guardrail behavior. Evaluation should consider relevance, correctness, grounding, and helpfulness.

## Future Improvements

- More comprehensive retrieval evaluation.
- Improved error handling and input validation.
- Deployment and performance optimization.
- Expanded career guidance and resume evaluation.
