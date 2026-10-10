# Dataset Quality Report

## Dataset
Source: Kaggle — Jobs on Naukri.com

## Raw Dataset
- Records: 22,000
- Columns: 14
- Duplicate records: 0

## Cleaning Performed
The raw dataset was cleaned before semantic search.

The cleaning process:
- Removed rows without a job title or job description.
- Removed duplicate records using the unique job identifier.
- Normalized whitespace in text fields.
- Removed email addresses.
- Removed phone-number-like contact information.
- Preserved useful job information such as title, company, industry,
  location, experience, education, skills, and job description.

## Cleaned Dataset
- Records: 21,996
- Output file: `data/jobs/cleaned_jobs.csv`
- Additional field: `job_document`

## Job Document
Each job is converted into a combined text document containing:

- Job Title
- Company
- Industry
- Location
- Experience
- Education
- Skills
- Job Description

This combined document is used as the input for semantic embedding
and job matching.

## Data Quality Notes
The original dataset contains some noisy recruitment and promotional
content. The cleaning process removes obvious contact information while
preserving potentially useful job-related information.

The `skills` column is not treated as the only source of skills because
some job records contain limited or broad values in that field. The
combined `job_document` therefore provides richer information for
semantic matching.