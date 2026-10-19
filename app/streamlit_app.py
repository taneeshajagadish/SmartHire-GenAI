import sys
import re
import json
import html
import hashlib
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st


from src.generate.cv_suggestions import generate_cv_suggestions
from src.mentor.rag_chain import ask_mentor

from src.parsing.loader import load_resume
from src.parsing.resume_parser import parse_resume
from src.search.embed import embed_text
from src.search.job_search import (
    build_candidate_profile_text,
    load_faiss_index,
    search_faiss,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartHire GenAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM EDITORIAL VISUAL SYSTEM
# ============================================================

st.html(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500;1,600&family=DM+Sans:wght@400;500;600;700;800&display=swap');

* {
    box-sizing: border-box;
}

html,
body,
[class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 75% 8%,
            rgba(205, 153, 179, 0.09),
            transparent 28%
        ),
        radial-gradient(
            circle at 15% 65%,
            rgba(150, 110, 135, 0.05),
            transparent 30%
        ),
        #0b0a0b;

    color: #f5f0f2;
}


/* =========================================================
   PAGE WIDTH
   ========================================================= */

.block-container {
    max-width: 1500px;
    padding: 1.5rem 4rem 6rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #0a090a;
    border-right: 1px solid rgba(255,255,255,.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

.sidebar-brand {
    padding: 10px 8px 28px;
}

.sidebar-logo {
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #f5edf1;
}

.sidebar-logo span {
    color: #d5a0b9;
}

.sidebar-subtitle {
    color: #817982;
    font-size: 14px;
    margin-top: 7px;
    line-height: 1.55;
}

.sidebar-section {
    color: #756b73;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    margin: 26px 0 11px;
}

.sidebar-item {
    color: #c9c1c5;
    font-size: 14px;
    padding: 11px 10px;
    border-radius: 10px;
    margin-bottom: 4px;
    background: rgba(255,255,255,.015);
}

.sidebar-footer {
    margin-top: 45px;
    border-top: 1px solid rgba(255,255,255,.07);
    padding-top: 18px;
    color: #665e64;
    font-size: 12px;
    line-height: 1.75;
}


/* =========================================================
   HERO
   ========================================================= */

.hero-shell {
    position: relative;
    overflow: hidden;

    margin: 10px 0 34px;
    padding: 72px 64px 78px;

    min-height: 600px;

    border: 1px solid rgba(255,255,255,.075);
    border-radius: 4px;

    background:
        radial-gradient(
            circle at 82% 25%,
            rgba(213,160,185,.10),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(255,255,255,.025),
            rgba(255,255,255,.008)
        );

    box-shadow:
        0 35px 100px rgba(0,0,0,.35),
        inset 0 1px 0 rgba(255,255,255,.025);
}


/* subtle editorial glow */

.hero-shell::after {
    content: "";
    position: absolute;

    width: 1px;
    height: 420px;

    right: 12%;
    top: 120px;

    background: linear-gradient(
        to bottom,
        transparent,
        rgba(213,160,185,.24),
        transparent
    );

    transform: rotate(28deg);
}


/* =========================================================
   HERO TEXT
   ========================================================= */

.hero-eyebrow {
    position: relative;
    z-index: 2;

    color: #c995b0;

    font-size: 12px;
    font-weight: 800;

    letter-spacing: 3.2px;

    margin-bottom: 30px;
}

.hero-title {
    position: relative;
    z-index: 2;

    font-family: 'Cormorant Garamond', serif;

    font-size: clamp(75px, 8vw, 125px);

    line-height: .78;

    letter-spacing: -5px;

    font-weight: 500;

    margin: 0;

    color: #f7f0f3;
}

.hero-title span {
    display: block;

    margin-left: 65px;
    margin-top: 20px;

    font-style: italic;

    color: #d8a1ba;
}

.hero-description {
    position: relative;
    z-index: 2;

    max-width: 900px;

    color: #9b9298;

    font-size: 18px;

    line-height: 1.9;

    margin-top: 38px;
}

.hero-pill {
    position: relative;
    z-index: 2;

    display: inline-flex;

    align-items: center;
    gap: 10px;

    margin-top: 34px;

    padding: 12px 17px;

    border-radius: 999px;

    background: rgba(213,160,185,.055);

    border: 1px solid rgba(213,160,185,.20);

    color: #c8afb9;

    font-size: 13px;
    font-weight: 700;

    letter-spacing: .5px;
}

.hero-dot {
    display: inline-block;

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background: #d8a1ba;

    box-shadow: 0 0 14px rgba(216,161,186,.8);
}


/* =========================================================
   FEATURE CARDS
   ========================================================= */

.feature-card {
    min-height: 250px;

    padding: 30px 26px;

    border-radius: 3px;

    border: 1px solid rgba(255,255,255,.075);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.035),
            rgba(255,255,255,.008)
        );

    box-shadow:
        0 20px 55px rgba(0,0,0,.20);

    transition:
        transform .2s ease,
        border-color .2s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    border-color: rgba(213,160,185,.22);
}

.feature-icon {
    color: #d7a0b8;

    font-size: 29px;

    margin-bottom: 38px;
}

.feature-title {
    color: #f3edf0;

    font-weight: 800;

    font-size: 18px;

    margin-bottom: 15px;

    letter-spacing: -.2px;
}

.feature-description {
    color: #91878d;

    font-size: 18px;

    line-height: 1.8;
}


/* =========================================================
   SYSTEM STATUS
   ========================================================= */

.system-status {
    display: flex;

    align-items: center;

    gap: 10px;

    margin: 42px 0 48px;

    color: #756c73;

    font-size: 12px;

    font-weight: 600;

    letter-spacing: 2px;
}


/* =========================================================
   SECTION TYPOGRAPHY
   ========================================================= */

.section-kicker {
    color: #c995b0;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 2.4px;

    text-transform: uppercase;

    margin-bottom: 9px;
}

.section-title {
    color: #f4eef1;

    font-size: 34px;

    font-weight: 800;

    letter-spacing: -1.2px;

    margin-top: 0;
}

.section-description {
    color: #8e858b;

    font-size: 18px;

    line-height: 1.8;

    margin: 8px 0 26px;

    max-width: 950px;
}


/* =========================================================
   STATUS CARDS
   ========================================================= */

.status-card {
    padding: 25px;

    min-height: 130px;

    border: 1px solid rgba(255,255,255,.075);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.035),
            rgba(255,255,255,.01)
        );

    border-radius: 3px;
}

.status-label {
    color: #756d73;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 1.8px;

    text-transform: uppercase;
}

.status-value {
    color: #eee7ea;

    font-size: 19px;

    font-weight: 700;

    margin-top: 12px;

    line-height: 1.4;
}


/* =========================================================
   PROFILE
   ========================================================= */

.profile-card {
    padding: 28px;

    min-height: 180px;

    border: 1px solid rgba(255,255,255,.075);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.035),
            rgba(255,255,255,.008)
        );

    border-radius: 3px;
}

.skill-chip {
    display: inline-flex;

    align-items: center;

    padding: 10px 15px;

    margin: 4px;

    border-radius: 999px;

    background: rgba(213,160,185,.07);

    border: 1px solid rgba(213,160,185,.17);

    color: #ded3d8;

    font-size: 14px;

    font-weight: 600;
}

.skills-wrap {
    margin-top: 13px;
}


/* =========================================================
   MATCH BANNER
   ========================================================= */

.match-banner {
    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 20px;

    padding: 30px;

    margin: 25px 0;

    border-radius: 3px;

    background:
        linear-gradient(
            100deg,
            rgba(213,160,185,.08),
            rgba(255,255,255,.015)
        );

    border: 1px solid rgba(213,160,185,.15);
}

.match-big {
    font-size: 38px;

    font-weight: 800;

    letter-spacing: -1.5px;
}

.match-small {
    color: #81787e;

    font-size: 14px;

    margin-top: 5px;
}


/* =========================================================
   JOB RESULT CARDS
   ========================================================= */

.result-card {
    padding: 25px 28px;

    margin: 12px 0;

    border: 1px solid rgba(255,255,255,.065);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.032),
            rgba(255,255,255,.008)
        );

    border-radius: 3px;
}

.result-title {
    color: #f1eaed;

    font-size: 18px;

    font-weight: 750;

    line-height: 1.45;
}

.result-company {
    color: #a69ca2;

    font-size: 16px;

    margin-top: 7px;
}

.result-meta {
    color: #766d73;

    font-size: 14px;

    margin-top: 10px;
}

.score {
    color: #d9a3ba;

    font-size: 27px;

    font-weight: 800;
}


/* =========================================================
   CV CARDS
   ========================================================= */

.cv-card {
    padding: 28px;

    min-height: 170px;

    border: 1px solid rgba(255,255,255,.075);

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.035),
            rgba(255,255,255,.008)
        );

    border-radius: 3px;
}

.cv-card-title {
    color: #8d8289;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 2px;

    text-transform: uppercase;
}

.cv-card-item {
    color: #d0c7cc;

    font-size: 17px;

    line-height: 1.75;

    margin-top: 13px;
}


/* =========================================================
   MENTOR
   ========================================================= */

.mentor-header {
    padding: 32px;

    border: 1px solid rgba(213,160,185,.14);

    background:
        linear-gradient(
            120deg,
            rgba(213,160,185,.075),
            rgba(255,255,255,.012)
        );

    border-radius: 3px;
}

.mentor-title {
    color: #f3ecef;

    font-size: 29px;

    font-weight: 800;

    letter-spacing: -.8px;
}

.mentor-subtitle {
    color: #91868d;

    font-size: 18px;

    line-height: 1.8;

    margin-top: 10px;

    max-width: 900px;
}


/* =========================================================
   SMALL NOTES
   ========================================================= */

.tiny-note {
    color: #70676d;

    font-size: 14px;

    line-height: 1.7;

    margin-top: 10px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

div.stButton > button {
    border-radius: 3px;

    border: 1px solid rgba(213,160,185,.30);

    background:
        linear-gradient(
            135deg,
            rgba(213,160,185,.16),
            rgba(213,160,185,.045)
        );

    color: #f4edf0;

    font-weight: 750;

    font-size: 15px;

    padding: .75rem 1.35rem;
}

div.stButton > button:hover {
    border-color: rgba(213,160,185,.55);

    background:
        linear-gradient(
            135deg,
            rgba(213,160,185,.23),
            rgba(213,160,185,.07)
        );
}


/* =========================================================
   STREAMLIT CLEANUP
   ========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

hr {
    border-color: rgba(255,255,255,.07) !important;

    margin: 45px 0 !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    border: 1px solid rgba(255,255,255,.075);

    padding: 10px;

    background: rgba(255,255,255,.012);
}


/* =========================================================
   CHAT
   ========================================================= */

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.015);
}

</style>
"""
)


# ============================================================
# SESSION STATE
# ============================================================

for key, default in {
    "profile": None,
    "resume_text": None,
    "job_matches": [],
    "cv_suggestions": None,
    "processed_file": None,
    "processed_file_hash": None,
}.items():

    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# PERFORMANCE CACHE
# ============================================================


@st.cache_resource(show_spinner=False)
def get_faiss_store():
    """Load and reuse the FAISS index and metadata."""
    return load_faiss_index()



@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the sentence-transformer model once per Streamlit process."""
    from src.search.embed import get_model

    return get_model()


@st.cache_data(show_spinner=False)
def parse_resume_cached(resume_text: str):
    """Cache structured resume parsing."""
    return parse_resume(resume_text)


@st.cache_data(show_spinner=False)
def embed_candidate_cached(candidate_text: str):
    """Create a normalized candidate embedding and cache the result."""
    import numpy as np

    model = get_embedding_model()

    embeddings = model.encode(
        [candidate_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings[0].astype(np.float32).tolist()

@st.cache_data(show_spinner=False, ttl=1)
def search_jobs_cached(candidate_embedding, candidate_text):
    index, metadata = get_faiss_store()
    return search_faiss(
        index,
        candidate_embedding,
        metadata,
        top_k=10,
        candidate_profile_text=candidate_text,
    )


@st.cache_data(show_spinner=False)
def generate_cv_suggestions_cached(
    profile_json: str,
    job_json: str,
):
    """Cache AI-generated CV improvement suggestions."""
    return generate_cv_suggestions(
        json.loads(profile_json),
        json.loads(job_json),
    )


@st.cache_data(show_spinner=False, ttl=3600)
def ask_mentor_cached(question: str):
    """Cache career mentor responses for one hour."""
    return ask_mentor(question)
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                ✦ <span>SmartHire</span> GenAI
            </div>

            <div class="sidebar-subtitle">
                Resume Matching & AI Career Mentor
            </div>

        </div>

        <div class="sidebar-section">
            WORKSPACE
        </div>

        <div class="sidebar-item">
            📄 &nbsp; Resume Analysis
        </div>

        <div class="sidebar-item">
            ◎ &nbsp; Semantic Job Matches
        </div>

        <div class="sidebar-item">
            ✧ &nbsp; CV Improvement
        </div>

        <div class="sidebar-item">
            ◉ &nbsp; AI Career Mentor
        </div>

        <div class="sidebar-section">
            SYSTEM
        </div>

        <div class="sidebar-item">
            🛡️ &nbsp; Guardrails Active
        </div>

        <div class="sidebar-item">
            ⚡ &nbsp; Cached Intelligence
        </div>

        <div class="sidebar-footer">
            <b style="color:#91878d">
                SmartHire GenAI
            </b>
            <br>
            LLMs · Embeddings · FAISS · RAG
        </div>
        """
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero-shell">

        <div class="hero-eyebrow">
            SMART CAREER INTELLIGENCE · GENAI
        </div>

        <h1 class="hero-title">
            find your fit.
            <span>know your worth.</span>
        </h1>

        <div class="hero-description">
            Upload your resume and let SmartHire turn it into career
            intelligence — understand your profile, discover semantically
            relevant opportunities, strengthen your CV, and get grounded
            guidance from your AI Career Mentor.
        </div>

        <div class="hero-pill">
            <span class="hero-dot"></span>
            SYSTEM ONLINE · RESUME → MATCH → IMPROVE → MENTOR
        </div>

    </div>
    """
)


# ============================================================
# FEATURE CARDS
# ============================================================

feature_columns = st.columns(4)

features = [

    (
        "✦",
        "Resume Intelligence",
        "Turn your PDF or DOCX into a structured candidate profile."
    ),

    (
        "◎",
        "Semantic Matching",
        "Discover relevant opportunities through embedding-powered FAISS search."
    ),

    (
        "✧",
        "CV Improvement",
        "Identify skill gaps, strengthen weak bullets, and rewrite your summary."
    ),

    (
        "◉",
        "AI Career Mentor",
        "Ask career questions and receive grounded guidance from your knowledge base."
    ),

]


for column, (icon, title, description) in zip(
    feature_columns,
    features,
):

    with column:

        st.html(
            f"""
            <div class="feature-card">

                <div class="feature-icon">
                    {icon}
                </div>

                <div class="feature-title">
                    {title}
                </div>

                <div class="feature-description">
                    {description}
                </div>

            </div>
            """
        )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.html(
    """
    <div class="system-status">

        <span class="hero-dot"></span>

        LOCAL EMBEDDINGS · FAISS READY · RAG READY · GUARDRAILS ACTIVE

    </div>
    """
)


# ============================================================
# RESUME ANALYSIS
# ============================================================

st.html(
    """
    <div class="section-kicker">
        BEGIN HERE
    </div>

    <div class="section-title">
        Resume Analysis
    </div>

    <div class="section-description">
        Upload a PDF or DOCX and let SmartHire build your career profile.
    </div>
    """
)


uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX",
)


if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    is_new_resume = (
        st.session_state.processed_file_hash
        != file_hash
    )


    if is_new_resume:

        st.html(
            """
            <div class="tiny-note">
                New resume detected. First analysis may call the LLM;
                repeat analyses of the same resume are cached.
            </div>
            """
        )


        if st.button("✦ Analyze My Resume"):

            with st.status(
                "Building your career profile...",
                expanded=True,
            ) as status:

                try:

                    suffix = Path(
                        uploaded_file.name
                    ).suffix.lower()


                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix,
                    ) as temp_file:

                        temp_file.write(
                            file_bytes
                        )

                        temp_path = temp_file.name


                    st.write(
                        "✓ Reading resume"
                    )

                    resume_text = load_resume(
                        temp_path
                    )


                    st.write(
                        "✓ Extracting structured profile"
                    )

                    profile = parse_resume_cached(
                        resume_text
                    )


                    st.write(
                        "✓ Creating candidate representation"
                    )

                    candidate_text = (
                        build_candidate_profile_text(
                            profile
                        )
                    )

                    candidate_embedding = (
                        embed_candidate_cached(
                            candidate_text
                        )
                    )


                    st.write(
                        "✓ Searching 21,996 indexed jobs"
                    )

                    job_matches = (
                        search_jobs_cached(
                            candidate_embedding,
                            candidate_text,
                        )
                    )
                    st.write("DEBUG — Jobs returned:", len(job_matches))


                    st.session_state.resume_text = (
                        resume_text
                    )

                    st.session_state.profile = (
                        profile
                    )

                    st.session_state.job_matches = (
                        job_matches
                    )

                    st.session_state.processed_file = (
                        uploaded_file.name
                    )

                    st.session_state.processed_file_hash = (
                        file_hash
                    )

                    st.session_state.cv_suggestions = None


                    status.update(
                        label="SmartHire analysis complete ✦",
                        state="complete",
                        expanded=False,
                    )

                    st.success(
                        "Your career intelligence is ready."
                    )


                except Exception as error:

                    status.update(
                        label="Analysis needs attention",
                        state="error",
                        expanded=True,
                    )

                    st.error(
                        f"Something went wrong while analyzing the resume: {error}"
                    )


    else:

        st.html(
            """
            <div class="hero-pill" style="margin-top:4px">

                <span class="hero-dot"></span>

                This resume is already analyzed —
                results restored from cache.

            </div>
            """
        )


# ============================================================
# PROFILE
# ============================================================

if st.session_state.profile:

    profile = st.session_state.profile

    st.markdown("---")

    st.html(
        """
        <div class="section-kicker">
            01 · RESUME INTELLIGENCE
        </div>

        <div class="section-title">
            This is what SmartHire sees.
        </div>

        <div class="section-description">
            Structured information extracted from your resume —
            ready for semantic matching.
        </div>
        """
    )


    cols = st.columns(4)

    values = [

        (
            "Candidate",
            profile.get(
                "name",
                "Not available"
            ),
        ),

        (
            "Target Role",
            profile.get(
                "target_role",
                "Not specified"
            ),
        ),

        (
            "Skills Detected",
            len(
                profile.get(
                    "skills",
                    []
                )
            ),
        ),

        (
            "Experience Items",
            len(
                profile.get(
                    "experience",
                    []
                )
            ),
        ),

    ]


    for col, (label, value) in zip(
        cols,
        values,
    ):

        with col:

            st.html(
                f"""
                <div class="status-card">

                    <div class="status-label">
                        {html.escape(str(label))}
                    </div>

                    <div class="status-value">
                        {html.escape(str(value))}
                    </div>

                </div>
                """
            )


    st.html(
        '<div style="height:20px"></div>'
    )


    left, right = st.columns(2)


    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    with left:

        st.html(
            """
            <div class="profile-card">

                <div class="section-title"
                     style="font-size:22px">

                    Skills

                </div>

                <div class="section-description"
                     style="font-size:16px">

                    Detected capabilities used
                    for semantic matching.

                </div>

            </div>
            """
        )


        skills = profile.get(
            "skills",
            []
        )


        if skills:

            skills_html = "".join(

                f"""
                <span class="skill-chip">
                    {html.escape(str(skill))}
                </span>
                """

                for skill in skills
            )


            st.html(
                f"""
                <div class="skills-wrap">
                    {skills_html}
                </div>
                """
            )

        else:

            st.html(
                """
                <div class="tiny-note">
                    No skills detected.
                </div>
                """
            )


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    with right:

        education = profile.get(
            "education",
            []
        )


        education_html = "".join(

            f"""
            <div class="cv-card-item">
                • {html.escape(str(item))}
            </div>
            """

            for item in education

        ) or """

        <div class="tiny-note">
            No education information detected.
        </div>

        """


        st.html(
            f"""
            <div class="profile-card">

                <div class="section-title"
                     style="font-size:22px">

                    Education

                </div>

                <div class="section-description"
                     style="font-size:16px">

                    Academic background extracted
                    from the resume.

                </div>

                {education_html}

            </div>
            """
        )


# ============================================================
# JOB MATCHES
# ============================================================

if st.session_state.job_matches:

    st.markdown("---")

    st.html(
        """
        <div class="section-kicker">
            02 · SEMANTIC DISCOVERY
        </div>

        <div class="section-title">
            Opportunities that understand your profile.
        </div>

        <div class="section-description">
            Ranked by embedding similarity against the
            pre-built FAISS job corpus.
        </div>
        """
    )


    st.html(
        f"""
        <div class="match-banner">

            <div>

                <div class="section-kicker">
                    MATCH ENGINE
                </div>

                <div class="match-big">
                    {min(len(st.session_state.job_matches), 10)}
                     opportunities
                </div>

                <div class="match-small">
                    Profile → embedding → FAISS → ranked results
                </div>

            </div>

            <div class="hero-pill">

                <span class="hero-dot"></span>

                SEARCH COMPLETE

            </div>

        </div>
        """
    )


    for position, job in enumerate(
        st.session_state.job_matches,
        start=1,
    ):

        title = (
            job.get("jobtitle")
            or job.get("title")
            or job.get("job_title")
            or "Job Opportunity"
        )


        company = (
            job.get("company")
            or "Company not specified"
        )


        location = (
            job.get("joblocation_address")
            or job.get("location")
            or "Location not specified"
        )


        # IMPORTANT:
        # Backend returns similarity_score.
        # Keep score as fallback for compatibility.

        score = job.get(
            "ranking_score",
            job.get(
                "similarity_score",
                job.get("score", 0),
            ),
        )


        try:

            score_display = (
                f"{float(score):.3f}"
            )

        except Exception:

            score_display = str(score)


        st.html(
            f"""
            <div class="result-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    gap:25px;
                ">

                    <div>

                        <div class="result-title">
                            {position}.
                            {html.escape(str(title))}
                        </div>

                        <div class="result-company">
                            {html.escape(str(company))}
                        </div>

                        <div class="result-meta">
                            📍 {html.escape(str(location))}
                        </div>

                    </div>


                    
                        <div style="text-align:right">

                            <div class="status-label">
                                Match Score
                        </div>


                        
                        <div class="score">
                             <div class="score">
                                {html.escape(score_display)}
                            </div>
                            <div class="status-label">
                                 Experience: {html.escape(str(job.get("experience_requirement", "Not specified in job data")))}
                            </div>
                        </div>


                    </div>

                </div>

            </div>
            """
        )


        description = (
            job.get("jobdescription")
            or job.get("description")
            or job.get("job_document")
            or ""
        )
        
        
        with st.expander("✦ Match insights", expanded=False):
            insight_cols = st.columns(2)

            role_score = float(job.get("role_match_score") or 0)
            skill_score = float(job.get("skill_match_score") or 0)

            with insight_cols[0]:
                st.metric("Role fit", f"{role_score:.0%}")

            with insight_cols[1]:
                st.metric("Skill overlap", f"{skill_score:.0%}")

            requirement = str(
                job.get(
                    "experience_requirement",
                    "Not specified in job data",
                )
            )

            candidate_level = str(
                job.get("candidate_experience_level", "unknown")
            ).lower()

            job_level = str(
                job.get("job_experience_level", "unknown")
            ).lower()

            requirement_lower = requirement.lower()

            
            # Assess experience suitability using the job's actual requirement.
            range_match = re.search(
                r"(\d{1,2})\s*(?:-|–|—|to)\s*(\d{1,2})\s*years?",
                requirement_lower,
            )

            if "senior/lead role" in requirement_lower:
                if candidate_level == "entry":
                    fit_label = "Experience gap — senior/lead role"
                elif candidate_level == "mid":
                    fit_label = "Potential experience gap — senior/lead role"
                elif candidate_level == "senior":
                    fit_label = "Potentially aligned — verify requirements"
                else:
                    fit_label = "Seniority cannot be determined"

            elif "not specified" in requirement_lower:
                fit_label = "Experience requirement not specified"

            elif range_match:
                minimum_years = int(range_match.group(1))
                maximum_years = int(range_match.group(2))

                if minimum_years > maximum_years:
                    fit_label = "Invalid experience range in job data"

                elif candidate_level == "entry":
                    if minimum_years == 0:
                        fit_label = (
                            "Potentially suitable for entry-level candidates"
                        )
                    elif minimum_years == 1:
                        fit_label = (
                            "Possible opportunity — check experience requirement"
                        )
                    else:
                        fit_label = (
                            "Experience gap — prior experience requested"
                        )

                elif candidate_level == "mid":
                    if minimum_years <= 2 and maximum_years >= 2:
                        fit_label = (
                            "Potentially aligned — verify actual experience"
                        )
                    elif minimum_years > 2:
                        fit_label = "Potential experience gap"
                    else:
                        fit_label = "Review experience requirements"

                elif candidate_level == "senior":
                    if maximum_years <= 2:
                        fit_label = "Potentially below experience level"
                    elif minimum_years >= 3:
                        fit_label = (
                            "Potentially aligned — verify actual experience"
                        )
                    else:
                        fit_label = "Review experience requirements"

                else:
                    fit_label = "Candidate experience level unknown"

            elif candidate_level == "entry" and job_level == "entry":
                fit_label = "Potentially aligned with entry-level profile"

            elif candidate_level == "senior" and job_level == "senior":
                fit_label = (
                    "Seniority appears aligned — verify requirements"
                )

            else:
                fit_label = "Review experience requirements"


            st.markdown("**Experience suitability**")
            st.markdown(
                f"""
                <div class="match-insight">
                    <div class="match-insight-label">EXPERIENCE FIT</div>
                    <div class="match-insight-title">
                        {html.escape(fit_label)}
                    </div>
                    <div class="match-insight-note">
                        Listed requirement: {html.escape(requirement)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            matched_skills = job.get("matched_skills") or []

            st.markdown("**Matching skills**")
            if matched_skills:
                st.write(
                    " · ".join(str(skill) for skill in matched_skills)
                )
            else:
                st.caption("No matching skills identified.")

            st.caption(
                f"Candidate level: {candidate_level.title()} "
                f"· Job level: {job_level.title()}"
            )


        with st.expander(
            "View job details"
        ):

            st.write(
                description
                if description
                else
                "No job description available."
            )


# ============================================================
# CV IMPROVEMENT
# ============================================================

if (
    st.session_state.profile
    and st.session_state.job_matches
):

    st.markdown("---")

    st.html(
        """
        <div class="section-kicker">
            03 · CV GLOW-UP
        </div>

        <div class="section-title">
            Make your resume hit harder.
        </div>

        <div class="section-description">
            Select a matched role and generate grounded
            suggestions based on your profile and the target job.
        </div>
        """
    )


    jobs = st.session_state.job_matches


    job_labels = [

        (
            f'{i + 1}. '
            f'{job.get("jobtitle") or job.get("title") or "Job Opportunity"}'
            f' — '
            f'{job.get("company") or "Unknown Company"}'
        )

        for i, job in enumerate(jobs)

    ]


    selected_label = st.selectbox(
        "Select a target job",
        job_labels,
    )


    selected_job = jobs[
        job_labels.index(
            selected_label
        )
    ]


    if st.button(
        "✧ Generate CV Improvements"
    ):

        with st.spinner(
            "Building your targeted CV review..."
        ):

            try:

                st.session_state.cv_suggestions = (
                    generate_cv_suggestions_cached(

                        json.dumps(
                            st.session_state.profile,
                            sort_keys=True,
                        ),

                        json.dumps(
                            selected_job,
                            sort_keys=True,
                        ),

                    )
                )

            except Exception as error:

                st.error(
                    f"Could not generate CV suggestions: {error}"
                )


# ============================================================
# CV RESULTS
# ============================================================

if st.session_state.cv_suggestions:

    suggestions = (
        st.session_state.cv_suggestions
    )


    missing_skills = (
        suggestions.get(
            "missing_skills",
            []
        )
    )


    weak_bullets = (
        suggestions.get(
            "weak_bullet_points",
            []
        )
    )


    improved_bullets = (
        suggestions.get(
            "improved_bullet_points",
            []
        )
    )


    summary = (
        suggestions.get(
            "rewritten_summary",
            ""
        )
    )


    a, b = st.columns(2)


    # --------------------------------------------------------
    # SKILL GAPS
    # --------------------------------------------------------

    with a:

        items = "".join(

            f"""
            <div class="cv-card-item">
                • {html.escape(str(x))}
            </div>
            """

            for x in missing_skills

        ) or """

        <div class="tiny-note">
            No major missing skills identified.
        </div>

        """


        st.html(
            f"""
            <div class="cv-card">

                <div class="cv-card-title">
                    SKILL GAPS
                </div>

                {items}

            </div>
            """
        )


    # --------------------------------------------------------
    # WEAK BULLETS
    # --------------------------------------------------------

    with b:

        items = "".join(

            f"""
            <div class="cv-card-item">
                • {html.escape(str(x))}
            </div>
            """

            for x in weak_bullets

        ) or """

        <div class="tiny-note">
            No weak bullet points identified.
        </div>

        """


        st.html(
            f"""
            <div class="cv-card">

                <div class="cv-card-title">
                    BULLETS TO STRENGTHEN
                </div>

                {items}

            </div>
            """
        )


    st.html(
        '<div style="height:18px"></div>'
    )


    # --------------------------------------------------------
    # IMPROVED BULLETS
    # --------------------------------------------------------

    improved = "".join(

        f"""
        <div class="cv-card-item">
            ✦ {html.escape(str(x))}
        </div>
        """

        for x in improved_bullets

    ) or """

    <div class="tiny-note">
        No rewritten bullets available.
    </div>

    """


    st.html(
        f"""
        <div class="cv-card">

            <div class="cv-card-title">
                IMPROVED BULLET POINTS
            </div>

            {improved}

        </div>
        """
    )


    # --------------------------------------------------------
    # REWRITTEN SUMMARY
    # --------------------------------------------------------

    if summary:

        st.html(
            f"""
            <div style="height:18px"></div>

            <div class="cv-card">

                <div class="cv-card-title">
                    REWRITTEN SUMMARY
                </div>

                <div class="cv-card-item">
                    {html.escape(str(summary))}
                </div>

            </div>
            """
        )


# ============================================================
# AI CAREER MENTOR
# ============================================================

st.markdown("---")


st.html(
    """
    <div class="mentor-header">

        <div class="mentor-title">
            ◉ AI Career Mentor
        </div>

        <div class="mentor-subtitle">
            Grounded career guidance powered by RAG over
            job postings and curated career guidance notes.
            Guardrails run before the mentor responds.
        </div>

    </div>
    """
)


st.html(
    """
    <div class="section-description">

        Ask about resumes, job matching, software engineering
        skills, interviews, projects, or career development.

    </div>
    """
)


question = st.chat_input(
    "Ask your AI Career Mentor..."
)


if question:

    with st.chat_message("user"):

        st.write(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking from your career knowledge base..."
        ):

            try:

                answer = ask_mentor_cached(
                    question
                )

                st.write(answer)

            except Exception as error:

                st.error(
                    f"Mentor error: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        margin-top:70px;
        padding-top:28px;
        border-top:1px solid rgba(255,255,255,.07);
        text-align:center;
        color:#625a60;
        font-size:13px;
        line-height:1.8;
    ">

        <b style="color:#897e85">
            SmartHire GenAI
        </b>

        <br>

        Resume Intelligence · Semantic Search ·
        CV Improvement · RAG Career Mentor · Guardrails

    </div>
    """
)