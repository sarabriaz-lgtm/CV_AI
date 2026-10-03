import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO

from groq import Groq
from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HireTech AI",
    page_icon="👔",
    layout="wide"
)

st.title("👔 HireTech AI")
st.subheader("AI-Powered CV Screening & Candidate Ranking")

st.write(
    "Upload multiple CVs, define the job requirements, and HireTech AI "
    "will analyze, score, and rank candidates."
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "candidate_data": [],
    "processed": False,
    "analysis_errors": [],
    "job_info": {},
    "weight_experience": 30,
    "weight_technical": 30,
    "weight_education": 15,
    "weight_requirements": 15,
    "weight_certification": 10,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SAFE DATA FUNCTIONS
# ============================================================

def safe_string(value):
    """Convert any value into safe display text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):

        # Education dictionary
        if "degree" in value:

            parts = []

            if value.get("degree"):
                parts.append(
                    str(value["degree"])
                )

            if value.get("institution"):
                parts.append(
                    str(value["institution"])
                )

            if value.get("year"):
                parts.append(
                    str(value["year"])
                )

            return " | ".join(parts)

        # Generic dictionary
        parts = []

        for key, val in value.items():

            if val is not None and str(val).strip():

                parts.append(
                    f"{key}: {val}"
                )

        return " | ".join(parts)

    if isinstance(value, (list, tuple)):

        return "; ".join(
            safe_string(x)
            for x in value
            if safe_string(x)
        )

    return str(value).strip()


def safe_list(value):
    """Always return a list."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return []

        return [value]

    return [value]


def safe_join(value):
    """Safely convert any value to Excel/display text."""

    items = safe_list(value)

    output = []

    for item in items:

        text = safe_string(item)

        if text:
            output.append(text)

    return "; ".join(output)


def safe_number(value, default=0):

    if value is None:
        return default

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_docx_text(uploaded_file):

    doc = Document(uploaded_file)

    text = []

    # Paragraphs
    for paragraph in doc.paragraphs:

        if paragraph.text.strip():

            text.append(
                paragraph.text.strip()
            )

    # Tables
    for table in doc.tables:

        for row in table.rows:

            row_text = []

            for cell in row.cells:

                cell_text = cell.text.strip()

                if cell_text:

                    row_text.append(
                        cell_text
                    )

            if row_text:

                text.append(
                    " | ".join(row_text)
                )

    return "\n".join(text)


def extract_text(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(
            uploaded_file
        )

    if filename.endswith(".docx"):

        return extract_docx_text(
            uploaded_file
        )

    raise ValueError(
        "Only PDF and DOCX files are supported."
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = safe_string(text).lower()

    # Common spelling variations
    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "—",
        "-"
    )

    # Normalize ampersand
    text = text.replace(
        "&",
        " and "
    )

    # Keep letters, numbers, #, +, dots and hyphens
    text = re.sub(
        r"[^a-z0-9+#.\- ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def keyword_variants(keyword):

    keyword = safe_string(
        keyword
    )

    normalized = normalize_text(
        keyword
    )

    variants = {
        normalized
    }

    # Modeling / Modelling
    if "modeling" in normalized:

        variants.add(
            normalized.replace(
                "modeling",
                "modelling"
            )
        )

    if "modelling" in normalized:

        variants.add(
            normalized.replace(
                "modelling",
                "modeling"
            )
        )

    # Organization / Organisation
    if "organization" in normalized:

        variants.add(
            normalized.replace(
                "organization",
                "organisation"
            )
        )

    if "organisation" in normalized:

        variants.add(
            normalized.replace(
                "organisation",
                "organization"
            )
        )

    return variants


# ============================================================
# KEYWORD MATCHING
# ============================================================

def find_keywords_in_cv(cv_text, required_keywords):

    """
    IMPORTANT:

    Keywords are matched directly against the COMPLETE CV text.

    This does not depend on what the AI extracted as skills.
    """

    cv_normalized = normalize_text(
        cv_text
    )

    matched = []
    missing = []

    for keyword in required_keywords:

        keyword = safe_string(
            keyword
        )

        if not keyword:
            continue

        variants = keyword_variants(
            keyword
        )

        found = False

        for variant in variants:

            if variant and variant in cv_normalized:

                found = True
                break

        if found:

            matched.append(
                keyword
            )

        else:

            missing.append(
                keyword
            )

    return matched, missing


def keyword_score(
    matched_keywords,
    required_keywords
):

    required = [
        safe_string(x).lower()
        for x in required_keywords
        if safe_string(x)
    ]

    matched = [
        safe_string(x).lower()
        for x in matched_keywords
        if safe_string(x)
    ]

    if not required:

        return 100

    required_set = set(
        required
    )

    matched_set = set(
        matched
    )

    percentage = (
        len(
            required_set.intersection(
                matched_set
            )
        )
        /
        len(required_set)
        *
        100
    )

    return round(
        percentage,
        2
    )


# ============================================================
# JSON FUNCTIONS
# ============================================================

def clean_json(response_text):

    if not response_text:

        raise ValueError(
            "AI returned an empty response."
        )

    text = response_text.strip()

    # Remove code fences
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "AI response did not contain valid JSON."
        )

    return text[
        start:end + 1
    ]


def repair_json(
    client,
    model_name,
    broken_json
):

    prompt = f"""
Repair the following malformed JSON.

Return ONLY valid JSON.

Do not explain anything.
Do not use Markdown.
Do not use code fences.
Do not change the information.

JSON:

{broken_json}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON repair tool. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    repaired = response.choices[0].message.content

    cleaned = clean_json(
        repaired
    )

    return json.loads(
        cleaned
    )


def parse_json(
    client,
    model_name,
    response_text
):

    cleaned = clean_json(
        response_text
    )

    try:

        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:

        return repair_json(
            client,
            model_name,
            cleaned
        )


# ============================================================
# AI CV ANALYSIS
# ============================================================

def analyze_cv(
    client,
    model_name,
    cv_text,
    job_info
):

    prompt = f"""
You are a professional recruitment screening assistant.

Analyze the following CV for the specified job.

IMPORTANT:

- Use ONLY information from the CV.
- Do not invent information.
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not use code fences.
- years_relevant_experience must be a number.
- Education must be returned as objects containing degree,
  institution and year where available.

JOB TITLE:
{job_info["job_title"]}

MINIMUM EXPERIENCE:
{job_info["min_experience"]} years

JOB DESCRIPTION:
{job_info["job_description"]}

REQUIRED KEYWORDS:
{", ".join(job_info["keywords"])}

RETURN EXACTLY THIS STRUCTURE:

{{
    "candidate_name": "Full Name",
    "education": [
        {{
            "degree": "Degree",
            "institution": "Institution",
            "year": "Year"
        }}
    ],
    "years_relevant_experience": 0,
    "previous_roles": [],
    "technical_skills": [],
    "software_tools": [],
    "certifications": [],
    "relevant_experience_evidence": [],
    "strengths": [],
    "potential_gaps": []
}}

CV:

{cv_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a recruitment screening assistant. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    response_text = (
        response.choices[0].message.content
    )

    return parse_json(
        client,
        model_name,
        response_text
    )


# ============================================================
# SCORING
# ============================================================

def experience_score(
    years,
    minimum
):

    years = safe_number(
        years
    )

    minimum = safe_number(
        minimum
    )

    if years >= minimum + 4:
        return 100

    if years >= minimum + 2:
        return 90

    if years >= minimum:
        return 75

    if years >= max(
        0,
        minimum - 1
    ):
        return 50

    if years > 0:
        return 30

    return 0


def education_score(
    education
):

    education_text = normalize_text(
        safe_join(
            education
        )
    )

    if not education_text:

        return 0

    score = 50

    if (
        "water resources" in education_text
        or "hydrology" in education_text
        or "hydraulic" in education_text
    ):

        score = 95

    elif "civil engineering" in education_text:

        score = 80

    elif "environmental engineering" in education_text:

        score = 75

    elif "engineering" in education_text:

        score = 70

    if (
        "master" in education_text
        or "msc" in education_text
        or "m.sc" in education_text
    ):

        score = min(
            score + 5,
            100
        )

    return score


def certification_score(
    certifications
):

    certifications = safe_list(
        certifications
    )

    valid = []

    for certification in certifications:

        text = safe_string(
            certification
        ).lower()

        if text not in [
            "",
            "none",
            "n/a",
            "na",
            "not specified"
        ]:

            valid.append(
                text
            )

    if valid:
        return 100

    return 0


def calculate_scores(
    candidate,
    job_info,
    weights
):

    # --------------------------------------------------------
    # MATCH KEYWORDS DIRECTLY FROM FULL CV
    # --------------------------------------------------------

    matched_keywords, missing_keywords = (
        find_keywords_in_cv(
            candidate.get(
                "cv_text",
                ""
            ),
            job_info["keywords"]
        )
    )

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    exp_score = experience_score(
        candidate.get(
            "years_relevant_experience",
            0
        ),
        job_info[
            "min_experience"
        ]
    )

    technical_score_value = keyword_score(
        matched_keywords,
        job_info["keywords"]
    )

    education_score_value = education_score(
        candidate.get(
            "education",
            []
        )
    )

    requirement_score_value = keyword_score(
        matched_keywords,
        job_info["keywords"]
    )

    certification_score_value = certification_score(
        candidate.get(
            "certifications",
            []
        )
    )

    # --------------------------------------------------------
    # WEIGHTED SCORE
    # --------------------------------------------------------

    total_weight = (
        weights["experience"]
        + weights["technical"]
        + weights["education"]
        + weights["requirements"]
        + weights["certification"]
    )

    if total_weight <= 0:
        total_weight = 1

    exp_contribution = (
        exp_score
        * weights["experience"]
        / total_weight
    )

    technical_contribution = (
        technical_score_value
        * weights["technical"]
        / total_weight
    )

    education_contribution = (
        education_score_value
        * weights["education"]
        / total_weight
    )

    requirement_contribution = (
        requirement_score_value
        * weights["requirements"]
        / total_weight
    )

    certification_contribution = (
        certification_score_value
        * weights["certification"]
        / total_weight
    )

    overall = (
        exp_contribution
        + technical_contribution
        + education_contribution
        + requirement_contribution
        + certification_contribution
    )

    return {

        "experience_score":
            round(exp_score, 2),

        "technical_score":
            round(technical_score_value, 2),

        "education_score":
            round(education_score_value, 2),

        "requirement_score":
            round(requirement_score_value, 2),

        "certification_score":
            round(certification_score_value, 2),

        "matched_keywords":
            matched_keywords,

        "missing_keywords":
            missing_keywords,

        "experience_contribution":
            round(
                exp_contribution,
                2
            ),

        "technical_contribution":
            round(
                technical_contribution,
                2
            ),

        "education_contribution":
            round(
                education_contribution,
                2
            ),

        "requirements_contribution":
            round(
                requirement_contribution,
                2
            ),

        "certification_contribution":
            round(
                certification_contribution,
                2
            ),

        "overall_score":
            round(
                overall,
                2
            )
    }


# ============================================================
# RECALCULATE ALL CANDIDATES
# ============================================================

def recalculate_results():

    weights = {

        "experience":
            st.session_state.weight_experience,

        "technical":
            st.session_state.weight_technical,

        "education":
            st.session_state.weight_education,

        "requirements":
            st.session_state.weight_requirements,

        "certification":
            st.session_state.weight_certification
    }

    results = []

    for candidate in (
        st.session_state.candidate_data
    ):

        scores = calculate_scores(
            candidate,
            st.session_state.job_info,
            weights
        )

        result = candidate.copy()

        result.update(
            scores
        )

        results.append(
            result
        )

    results.sort(
        key=lambda x: x["overall_score"],
        reverse=True
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        result["rank"] = index

    return results


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ AI Settings"
)

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "AI Model",
    [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b"
    ]
)

st.sidebar.markdown("---")

st.sidebar.header(
    "⚖️ Scoring Weightage"
)

st.sidebar.slider(
    "Relevant Experience (%)",
    0,
    100,
    step=5,
    key="weight_experience"
)

st.sidebar.slider(
    "Technical Skills (%)",
    0,
    100,
    step=5,
    key="weight_technical"
)

st.sidebar.slider(
    "Education (%)",
    0,
    100,
    step=5,
    key="weight_education"
)

st.sidebar.slider(
    "Job Requirements (%)",
    0,
    100,
    step=5,
    key="weight_requirements"
)

st.sidebar.slider(
    "Certifications (%)",
    0,
    100,
    step=5,
    key="weight_certification"
)

total_weight = (
    st.session_state.weight_experience
    + st.session_state.weight_technical
    + st.session_state.weight_education
    + st.session_state.weight_requirements
    + st.session_state.weight_certification
)

st.sidebar.markdown(
    f"### Total Weight: **{total_weight}%**"
)

st.sidebar.caption(
    "Weights are automatically normalized."
)


# ============================================================
# JOB REQUIREMENTS
# ============================================================

st.header(
    "1️⃣ Job Requirements"
)

col1, col2 = st.columns(2)

with col1:

    job_title = st.text_input(
        "Job Title",
        placeholder="e.g. Senior Hydrologist"
    )

    min_experience = st.number_input(
        "Minimum Relevant Experience (Years)",
        min_value=0,
        max_value=50,
        value=3
    )

with col2:

    required_keywords_input = st.text_input(
        "Required Keywords",
        placeholder=(
            "HEC-RAS, HEC-HMS, ArcGIS, Hydrology"
        )
    )

    job_description = st.text_area(
        "Job Description",
        height=150
    )


required_keywords = [
    x.strip()
    for x in required_keywords_input.split(",")
    if x.strip()
]


# ============================================================
# CV UPLOAD
# ============================================================

st.header(
    "2️⃣ Upload Candidate CVs"
)

uploaded_files = st.file_uploader(
    "Upload multiple PDF or Word CVs",
    type=[
        "pdf",
        "docx"
    ],
    accept_multiple_files=True
)


# ============================================================
# ANALYZE
# ============================================================

st.header(
    "3️⃣ Analyze Candidates"
)

analyze = st.button(
    "🚀 Analyze CVs",
    type="primary",
    use_container_width=True
)


if analyze:

    if not api_key:

        st.error(
            "Please enter your Groq API key."
        )

    elif not uploaded_files:

        st.error(
            "Please upload CVs."
        )

    elif not job_title:

        st.error(
            "Please enter the job title."
        )

    else:

        try:

            client = Groq(
                api_key=api_key
            )

            st.session_state.job_info = {

                "job_title":
                    job_title,

                "min_experience":
                    min_experience,

                "job_description":
                    job_description,

                "keywords":
                    required_keywords
            }

            st.session_state.candidate_data = []

            st.session_state.analysis_errors = []

            progress = st.progress(
                0
            )

            status = st.empty()

            total = len(
                uploaded_files
            )

            for i, uploaded_file in enumerate(
                uploaded_files
            ):

                status.info(
                    f"Analyzing {uploaded_file.name}..."
                )

                try:

                    cv_text = extract_text(
                        uploaded_file
                    )

                    if not cv_text.strip():

                        raise ValueError(
                            "No readable text found in CV."
                        )

                    candidate = analyze_cv(
                        client,
                        model_name,
                        cv_text,
                        st.session_state.job_info
                    )

                    # Normalize candidate data
                    candidate["candidate_name"] = (
                        safe_string(
                            candidate.get(
                                "candidate_name",
                                "Not specified"
                            )
                        )
                    )

                    candidate["education"] = safe_list(
                        candidate.get(
                            "education",
                            []
                        )
                    )

                    candidate["previous_roles"] = safe_list(
                        candidate.get(
                            "previous_roles",
                            []
                        )
                    )

                    candidate["technical_skills"] = safe_list(
                        candidate.get(
                            "technical_skills",
                            []
                        )
                    )

                    candidate["software_tools"] = safe_list(
                        candidate.get(
                            "software_tools",
                            []
                        )
                    )

                    candidate["certifications"] = safe_list(
                        candidate.get(
                            "certifications",
                            []
                        )
                    )

                    candidate[
                        "relevant_experience_evidence"
                    ] = safe_list(
                        candidate.get(
                            "relevant_experience_evidence",
                            []
                        )
                    )

                    candidate["strengths"] = safe_list(
                        candidate.get(
                            "strengths",
                            []
                        )
                    )

                    candidate["potential_gaps"] = safe_list(
                        candidate.get(
                            "potential_gaps",
                            []
                        )
                    )

                    candidate[
                        "years_relevant_experience"
                    ] = safe_number(
                        candidate.get(
                            "years_relevant_experience",
                            0
                        )
                    )

                    # VERY IMPORTANT:
                    # Store complete CV text for reliable keyword matching
                    candidate["cv_text"] = cv_text

                    candidate["cv_file"] = (
                        uploaded_file.name
                    )

                    st.session_state.candidate_data.append(
                        candidate
                    )

                except Exception as e:

                    st.session_state.analysis_errors.append(
                        {
                            "file":
                                uploaded_file.name,

                            "error":
                                str(e)
                        }
                    )

                progress.progress(
                    (i + 1) / total
                )

            st.session_state.processed = True

            status.success(
                "CV analysis completed."
            )

        except Exception as e:

            st.error(
                f"Unable to start AI analysis: {str(e)}"
            )


# ============================================================
# ERRORS
# ============================================================

if st.session_state.analysis_errors:

    st.warning(
        "The following CVs could not be processed:"
    )

    for error in (
        st.session_state.analysis_errors
    ):

        st.error(
            f"{error['file']} — {error['error']}"
        )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.processed
    and st.session_state.candidate_data
):

    results = recalculate_results()

    st.markdown("---")

    st.header(
        "📊 Candidate Ranking"
    )


    # ========================================================
    # METRICS
    # ========================================================

    average_score = (
        sum(
            x["overall_score"]
            for x in results
        )
        / len(results)
    )

    highest_score = max(
        x["overall_score"]
        for x in results
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Candidates",
        len(results)
    )

    c2.metric(
        "Average Score",
        f"{average_score:.1f}%"
    )

    c3.metric(
        "Highest Score",
        f"{highest_score:.1f}%"
    )

    c4.metric(
        "Required Keywords",
        len(required_keywords)
    )


    # ========================================================
    # RANKING TABLE
    # ========================================================

    st.subheader(
        "🏆 Candidate Ranking"
    )

    ranking_rows = []

    for candidate in results:

        ranking_rows.append(
            {
                "Rank":
                    candidate["rank"],

                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            "Not specified"
                        )
                    ),

                "CV":
                    safe_string(
                        candidate.get(
                            "cv_file",
                            ""
                        )
                    ),

                "Overall Score":
                    candidate["overall_score"],

                "Experience":
                    candidate["experience_score"],

                "Technical":
                    candidate["technical_score"],

                "Education":
                    candidate["education_score"],

                "Requirements":
                    candidate["requirement_score"],

                "Certification":
                    candidate["certification_score"],

                "Matched Keywords":
                    safe_join(
                        candidate[
                            "matched_keywords"
                        ]
                    ),

                "Missing Keywords":
                    safe_join(
                        candidate[
                            "missing_keywords"
                        ]
                    )
            }
        )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # KEYWORD STATUS
    # ========================================================

    st.subheader(
        "🔑 Keyword Matching"
    )

    st.info(
        "Keyword matching is performed directly against the complete "
        "CV text. It does not depend on the AI's extracted skill list."
    )

    keyword_rows = []

    for candidate in results:

        matched = candidate[
            "matched_keywords"
        ]

        missing = candidate[
            "missing_keywords"
        ]

        keyword_rows.append(
            {
                "Rank":
                    candidate["rank"],

                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            "Not specified"
                        )
                    ),

                "Matched":
                    safe_join(
                        matched
                    ),

                "Missing":
                    safe_join(
                        missing
                    ),

                "Matched Count":
                    len(matched),

                "Missing Count":
                    len(missing),

                "Match %":
                    candidate[
                        "requirement_score"
                    ]
            }
        )

    keyword_df = pd.DataFrame(
        keyword_rows
    )

    st.dataframe(
        keyword_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SCORE COMPARISON
    # ========================================================

    st.subheader(
        "📈 Overall Score Comparison"
    )

    chart = ranking_df[
        [
            "Candidate",
            "Overall Score"
        ]
    ].copy()

    chart = chart.set_index(
        "Candidate"
    )

    st.bar_chart(
        chart
    )


    # ========================================================
    # WEIGHT CONTRIBUTION
    # ========================================================

    st.subheader(
        "⚖️ Weight Contribution"
    )

    contribution_rows = []

    for candidate in results:

        contribution_rows.append(
            {
                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            "Not specified"
                        )
                    ),

                "Experience":
                    candidate[
                        "experience_contribution"
                    ],

                "Technical":
                    candidate[
                        "technical_contribution"
                    ],

                "Education":
                    candidate[
                        "education_contribution"
                    ],

                "Requirements":
                    candidate[
                        "requirements_contribution"
                    ],

                "Certification":
                    candidate[
                        "certification_contribution"
                    ],

                "Overall":
                    candidate[
                        "overall_score"
                    ]
            }
        )

    contribution_df = pd.DataFrame(
        contribution_rows
    )

    st.dataframe(
        contribution_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DETAILED CANDIDATE
    # ========================================================

    st.markdown("---")

    st.header(
        "🔎 Detailed Candidate Analysis"
    )

    candidate_options = [
        (
            f"{x['rank']}. "
            f"{safe_string(x.get('candidate_name', 'Not specified'))}"
        )
        for x in results
    ]

    selected = st.selectbox(
        "Select Candidate",
        candidate_options
    )

    selected_index = (
        candidate_options.index(
            selected
        )
    )

    candidate = results[
        selected_index
    ]


    # ========================================================
    # CANDIDATE SUMMARY
    # ========================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Overall Score",
        f"{candidate['overall_score']:.1f}%"
    )

    c2.metric(
        "Rank",
        f"#{candidate['rank']}"
    )

    c3.metric(
        "Relevant Experience",
        f"{candidate['years_relevant_experience']:.1f} years"
    )

    st.write(
        f"**CV:** {candidate['cv_file']}"
    )


    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.subheader(
        "Score Breakdown"
    )

    score_df = pd.DataFrame(
        {
            "Factor": [
                "Experience",
                "Technical Skills",
                "Education",
                "Job Requirements",
                "Certifications"
            ],

            "Score": [
                candidate[
                    "experience_score"
                ],

                candidate[
                    "technical_score"
                ],

                candidate[
                    "education_score"
                ],

                candidate[
                    "requirement_score"
                ],

                candidate[
                    "certification_score"
                ]
            ],

            "Weight (%)": [
                st.session_state.weight_experience,
                st.session_state.weight_technical,
                st.session_state.weight_education,
                st.session_state.weight_requirements,
                st.session_state.weight_certification
            ],

            "Contribution": [
                candidate[
                    "experience_contribution"
                ],

                candidate[
                    "technical_contribution"
                ],

                candidate[
                    "education_contribution"
                ],

                candidate[
                    "requirements_contribution"
                ],

                candidate[
                    "certification_contribution"
                ]
            ]
        }
    )

    st.dataframe(
        score_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MATCHED / MISSING KEYWORDS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "✅ Matched Keywords"
        )

        if candidate[
            "matched_keywords"
        ]:

            for keyword in candidate[
                "matched_keywords"
            ]:

                st.success(
                    keyword
                )

        else:

            st.write(
                "No required keywords found."
            )

    with col2:

        st.subheader(
            "❌ Missing Keywords"
        )

        if candidate[
            "missing_keywords"
        ]:

            for keyword in candidate[
                "missing_keywords"
            ]:

                st.error(
                    keyword
                )

        else:

            st.success(
                "All required keywords found."
            )


    # ========================================================
    # EDUCATION
    # ========================================================

    st.subheader(
        "🎓 Education"
    )

    education = safe_list(
        candidate.get(
            "education",
            []
        )
    )

    if education:

        for item in education:

            # Display dictionary education properly
            if isinstance(item, dict):

                degree = safe_string(
                    item.get(
                        "degree",
                        ""
                    )
                )

                institution = safe_string(
                    item.get(
                        "institution",
                        ""
                    )
                )

                year = safe_string(
                    item.get(
                        "year",
                        ""
                    )
                )

                text_parts = []

                if degree:
                    text_parts.append(
                        degree
                    )

                if institution:
                    text_parts.append(
                        institution
                    )

                if year:
                    text_parts.append(
                        year
                    )

                st.write(
                    "• "
                    + " — ".join(
                        text_parts
                    )
                )

            else:

                st.write(
                    "• "
                    + safe_string(item)
                )

    else:

        st.write(
            "Not specified."
        )


    # ========================================================
    # TECHNICAL INFORMATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🛠 Technical Skills"
        )

        skills = safe_list(
            candidate.get(
                "technical_skills",
                []
            )
        )

        if skills:

            for item in skills:

                st.write(
                    "• "
                    + safe_string(item)
                )

        else:

            st.write(
                "Not specified."
            )


    with col2:

        st.subheader(
            "💻 Software / Tools"
        )

        software = safe_list(
            candidate.get(
                "software_tools",
                []
            )
        )

        if software:

            for item in software:

                st.write(
                    "• "
                    + safe_string(item)
                )

        else:

            st.write(
                "Not specified."
            )


    # ========================================================
    # OTHER INFORMATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "💼 Previous Roles"
        )

        roles = safe_list(
            candidate.get(
                "previous_roles",
                []
            )
        )

        if roles:

            for item in roles:

                st.write(
                    "• "
                    + safe_string(item)
                )

        else:

            st.write(
                "Not specified."
            )


        st.subheader(
            "📜 Certifications"
        )

        certifications = safe_list(
            candidate.get(
                "certifications",
                []
            )
        )

        if certifications:

            for item in certifications:

                st.write(
                    "• "
                    + safe_string(item)
                )

        else:

            st.write(
                "Not specified."
            )


    with col2:

        st.subheader(
            "📌 Relevant Experience Evidence"
        )

        evidence = safe_list(
            candidate.get(
                "relevant_experience_evidence",
                []
            )
        )

        if evidence:

            for item in evidence:

                st.write(
                    "• "
                    + safe_string(item)
                )

        else:

            st.write(
                "Not specified."
            )


    # ========================================================
    # STRENGTHS / GAPS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "💪 Strengths"
        )

        strengths = safe_list(
            candidate.get(
                "strengths",
                []
            )
        )

        if strengths:

            for item in strengths:

                st.success(
                    safe_string(item)
                )

        else:

            st.write(
                "Not specified."
            )


    with col2:

        st.subheader(
            "⚠️ Potential Gaps"
        )

        gaps = safe_list(
            candidate.get(
                "potential_gaps",
                []
            )
        )

        if gaps:

            for item in gaps:

                st.warning(
                    safe_string(item)
                )

        else:

            st.write(
                "No significant gaps identified."
            )


    # ========================================================
    # EXCEL EXPORT
    # ========================================================

    st.markdown("---")

    st.header(
        "📥 Download Results"
    )

    # Ranking
    export_ranking = ranking_df.copy()

    # Candidate details
    details_rows = []

    for candidate in results:

        details_rows.append(
            {
                "Rank":
                    candidate["rank"],

                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            ""
                        )
                    ),

                "CV File":
                    safe_string(
                        candidate.get(
                            "cv_file",
                            ""
                        )
                    ),

                "Overall Score":
                    candidate[
                        "overall_score"
                    ],

                "Relevant Experience (Years)":
                    candidate[
                        "years_relevant_experience"
                    ],

                "Experience Score":
                    candidate[
                        "experience_score"
                    ],

                "Technical Score":
                    candidate[
                        "technical_score"
                    ],

                "Education Score":
                    candidate[
                        "education_score"
                    ],

                "Job Requirements Score":
                    candidate[
                        "requirement_score"
                    ],

                "Certification Score":
                    candidate[
                        "certification_score"
                    ],

                "Matched Keywords":
                    safe_join(
                        candidate[
                            "matched_keywords"
                        ]
                    ),

                "Missing Keywords":
                    safe_join(
                        candidate[
                            "missing_keywords"
                        ]
                    ),

                "Education":
                    safe_join(
                        candidate.get(
                            "education",
                            []
                        )
                    ),

                "Technical Skills":
                    safe_join(
                        candidate.get(
                            "technical_skills",
                            []
                        )
                    ),

                "Software Tools":
                    safe_join(
                        candidate.get(
                            "software_tools",
                            []
                        )
                    ),

                "Certifications":
                    safe_join(
                        candidate.get(
                            "certifications",
                            []
                        )
                    ),

                "Previous Roles":
                    safe_join(
                        candidate.get(
                            "previous_roles",
                            []
                        )
                    ),

                "Experience Evidence":
                    safe_join(
                        candidate.get(
                            "relevant_experience_evidence",
                            []
                        )
                    ),

                "Strengths":
                    safe_join(
                        candidate.get(
                            "strengths",
                            []
                        )
                    ),

                "Potential Gaps":
                    safe_join(
                        candidate.get(
                            "potential_gaps",
                            []
                        )
                    )
            }
        )

    details_df = pd.DataFrame(
        details_rows
    )


    # Weights
    weights_df = pd.DataFrame(
        {
            "Scoring Factor": [
                "Relevant Experience",
                "Technical Skills",
                "Education",
                "Job Requirements",
                "Certifications"
            ],

            "Weight (%)": [
                st.session_state.weight_experience,
                st.session_state.weight_technical,
                st.session_state.weight_education,
                st.session_state.weight_requirements,
                st.session_state.weight_certification
            ]
        }
    )


    # Keyword sheet
    keyword_rows = []

    for candidate in results:

        matched = candidate[
            "matched_keywords"
        ]

        missing = candidate[
            "missing_keywords"
        ]

        keyword_rows.append(
            {
                "Rank":
                    candidate["rank"],

                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            ""
                        )
                    ),

                "Matched Keywords":
                    safe_join(
                        matched
                    ),

                "Matched Count":
                    len(matched),

                "Missing Keywords":
                    safe_join(
                        missing
                    ),

                "Missing Count":
                    len(missing),

                "Total Required":
                    len(required_keywords),

                "Keyword Match %":
                    candidate[
                        "requirement_score"
                    ]
            }
        )

    keyword_export_df = pd.DataFrame(
        keyword_rows
    )


    # --------------------------------------------------------
    # CREATE EXCEL
    # --------------------------------------------------------

    try:

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            export_ranking.to_excel(
                writer,
                sheet_name="Candidate Ranking",
                index=False
            )

            details_df.to_excel(
                writer,
                sheet_name="Candidate Details",
                index=False
            )

            weights_df.to_excel(
                writer,
                sheet_name="Scoring Weights",
                index=False
            )

            keyword_export_df.to_excel(
                writer,
                sheet_name="Keyword Matching",
                index=False
            )

        excel_buffer.seek(0)

        st.download_button(
            label="📊 Download Excel Results",
            data=excel_buffer.getvalue(),
            file_name="HireTech_AI_Results.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            "Excel file could not be generated. "
            f"Please check the data format. Details: {str(e)}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "HireTech AI | AI-assisted recruitment screening "
    "and transparent candidate scoring"
)
