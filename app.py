import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
from groq import Groq
from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HireTech AI",
    page_icon="👔",
    layout="wide"
)

st.title("👔 HireTech AI")
st.subheader("AI-Powered CV Screening & Candidate Ranking")

st.markdown(
    """
    **HireTech AI** analyzes multiple CVs against job-specific requirements,
    extracts relevant candidate information, calculates transparent scores,
    and ranks candidates based on recruiter-defined weightages.
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "candidate_data" not in st.session_state:
    st.session_state.candidate_data = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "analysis_errors" not in st.session_state:
    st.session_state.analysis_errors = []

if "job_info" not in st.session_state:
    st.session_state.job_info = {}

if "weight_experience" not in st.session_state:
    st.session_state.weight_experience = 30

if "weight_technical" not in st.session_state:
    st.session_state.weight_technical = 30

if "weight_education" not in st.session_state:
    st.session_state.weight_education = 15

if "weight_requirements" not in st.session_state:
    st.session_state.weight_requirements = 15

if "weight_certification" not in st.session_state:
    st.session_state.weight_certification = 10


# ============================================================
# GENERAL DATA-SAFETY HELPERS
# ============================================================

def safe_list(value):
    """
    Convert AI output into a safe Python list.

    Handles:
    - list
    - tuple
    - string
    - None
    - numbers
    - unexpected values
    """

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


def safe_string(value):
    """
    Convert any value safely into a string.
    """

    if value is None:
        return ""

    return str(value).strip()


def safe_join(value, separator="; "):
    """
    Safely convert a list/string/None into text.

    This prevents Excel export errors caused by:
    '; '.join(None)
    or
    '; '.join([1, 2, 3])
    """

    items = safe_list(value)

    cleaned = []

    for item in items:

        if item is None:
            continue

        if isinstance(item, dict):
            cleaned.append(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
            )

        else:
            text = str(item).strip()

            if text:
                cleaned.append(text)

    return separator.join(cleaned)


def safe_number(value, default=0):
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        return float(value)

    except (ValueError, TypeError):
        return default


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = []

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text.append(page_text)

        return "\n".join(text)

    except Exception as e:

        raise Exception(
            f"PDF extraction failed: {str(e)}"
        )


def extract_docx_text(uploaded_file):

    try:

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

    except Exception as e:

        raise Exception(
            f"DOCX extraction failed: {str(e)}"
        )


def extract_text(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(
            uploaded_file
        )

    elif filename.endswith(".docx"):

        return extract_docx_text(
            uploaded_file
        )

    else:

        raise Exception(
            "Unsupported file format."
        )


# ============================================================
# JSON HANDLING
# ============================================================

def clean_json_response(response_text):

    if not response_text:

        raise ValueError(
            "AI returned an empty response."
        )

    response_text = response_text.strip()

    # Remove Markdown code fences
    response_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"\s*```$",
        "",
        response_text
    )

    response_text = response_text.strip()

    # Extract JSON object
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found in AI response."
        )

    return response_text[
        start:end + 1
    ]


def repair_json_with_ai(
    client,
    model_name,
    broken_response
):

    repair_prompt = f"""
You are a JSON repair assistant.

The following response should be a JSON object,
but it contains JSON syntax errors.

Repair the JSON.

IMPORTANT:
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not use code fences.
- Do not add explanations.
- Do not change the meaning of the information.
- Make sure all commas, brackets and quotation marks are correct.

BROKEN RESPONSE:

{broken_response}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You repair malformed JSON. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": repair_prompt
            }
        ],
        temperature=0
    )

    repaired = response.choices[0].message.content

    cleaned = clean_json_response(
        repaired
    )

    return json.loads(cleaned)


def parse_ai_json(
    client,
    model_name,
    response_text
):

    cleaned = clean_json_response(
        response_text
    )

    try:

        return json.loads(cleaned)

    except json.JSONDecodeError:

        return repair_json_with_ai(
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

    keywords = job_info["keywords"]

    prompt = f"""
You are an AI recruitment screening assistant.

Analyze the candidate CV against the job requirements.

IMPORTANT RULES:

1. Use ONLY information contained in the CV.
2. Do NOT invent qualifications or experience.
3. years_relevant_experience must be a number.
4. Identify experience relevant to the specified job.
5. Identify technical skills and software tools.
6. Identify certifications.
7. Return ONLY one valid JSON object.
8. Do not use Markdown.
9. Do not use code fences.
10. Do not write explanations outside the JSON.
11. Make sure the JSON is syntactically valid.

JOB INFORMATION

Job Title:
{job_info["job_title"]}

Minimum Relevant Experience:
{job_info["min_experience"]} years

Job Description:
{job_info["job_description"]}

Required Keywords:
{", ".join(keywords)}

RETURN EXACTLY THIS JSON STRUCTURE:

{{
    "candidate_name": "Full name or Not specified",
    "education": [],
    "years_relevant_experience": 0,
    "previous_roles": [],
    "technical_skills": [],
    "software_tools": [],
    "certifications": [],
    "required_keywords_found": [],
    "required_keywords_missing": [],
    "relevant_experience_evidence": [],
    "strengths": [],
    "potential_gaps": []
}}

CANDIDATE CV:

{cv_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional recruitment "
                    "screening assistant. Return valid JSON only."
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

    return parse_ai_json(
        client,
        model_name,
        response_text
    )


# ============================================================
# KEYWORD MATCHING
# ============================================================

def normalize_text(text):

    text = safe_string(text).lower()

    # Normalize common variations
    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "—",
        "-"
    )

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

    """
    Generate simple variants for better matching.
    """

    original = safe_string(
        keyword
    )

    normalized = normalize_text(
        original
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


def find_matched_keywords(
    candidate,
    required_keywords
):

    """
    Determine matched and missing job keywords.

    Matching is based on:
    - Technical skills
    - Software/tools
    - Previous roles
    - Education
    - Relevant experience evidence
    - Strengths
    """

    searchable_items = []

    searchable_items.extend(
        safe_list(
            candidate.get(
                "technical_skills",
                []
            )
        )
    )

    searchable_items.extend(
        safe_list(
            candidate.get(
                "software_tools",
                []
            )
        )
    )

    searchable_items.extend(
        safe_list(
            candidate.get(
                "previous_roles",
                []
            )
        )
    )

    searchable_items.extend(
        safe_list(
            candidate.get(
                "education",
                []
            )
        )
    )

    searchable_items.extend(
        safe_list(
            candidate.get(
                "relevant_experience_evidence",
                []
            )
        )
    )

    searchable_items.extend(
        safe_list(
            candidate.get(
                "strengths",
                []
            )
        )
    )

    searchable_text = normalize_text(
        " ".join(
            safe_string(x)
            for x in searchable_items
        )
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

            if variant and variant in searchable_text:

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


def calculate_keyword_score(
    matched,
    required
):

    required = [
        safe_string(x).lower()
        for x in safe_list(required)
        if safe_string(x)
    ]

    matched = [
        safe_string(x).lower()
        for x in safe_list(matched)
        if safe_string(x)
    ]

    if not required:

        return 100

    unique_required = set(
        required
    )

    unique_matched = set(
        matched
    )

    count = len(
        unique_required.intersection(
            unique_matched
        )
    )

    return round(
        count /
        len(unique_required)
        * 100,
        2
    )


# ============================================================
# EXPERIENCE SCORING
# ============================================================

def calculate_experience_score(
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

    elif years >= minimum + 2:

        return 90

    elif years >= minimum:

        return 75

    elif years >= max(
        0,
        minimum - 1
    ):

        return 50

    elif years > 0:

        return 30

    return 0


# ============================================================
# EDUCATION SCORING
# ============================================================

def calculate_education_score(
    education
):

    education_list = safe_list(
        education
    )

    if not education_list:

        return 0

    text = normalize_text(
        " ".join(
            safe_string(x)
            for x in education_list
        )
    )

    score = 50

    if (
        "water resources" in text
        or "hydrology" in text
        or "hydraulic" in text
    ):

        score = 95

    elif "civil engineering" in text:

        score = 80

    elif "environmental engineering" in text:

        score = 75

    elif "engineering" in text:

        score = 70

    if (
        "master" in text
        or "msc" in text
        or "m.sc" in text
    ):

        score = min(
            score + 5,
            100
        )

    return score


# ============================================================
# CERTIFICATION SCORING
# ============================================================

def calculate_certification_score(
    certifications
):

    certifications = safe_list(
        certifications
    )

    valid = []

    for item in certifications:

        value = safe_string(
            item
        ).lower()

        if value not in [
            "",
            "none",
            "not specified",
            "n/a",
            "na"
        ]:

            valid.append(
                value
            )

    if valid:

        return 100

    return 0


# ============================================================
# COMPLETE CANDIDATE SCORING
# ============================================================

def calculate_candidate_scores(
    candidate,
    job_info,
    weights
):

    required_keywords = job_info[
        "keywords"
    ]

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    matched_keywords, missing_keywords = (
        find_matched_keywords(
            candidate,
            required_keywords
        )
    )

    # --------------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------------

    experience_score = (
        calculate_experience_score(
            candidate.get(
                "years_relevant_experience",
                0
            ),
            job_info[
                "min_experience"
            ]
        )
    )

    # --------------------------------------------------------
    # TECHNICAL
    # --------------------------------------------------------

    technical_score = (
        calculate_keyword_score(
            matched_keywords,
            required_keywords
        )
    )

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    education_score = (
        calculate_education_score(
            candidate.get(
                "education",
                []
            )
        )
    )

    # --------------------------------------------------------
    # JOB REQUIREMENTS
    # --------------------------------------------------------

    requirement_score = (
        calculate_keyword_score(
            matched_keywords,
            required_keywords
        )
    )

    # --------------------------------------------------------
    # CERTIFICATION
    # --------------------------------------------------------

    certification_score = (
        calculate_certification_score(
            candidate.get(
                "certifications",
                []
            )
        )
    )

    # --------------------------------------------------------
    # WEIGHTED CALCULATION
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

    experience_contribution = (
        experience_score
        * weights["experience"]
        / total_weight
    )

    technical_contribution = (
        technical_score
        * weights["technical"]
        / total_weight
    )

    education_contribution = (
        education_score
        * weights["education"]
        / total_weight
    )

    requirements_contribution = (
        requirement_score
        * weights["requirements"]
        / total_weight
    )

    certification_contribution = (
        certification_score
        * weights["certification"]
        / total_weight
    )

    overall_score = (
        experience_contribution
        + technical_contribution
        + education_contribution
        + requirements_contribution
        + certification_contribution
    )

    return {

        "experience_score":
            round(
                experience_score,
                2
            ),

        "technical_score":
            round(
                technical_score,
                2
            ),

        "education_score":
            round(
                education_score,
                2
            ),

        "requirement_score":
            round(
                requirement_score,
                2
            ),

        "certification_score":
            round(
                certification_score,
                2
            ),

        "matched_keywords":
            matched_keywords,

        "missing_keywords":
            missing_keywords,

        "experience_contribution":
            round(
                experience_contribution,
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
                requirements_contribution,
                2
            ),

        "certification_contribution":
            round(
                certification_contribution,
                2
            ),

        "overall_score":
            round(
                overall_score,
                2
            )
    }


# ============================================================
# RECALCULATE RESULTS
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

        scores = (
            calculate_candidate_scores(
                candidate,
                st.session_state.job_info,
                weights
            )
        )

        result = candidate.copy()

        result.update(
            scores
        )

        results.append(
            result
        )

    # Highest score first
    results.sort(
        key=lambda x:
            x["overall_score"],
        reverse=True
    )

    # Assign ranking
    for index, candidate in enumerate(
        results,
        start=1
    ):

        candidate["rank"] = index

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

st.sidebar.caption(
    "These weights determine how much each factor contributes "
    "to the final candidate score."
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

st.sidebar.info(
    "Weights are automatically normalized, so they do not "
    "have to total exactly 100%."
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
        placeholder="e.g., Senior Hydrologist"
    )

    min_experience = st.number_input(
        "Minimum Relevant Experience (Years)",
        min_value=0,
        max_value=50,
        value=3,
        step=1
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
        height=150,
        placeholder=(
            "Describe the responsibilities, qualifications "
            "and technical requirements."
        )
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
    "Upload multiple CVs",
    type=[
        "pdf",
        "docx"
    ],
    accept_multiple_files=True
)


# ============================================================
# ANALYSIS
# ============================================================

st.header(
    "3️⃣ Analyze Candidates"
)

analyze_button = st.button(
    "🚀 Analyze CVs",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not api_key:

        st.error(
            "Please enter your Groq API key."
        )

    elif not uploaded_files:

        st.error(
            "Please upload at least one CV."
        )

    elif not job_title:

        st.error(
            "Please enter the Job Title."
        )

    else:

        try:

            client = Groq(
                api_key=api_key
            )

            job_info = {

                "job_title":
                    job_title,

                "min_experience":
                    min_experience,

                "job_description":
                    job_description,

                "keywords":
                    required_keywords
            }

            st.session_state.job_info = (
                job_info
            )

            # Reset previous analysis
            st.session_state.candidate_data = []

            st.session_state.analysis_errors = []

            progress = st.progress(
                0
            )

            status = st.empty()

            total_files = len(
                uploaded_files
            )

            for index, uploaded_file in enumerate(
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

                        raise Exception(
                            "No readable text found in CV."
                        )

                    candidate = analyze_cv(
                        client,
                        model_name,
                        cv_text,
                        job_info
                    )

                    # Normalize AI output fields
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
                        "required_keywords_found"
                    ] = safe_list(
                        candidate.get(
                            "required_keywords_found",
                            []
                        )
                    )

                    candidate[
                        "required_keywords_missing"
                    ] = safe_list(
                        candidate.get(
                            "required_keywords_missing",
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

                    candidate["cv_file"] = (
                        uploaded_file.name
                    )

                    candidate["cv_text"] = (
                        cv_text
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
                    (index + 1)
                    / total_files
                )

            st.session_state.processed = True

            status.success(
                "CV analysis completed."
            )

        except Exception as e:

            st.error(
                f"Could not initialize Groq: {str(e)}"
            )


# ============================================================
# ANALYSIS ERRORS
# ============================================================

if st.session_state.analysis_errors:

    st.warning(
        "Some CVs could not be processed."
    )

    for error in (
        st.session_state.analysis_errors
    ):

        st.error(
            f"{error['file']}: {error['error']}"
        )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.processed
    and st.session_state.candidate_data
    and st.session_state.job_info
):

    results = recalculate_results()

    st.markdown("---")

    st.header(
        "📊 Candidate Ranking"
    )

    # ========================================================
    # CURRENT WEIGHTS
    # ========================================================

    st.subheader(
        "Current Scoring Weights"
    )

    weight_df = pd.DataFrame(
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

    st.dataframe(
        weight_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Changing the weights recalculates scores locally. "
        "The CVs are not sent to the AI again."
    )


    # ========================================================
    # DASHBOARD METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

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

    col1.metric(
        "Candidates Analyzed",
        len(results)
    )

    col2.metric(
        "Average Score",
        f"{average_score:.1f}%"
    )

    col3.metric(
        "Highest Score",
        f"{highest_score:.1f}%"
    )

    col4.metric(
        "Required Keywords",
        len(required_keywords)
    )


    # ========================================================
    # RANKING TABLE
    # ========================================================

    ranking_data = []

    for candidate in results:

        ranking_data.append(
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

                "CV File":
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

                "Technical Skills":
                    candidate["technical_score"],

                "Education":
                    candidate["education_score"],

                "Job Requirements":
                    candidate["requirement_score"],

                "Certifications":
                    candidate["certification_score"],

                "Matched Keywords":
                    safe_join(
                        candidate.get(
                            "matched_keywords",
                            []
                        )
                    ),

                "Missing Keywords":
                    safe_join(
                        candidate.get(
                            "missing_keywords",
                            []
                        )
                    )
            }
        )

    ranking_df = pd.DataFrame(
        ranking_data
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SCORE COMPARISON
    # ========================================================

    st.subheader(
        "📈 Candidate Score Comparison"
    )

    chart_df = ranking_df[
        [
            "Candidate",
            "Overall Score"
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "Candidate"
    )

    st.bar_chart(
        chart_df
    )


    # ========================================================
    # WEIGHT CONTRIBUTION
    # ========================================================

    st.subheader(
        "⚖️ Weight Contribution Analysis"
    )

    st.caption(
        "The contribution values show how much each factor "
        "adds to the final score."
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

                "Experience Contribution":
                    candidate[
                        "experience_contribution"
                    ],

                "Technical Contribution":
                    candidate[
                        "technical_contribution"
                    ],

                "Education Contribution":
                    candidate[
                        "education_contribution"
                    ],

                "Requirements Contribution":
                    candidate[
                        "requirements_contribution"
                    ],

                "Certification Contribution":
                    candidate[
                        "certification_contribution"
                    ],

                "Overall Score":
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
    # DETAILED ANALYSIS
    # ========================================================

    st.markdown("---")

    st.header(
        "🔎 Detailed Candidate Analysis"
    )

    candidate_names = []

    for candidate in results:

        candidate_names.append(
            f"{candidate['rank']}. "
            f"{safe_string(candidate.get('candidate_name', 'Not specified'))}"
        )

    selected_candidate_name = st.selectbox(
        "Select Candidate",
        candidate_names
    )

    selected_index = (
        candidate_names.index(
            selected_candidate_name
        )
    )

    candidate = results[
        selected_index
    ]


    # ========================================================
    # CANDIDATE SUMMARY
    # ========================================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overall Score",
        f"{candidate['overall_score']:.1f}%"
    )

    col2.metric(
        "Rank",
        f"#{candidate['rank']}"
    )

    col3.metric(
        "Relevant Experience",
        f"{safe_number(candidate.get('years_relevant_experience', 0)):.1f} years"
    )

    st.write(
        f"**CV File:** {safe_string(candidate.get('cv_file', ''))}"
    )


    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.subheader(
        "Score Breakdown"
    )

    score_breakdown = pd.DataFrame(
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
        score_breakdown,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # KEYWORD MATCHING
    # ========================================================

    st.subheader(
        "🔑 Keyword Matching"
    )

    matched = safe_list(
        candidate.get(
            "matched_keywords",
            []
        )
    )

    missing = safe_list(
        candidate.get(
            "missing_keywords",
            []
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### ✅ Matched Keywords"
        )

        if matched:

            for keyword in matched:

                st.success(
                    safe_string(keyword)
                )

        else:

            st.write(
                "No required keywords matched."
            )

    with col2:

        st.markdown(
            "### ❌ Missing Keywords"
        )

        if missing:

            for keyword in missing:

                st.error(
                    safe_string(keyword)
                )

        else:

            st.success(
                "All required keywords matched."
            )


    # ========================================================
    # CANDIDATE INFORMATION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

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

                st.write(
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "Not specified"
            )


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
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "Not specified"
            )


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
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "Not specified"
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
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "Not specified"
            )


    with col2:

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
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "Not specified"
            )


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
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "Not specified"
            )


    # ========================================================
    # STRENGTHS AND GAPS
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
                "No specific strengths identified."
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
        "📥 Export Results"
    )

    # --------------------------------------------------------
    # Ranking sheet
    # --------------------------------------------------------

    export_ranking = ranking_df.copy()

    # --------------------------------------------------------
    # Candidate details
    # --------------------------------------------------------

    export_details = []

    for candidate in results:

        export_details.append(
            {
                "Rank":
                    candidate.get(
                        "rank",
                        ""
                    ),

                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            "Not specified"
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
                    candidate.get(
                        "overall_score",
                        0
                    ),

                "Relevant Experience (Years)":
                    safe_number(
                        candidate.get(
                            "years_relevant_experience",
                            0
                        )
                    ),

                "Experience Score":
                    candidate.get(
                        "experience_score",
                        0
                    ),

                "Technical Score":
                    candidate.get(
                        "technical_score",
                        0
                    ),

                "Education Score":
                    candidate.get(
                        "education_score",
                        0
                    ),

                "Requirement Score":
                    candidate.get(
                        "requirement_score",
                        0
                    ),

                "Certification Score":
                    candidate.get(
                        "certification_score",
                        0
                    ),

                "Experience Contribution":
                    candidate.get(
                        "experience_contribution",
                        0
                    ),

                "Technical Contribution":
                    candidate.get(
                        "technical_contribution",
                        0
                    ),

                "Education Contribution":
                    candidate.get(
                        "education_contribution",
                        0
                    ),

                "Requirements Contribution":
                    candidate.get(
                        "requirements_contribution",
                        0
                    ),

                "Certification Contribution":
                    candidate.get(
                        "certification_contribution",
                        0
                    ),

                "Matched Keywords":
                    safe_join(
                        candidate.get(
                            "matched_keywords",
                            []
                        )
                    ),

                "Missing Keywords":
                    safe_join(
                        candidate.get(
                            "missing_keywords",
                            []
                        )
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

                "Relevant Experience Evidence":
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
        export_details
    )


    # --------------------------------------------------------
    # Scoring weights sheet
    # --------------------------------------------------------

    weight_export = pd.DataFrame(
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


    # --------------------------------------------------------
    # Keyword sheet
    # --------------------------------------------------------

    keyword_export = []

    for candidate in results:

        matched = safe_list(
            candidate.get(
                "matched_keywords",
                []
            )
        )

        missing = safe_list(
            candidate.get(
                "missing_keywords",
                []
            )
        )

        keyword_export.append(
            {
                "Rank":
                    candidate.get(
                        "rank",
                        ""
                    ),

                "Candidate":
                    safe_string(
                        candidate.get(
                            "candidate_name",
                            "Not specified"
                        )
                    ),

                "Matched Keywords":
                    safe_join(
                        matched
                    ),

                "Number Matched":
                    len(matched),

                "Missing Keywords":
                    safe_join(
                        missing
                    ),

                "Number Missing":
                    len(missing),

                "Total Required":
                    len(required_keywords),

                "Keyword Match %":
                    candidate.get(
                        "requirement_score",
                        0
                    )
            }
        )

    keyword_df = pd.DataFrame(
        keyword_export
    )


    # --------------------------------------------------------
    # Create Excel workbook
    # --------------------------------------------------------

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

        weight_export.to_excel(
            writer,
            sheet_name="Scoring Weights",
            index=False
        )

        keyword_df.to_excel(
            writer,
            sheet_name="Keyword Matching",
            index=False
        )


    excel_buffer.seek(0)


    # --------------------------------------------------------
    # Download button
    # --------------------------------------------------------

    st.download_button(
        label="📊 Download Excel Ranking",
        data=excel_buffer.getvalue(),
        file_name="HireTech_AI_Candidate_Ranking.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "HireTech AI | AI-assisted recruitment screening "
    "and transparent candidate scoring"
)
