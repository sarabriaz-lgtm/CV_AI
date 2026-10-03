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
    page_icon="👷",
    layout="wide"
)

st.title("👷 HireTech AI")
st.caption(
    "AI-powered recruitment and candidate ranking system for Civil Engineering roles"
)


# ============================================================
# DEFAULT JOB DESCRIPTION
# ============================================================

DEFAULT_JOB_DESCRIPTION = """
We are looking for a qualified Civil Engineer to join our engineering team and
contribute to the planning, design, analysis, and supervision of civil
engineering projects.

The candidate should have a relevant degree in Civil Engineering or a closely
related field and relevant professional experience in civil engineering,
construction, infrastructure, water resources, transportation, structural or
environmental engineering projects.

Key responsibilities include reviewing engineering drawings, technical
reports, design calculations, BOQs, quantity estimates, specifications and
project documentation; assisting with project planning and coordination;
conducting engineering analysis; monitoring project progress; coordinating
with consultants, contractors and multidisciplinary teams; and preparing
clear technical reports and engineering documentation.

Relevant technical knowledge and experience with engineering software such as
AutoCAD, Civil 3D, ArcGIS, QGIS, HEC-RAS, HEC-HMS, SWMM, Primavera P6,
MS Project, Microsoft Excel or similar engineering tools will be considered
valuable.

The candidate should demonstrate strong analytical, problem-solving,
communication, report-writing, teamwork and project coordination skills.
Experience with site supervision, construction management, quantity
estimation, hydrology, hydraulics, drainage, roads, infrastructure,
surveying or related civil engineering activities is an advantage.

The ideal candidate should be able to understand technical requirements,
evaluate engineering information, work effectively with multidisciplinary
teams, and deliver accurate and professional engineering work within project
requirements and timelines.
"""


# ============================================================
# SESSION STATE
# ============================================================

if "candidates" not in st.session_state:
    st.session_state.candidates = []

if "analysis_errors" not in st.session_state:
    st.session_state.analysis_errors = []

if "job_description" not in st.session_state:
    st.session_state.job_description = DEFAULT_JOB_DESCRIPTION

if "job_title" not in st.session_state:
    st.session_state.job_title = "Civil Engineer"

if "required_keywords" not in st.session_state:
    st.session_state.required_keywords = ""

if "min_experience" not in st.session_state:
    st.session_state.min_experience = 0

if "weight_experience" not in st.session_state:
    st.session_state.weight_experience = 30

if "weight_technical" not in st.session_state:
    st.session_state.weight_technical = 25

if "weight_education" not in st.session_state:
    st.session_state.weight_education = 15

if "weight_requirements" not in st.session_state:
    st.session_state.weight_requirements = 20

if "weight_certification" not in st.session_state:
    st.session_state.weight_certification = 10


# ============================================================
# GROQ API KEY FROM STREAMLIT SECRETS
# ============================================================

def get_groq_api_key():
    try:
        return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        return ""


api_key = get_groq_api_key()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    if api_key:
        st.success("Groq API loaded from Secrets")
    else:
        st.error("GROQ_API_KEY is missing")

    model = st.selectbox(
        "Groq Model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b"
        ],
        index=0
    )

    st.divider()

    st.subheader("⚖️ Scoring Weights")

    weight_experience = st.slider(
        "Experience",
        0,
        100,
        st.session_state.weight_experience,
        key="weight_experience"
    )

    weight_technical = st.slider(
        "Technical Skills",
        0,
        100,
        st.session_state.weight_technical,
        key="weight_technical"
    )

    weight_education = st.slider(
        "Education",
        0,
        100,
        st.session_state.weight_education,
        key="weight_education"
    )

    weight_requirements = st.slider(
        "Job Requirements",
        0,
        100,
        st.session_state.weight_requirements,
        key="weight_requirements"
    )

    weight_certification = st.slider(
        "Certifications",
        0,
        100,
        st.session_state.weight_certification,
        key="weight_certification"
    )

    total_weight = (
        weight_experience
        + weight_technical
        + weight_education
        + weight_requirements
        + weight_certification
    )

    st.metric("Total Weight", f"{total_weight}%")

    if total_weight != 100:
        st.warning(
            "The weights do not total 100%. Scores will be normalized automatically."
        )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_string(value):
    """
    Converts dictionaries, lists and other values into readable strings.
    Prevents errors when displaying/exporting AI-generated data.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        parts = []

        for key, val in value.items():
            if val is not None and str(val).strip():
                parts.append(f"{key}: {val}")

        return ", ".join(parts)

    if isinstance(value, list):
        return ", ".join(safe_string(x) for x in value if safe_string(x))

    return str(value).strip()


def safe_list(value):
    """
    Converts AI output into a clean list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        result = []

        for item in value:
            text = safe_string(item)

            if text:
                result.append(text)

        return result

    if isinstance(value, dict):
        return [safe_string(value)]

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    return [str(value)]


def safe_join(value, separator=", "):
    return separator.join(safe_list(value))


# ============================================================
# FILE TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    uploaded_file.seek(0)

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        try:
            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception:
            continue

    return "\n".join(pages)


def extract_docx_text(uploaded_file):

    uploaded_file.seek(0)

    document = Document(uploaded_file)

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    # Also extract table text
    for table in document.tables:

        for row in table.rows:

            row_text = []

            for cell in row.cells:

                if cell.text.strip():
                    row_text.append(cell.text.strip())

            if row_text:
                paragraphs.append(" | ".join(row_text))

    return "\n".join(paragraphs)


def extract_text(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    if file_name.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    return ""


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_keyword(keyword):

    keyword = normalize_text(keyword)

    keyword = re.sub(r"\s+", " ", keyword)

    return keyword.strip()


# ============================================================
# KEYWORD MATCHING
# ============================================================

def find_keywords_in_cv(cv_text, required_keywords):

    normalized_cv = normalize_text(cv_text)

    matched = []
    missing = []

    for keyword in required_keywords:

        keyword = normalize_keyword(keyword)

        if not keyword:
            continue

        # Direct text matching against the complete CV
        if keyword in normalized_cv:
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing


def keyword_score(matched, required_keywords):

    total = len(required_keywords)

    if total == 0:
        return 0

    return round((len(matched) / total) * 100, 2)


# ============================================================
# JSON CLEANING
# ============================================================

def clean_json_response(response_text):

    if not response_text:
        return None

    text = response_text.strip()

    # Remove markdown code fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    text = text.strip()

    # Locate JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1:

        text = text[first_brace:last_brace + 1]

    # Remove problematic control characters
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        # Attempt small repairs
        repaired = text

        repaired = repaired.replace("\n", " ")
        repaired = repaired.replace("\t", " ")

        try:
            return json.loads(repaired)
        except Exception:
            return None


# ============================================================
# AI CV ANALYSIS
# ============================================================

def analyze_cv(cv_text, job_description, model_name):

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is not configured in Streamlit Secrets."
        )

    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert HR recruitment assistant specializing in Civil Engineering.

Analyze the candidate CV against the job description.

Return ONLY valid JSON.

Do not add markdown.
Do not add explanations outside JSON.

JOB DESCRIPTION:
{job_description}

CANDIDATE CV:
{cv_text}

Return this exact JSON structure:

{{
    "candidate_name": "",
    "education": [],
    "years_relevant_experience": 0,
    "previous_roles": [],
    "technical_skills": [],
    "software_tools": [],
    "certifications": [],
    "relevant_experience_evidence": [],
    "strengths": [],
    "potential_gaps": []
}}

Rules:

1. candidate_name:
   Extract the candidate's full name.

2. education:
   Return readable education entries.
   Example:
   "M.Sc. Civil Engineering - University of Central Punjab, Lahore - 2023"

3. years_relevant_experience:
   Estimate relevant professional Civil Engineering experience.
   Return a number.

4. previous_roles:
   List previous job titles and organizations where available.

5. technical_skills:
   List engineering and technical skills.

6. software_tools:
   List software such as AutoCAD, Civil 3D, ArcGIS, QGIS,
   HEC-RAS, HEC-HMS, Primavera, MS Project, Excel etc.

7. certifications:
   List professional certifications.

8. relevant_experience_evidence:
   Provide short evidence from the CV showing relevant experience.

9. strengths:
   List important candidate strengths for the job.

10. potential_gaps:
   List missing or weaker areas relevant to the job.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a professional HR recruitment and CV analysis assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=4000
    )

    response_text = response.choices[0].message.content

    data = clean_json_response(response_text)

    if data is None:

        raise ValueError(
            "The AI returned an invalid JSON response. Please try again."
        )

    return data


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def experience_score(years, minimum_required):

    try:
        years = float(years)
        minimum_required = float(minimum_required)
    except Exception:
        return 0

    if minimum_required <= 0:

        if years <= 0:
            return 0

        return min(100, years * 20)

    score = (years / minimum_required) * 100

    return round(min(100, score), 2)


def education_score(education):

    education_text = normalize_text(safe_join(education))

    if not education_text:
        return 0

    score = 0

    if "civil engineering" in education_text:
        score += 100

    elif "engineering" in education_text:
        score += 80

    elif "construction" in education_text:
        score += 70

    elif "technology" in education_text:
        score += 60

    else:
        score += 40

    return min(score, 100)


def certification_score(certifications):

    certifications = safe_list(certifications)

    if not certifications:
        return 0

    return 100


def calculate_scores(
    candidate,
    required_keywords,
    min_experience,
    weights
):

    matched = candidate.get("matched_keywords", [])

    years = candidate.get("years_relevant_experience", 0)

    education = candidate.get("education", [])

    certifications = candidate.get("certifications", [])

    requirement_score = keyword_score(
        matched,
        required_keywords
    )

    technical_score = requirement_score

    exp_score = experience_score(
        years,
        min_experience
    )

    edu_score = education_score(
        education
    )

    cert_score = certification_score(
        certifications
    )

    total_weight = sum(weights.values())

    if total_weight <= 0:
        total_weight = 100

    final_score = (

        exp_score * weights["experience"]

        + technical_score * weights["technical"]

        + edu_score * weights["education"]

        + requirement_score * weights["requirements"]

        + cert_score * weights["certification"]

    ) / total_weight

    candidate["experience_score"] = round(exp_score, 2)
    candidate["technical_score"] = round(technical_score, 2)
    candidate["education_score"] = round(edu_score, 2)
    candidate["requirements_score"] = round(requirement_score, 2)
    candidate["certification_score"] = round(cert_score, 2)
    candidate["final_score"] = round(final_score, 2)

    return candidate


def recalculate_results():

    required_keywords = [
        normalize_keyword(x)
        for x in st.session_state.required_keywords.split(",")
        if normalize_keyword(x)
    ]

    weights = {
        "experience": st.session_state.weight_experience,
        "technical": st.session_state.weight_technical,
        "education": st.session_state.weight_education,
        "requirements": st.session_state.weight_requirements,
        "certification": st.session_state.weight_certification
    }

    for candidate in st.session_state.candidates:

        matched, missing = find_keywords_in_cv(
            candidate.get("cv_text", ""),
            required_keywords
        )

        candidate["matched_keywords"] = matched
        candidate["missing_keywords"] = missing

        calculate_scores(
            candidate,
            required_keywords,
            st.session_state.min_experience,
            weights
        )

    st.session_state.candidates.sort(
        key=lambda x: x.get("final_score", 0),
        reverse=True
    )


# ============================================================
# JOB REQUIREMENTS
# ============================================================

st.header("📋 Job Requirements")

col1, col2 = st.columns(2)

with col1:

    st.text_input(
        "Job Title",
        key="job_title"
    )

    st.number_input(
        "Minimum Relevant Experience (Years)",
        min_value=0,
        max_value=50,
        step=1,
        key="min_experience"
    )

with col2:

    st.text_input(
        "Required Keywords",
        key="required_keywords",
        placeholder="AutoCAD, Civil 3D, Hydrology, HEC-RAS, BOQ..."
    )

    st.caption(
        "Separate keywords using commas. Matching is performed directly against the complete CV text."
    )


st.subheader("📝 Job Description")

st.text_area(
    "Job Description",
    key="job_description",
    height=300,
    help="The description is pre-filled for a Civil Engineering role. You can edit it whenever required."
)

st.caption(
    "The job description is pre-filled by default, but recruiters can modify it for any vacancy."
)


# ============================================================
# CV UPLOAD
# ============================================================

st.header("📄 Upload Candidate CVs")

uploaded_files = st.file_uploader(
    "Upload CVs in PDF or DOCX format",
    type=["pdf", "docx"],
    accept_multiple_files=True
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🚀 Analyze & Rank Candidates",
    type="primary",
    use_container_width=True
):

    st.session_state.candidates = []
    st.session_state.analysis_errors = []

    if not api_key:

        st.error(
            "GROQ_API_KEY is missing. Please add it to Streamlit Secrets."
        )

    elif not uploaded_files:

        st.warning(
            "Please upload at least one CV."
        )

    elif not st.session_state.job_description.strip():

        st.warning(
            "Please provide a job description."
        )

    else:

        progress = st.progress(0)

        total_files = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):

            try:

                cv_text = extract_text(uploaded_file)

                if not cv_text.strip():

                    raise ValueError(
                        "Could not extract readable text from this CV."
                    )

                with st.spinner(
                    f"Analyzing {uploaded_file.name}..."
                ):

                    analysis = analyze_cv(
                        cv_text,
                        st.session_state.job_description,
                        model
                    )

                candidate = {
                    "file_name": uploaded_file.name,
                    "cv_text": cv_text,
                    "candidate_name": safe_string(
                        analysis.get("candidate_name", "")
                    ),
                    "education": safe_list(
                        analysis.get("education", [])
                    ),
                    "years_relevant_experience": analysis.get(
                        "years_relevant_experience",
                        0
                    ),
                    "previous_roles": safe_list(
                        analysis.get("previous_roles", [])
                    ),
                    "technical_skills": safe_list(
                        analysis.get("technical_skills", [])
                    ),
                    "software_tools": safe_list(
                        analysis.get("software_tools", [])
                    ),
                    "certifications": safe_list(
                        analysis.get("certifications", [])
                    ),
                    "relevant_experience_evidence": safe_list(
                        analysis.get(
                            "relevant_experience_evidence",
                            []
                        )
                    ),
                    "strengths": safe_list(
                        analysis.get("strengths", [])
                    ),
                    "potential_gaps": safe_list(
                        analysis.get("potential_gaps", [])
                    )
                }

                st.session_state.candidates.append(candidate)

            except Exception as e:

                st.session_state.analysis_errors.append(
                    f"{uploaded_file.name}: {str(e)}"
                )

            progress.progress(
                (index + 1) / total_files
            )

        recalculate_results()

        if st.session_state.candidates:

            st.success(
                f"Successfully analyzed {len(st.session_state.candidates)} candidate(s)."
            )


# ============================================================
# ERRORS
# ============================================================

if st.session_state.analysis_errors:

    st.subheader("⚠️ Analysis Errors")

    for error in st.session_state.analysis_errors:

        st.error(error)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.candidates:

    st.header("🏆 Candidate Ranking")

    ranking_data = []

    for index, candidate in enumerate(
        st.session_state.candidates,
        start=1
    ):

        ranking_data.append({

            "Rank": index,

            "Candidate":
                candidate.get(
                    "candidate_name",
                    "Unknown"
                ),

            "Experience":
                candidate.get(
                    "years_relevant_experience",
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

            "Requirements Score":
                candidate.get(
                    "requirements_score",
                    0
                ),

            "Certification Score":
                candidate.get(
                    "certification_score",
                    0
                ),

            "Final Score":
                candidate.get(
                    "final_score",
                    0
                )
        })

    ranking_df = pd.DataFrame(
        ranking_data
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RE-RANK AFTER WEIGHT CHANGES
    # ========================================================

    if st.button(
        "🔄 Recalculate Scores Using Current Weights",
        use_container_width=True
    ):

        recalculate_results()

        st.success(
            "Scores and ranking recalculated using the current weightage."
        )

        st.rerun()


    # ========================================================
    # KEYWORD MATCHING
    # ========================================================

    st.header("🔎 Keyword Matching")

    required_keywords = [
        normalize_keyword(x)
        for x in st.session_state.required_keywords.split(",")
        if normalize_keyword(x)
    ]

    if required_keywords:

        keyword_rows = []

        for candidate in st.session_state.candidates:

            matched = candidate.get(
                "matched_keywords",
                []
            )

            missing = candidate.get(
                "missing_keywords",
                []
            )

            keyword_rows.append({

                "Candidate":
                    candidate.get(
                        "candidate_name",
                        "Unknown"
                    ),

                "Matched Keywords":
                    ", ".join(matched),

                "Missing Keywords":
                    ", ".join(missing),

                "Keyword Match %":
                    keyword_score(
                        matched,
                        required_keywords
                    )
            })

        keyword_df = pd.DataFrame(
            keyword_rows
        )

        st.dataframe(
            keyword_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Enter required keywords above to see matched and missing keywords."
        )


    # ========================================================
    # SCORE CHART
    # ========================================================

    st.header("📊 Candidate Scores")

    chart_df = ranking_df[
        ["Candidate", "Final Score"]
    ].set_index("Candidate")

    st.bar_chart(chart_df)


    # ========================================================
    # SCORE WEIGHT CONTRIBUTION
    # ========================================================

    st.header("⚖️ Score Contribution")

    selected_candidate_name = st.selectbox(
        "Select Candidate",
        [
            candidate.get(
                "candidate_name",
                "Unknown"
            )
            for candidate in st.session_state.candidates
        ]
    )

    selected_candidate = next(
        (
            candidate
            for candidate in st.session_state.candidates
            if candidate.get(
                "candidate_name",
                "Unknown"
            ) == selected_candidate_name
        ),
        None
    )

    if selected_candidate:

        contribution_data = {

            "Experience":
                selected_candidate.get(
                    "experience_score",
                    0
                ) * st.session_state.weight_experience / 100,

            "Technical":
                selected_candidate.get(
                    "technical_score",
                    0
                ) * st.session_state.weight_technical / 100,

            "Education":
                selected_candidate.get(
                    "education_score",
                    0
                ) * st.session_state.weight_education / 100,

            "Requirements":
                selected_candidate.get(
                    "requirements_score",
                    0
                ) * st.session_state.weight_requirements / 100,

            "Certification":
                selected_candidate.get(
                    "certification_score",
                    0
                ) * st.session_state.weight_certification / 100
        }

        contribution_df = pd.DataFrame(
            {
                "Category": list(
                    contribution_data.keys()
                ),
                "Contribution": list(
                    contribution_data.values()
                )
            }
        )

        st.dataframe(
            contribution_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # DETAILED CANDIDATE VIEW
    # ========================================================

    st.header("👤 Candidate Details")

    candidate_options = [
        candidate.get(
            "candidate_name",
            "Unknown"
        )
        for candidate in st.session_state.candidates
    ]

    selected_detail_name = st.selectbox(
        "View Candidate",
        candidate_options,
        key="candidate_detail_selector"
    )

    detail_candidate = next(
        (
            candidate
            for candidate in st.session_state.candidates
            if candidate.get(
                "candidate_name",
                "Unknown"
            ) == selected_detail_name
        ),
        None
    )

    if detail_candidate:

        st.subheader(
            detail_candidate.get(
                "candidate_name",
                "Unknown"
            )
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Final Score",
                f"{detail_candidate.get('final_score', 0):.2f}%"
            )

        with col2:

            st.metric(
                "Relevant Experience",
                f"{detail_candidate.get('years_relevant_experience', 0)} years"
            )

        with col3:

            st.metric(
                "Keyword Match",
                f"{keyword_score(
                    detail_candidate.get('matched_keywords', []),
                    required_keywords
                ):.2f}%"
            )


        # ----------------------------------------------------
        # MATCHED / MISSING KEYWORDS
        # ----------------------------------------------------

        st.subheader("🔎 Keyword Analysis")

        matched_keywords = detail_candidate.get(
            "matched_keywords",
            []
        )

        missing_keywords = detail_candidate.get(
            "missing_keywords",
            []
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### ✅ Matched Keywords")

            if matched_keywords:

                for keyword in matched_keywords:

                    st.success(keyword)

            else:

                st.info("No required keywords matched.")


        with col2:

            st.markdown("### ❌ Missing Keywords")

            if missing_keywords:

                for keyword in missing_keywords:

                    st.error(keyword)

            else:

                st.success(
                    "All required keywords were found."
                )


        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        st.subheader("🎓 Education")

        education = detail_candidate.get(
            "education",
            []
        )

        if education:

            for item in education:

                st.write(
                    f"• {safe_string(item)}"
                )

        else:

            st.write("No education information found.")


        # ----------------------------------------------------
        # PREVIOUS ROLES
        # ----------------------------------------------------

        st.subheader("💼 Previous Roles")

        roles = detail_candidate.get(
            "previous_roles",
            []
        )

        if roles:

            for role in roles:

                st.write(
                    f"• {safe_string(role)}"
                )

        else:

            st.write("No previous roles found.")


        # ----------------------------------------------------
        # TECHNICAL SKILLS
        # ----------------------------------------------------

        st.subheader("🛠️ Technical Skills")

        skills = detail_candidate.get(
            "technical_skills",
            []
        )

        if skills:

            st.write(
                ", ".join(
                    safe_list(skills)
                )
            )

        else:

            st.write(
                "No technical skills identified."
            )


        # ----------------------------------------------------
        # SOFTWARE
        # ----------------------------------------------------

        st.subheader("💻 Software & Tools")

        software = detail_candidate.get(
            "software_tools",
            []
        )

        if software:

            st.write(
                ", ".join(
                    safe_list(software)
                )
            )

        else:

            st.write(
                "No software tools identified."
            )


        # ----------------------------------------------------
        # CERTIFICATIONS
        # ----------------------------------------------------

        st.subheader("📜 Certifications")

        certifications = detail_candidate.get(
            "certifications",
            []
        )

        if certifications:

            for certification in certifications:

                st.write(
                    f"• {safe_string(certification)}"
                )

        else:

            st.write(
                "No certifications identified."
            )


        # ----------------------------------------------------
        # EXPERIENCE EVIDENCE
        # ----------------------------------------------------

        st.subheader("📌 Relevant Experience Evidence")

        evidence = detail_candidate.get(
            "relevant_experience_evidence",
            []
        )

        if evidence:

            for item in evidence:

                st.write(
                    f"• {safe_string(item)}"
                )

        else:

            st.write(
                "No specific evidence identified."
            )


        # ----------------------------------------------------
        # STRENGTHS
        # ----------------------------------------------------

        st.subheader("💪 Strengths")

        strengths = detail_candidate.get(
            "strengths",
            []
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


        # ----------------------------------------------------
        # POTENTIAL GAPS
        # ----------------------------------------------------

        st.subheader("⚠️ Potential Gaps")

        gaps = detail_candidate.get(
            "potential_gaps",
            []
        )

        if gaps:

            for item in gaps:

                st.warning(
                    safe_string(item)
                )

        else:

            st.success(
                "No significant gaps identified."
            )


# ============================================================
# EXCEL EXPORT
# ============================================================

if st.session_state.candidates:

    st.header("📥 Export Results")

    ranking_export = []

    details_export = []

    keyword_export = []

    for index, candidate in enumerate(
        st.session_state.candidates,
        start=1
    ):

        ranking_export.append({

            "Rank":
                index,

            "Candidate Name":
                candidate.get(
                    "candidate_name",
                    ""
                ),

            "File":
                candidate.get(
                    "file_name",
                    ""
                ),

            "Relevant Experience (Years)":
                candidate.get(
                    "years_relevant_experience",
                    0
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

            "Requirements Score":
                candidate.get(
                    "requirements_score",
                    0
                ),

            "Certification Score":
                candidate.get(
                    "certification_score",
                    0
                ),

            "Final Score":
                candidate.get(
                    "final_score",
                    0
                )
        })


        details_export.append({

            "Candidate Name":
                candidate.get(
                    "candidate_name",
                    ""
                ),

            "Education":
                safe_join(
                    candidate.get(
                        "education",
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
        })


        keyword_export.append({

            "Candidate Name":
                candidate.get(
                    "candidate_name",
                    ""
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

            "Keyword Match %":
                keyword_score(
                    candidate.get(
                        "matched_keywords",
                        []
                    ),
                    required_keywords
                )
        })


    ranking_export_df = pd.DataFrame(
        ranking_export
    )

    details_export_df = pd.DataFrame(
        details_export
    )

    keyword_export_df = pd.DataFrame(
        keyword_export
    )

    weights_export_df = pd.DataFrame([
        {
            "Scoring Category": "Experience",
            "Weight": st.session_state.weight_experience
        },
        {
            "Scoring Category": "Technical Skills",
            "Weight": st.session_state.weight_technical
        },
        {
            "Scoring Category": "Education",
            "Weight": st.session_state.weight_education
        },
        {
            "Scoring Category": "Job Requirements",
            "Weight": st.session_state.weight_requirements
        },
        {
            "Scoring Category": "Certifications",
            "Weight": st.session_state.weight_certification
        }
    ])


    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        ranking_export_df.to_excel(
            writer,
            sheet_name="Candidate Ranking",
            index=False
        )

        details_export_df.to_excel(
            writer,
            sheet_name="Candidate Details",
            index=False
        )

        weights_export_df.to_excel(
            writer,
            sheet_name="Scoring Weights",
            index=False
        )

        keyword_export_df.to_excel(
            writer,
            sheet_name="Keyword Matching",
            index=False
        )


    output.seek(0)

    st.download_button(
        label="📥 Download Excel Report",
        data=output,
        file_name="HireTech_AI_Candidate_Ranking.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "HireTech AI • AI-assisted recruitment and candidate analysis"
)
