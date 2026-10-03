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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_file):
    """Extract text from PDF."""
    try:
        reader = PdfReader(uploaded_file)
        text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return "\n".join(text)

    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")


def extract_docx_text(uploaded_file):
    """Extract text from DOCX including tables."""
    try:
        doc = Document(uploaded_file)

        text = []

        # Paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)

        # Tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []

                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())

                if row_text:
                    text.append(" | ".join(row_text))

        return "\n".join(text)

    except Exception as e:
        raise Exception(f"DOCX extraction failed: {str(e)}")


def extract_text(uploaded_file):
    """Extract text according to file type."""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_pdf_text(uploaded_file)

    elif file_name.endswith(".docx"):
        return extract_docx_text(uploaded_file)

    else:
        raise Exception("Unsupported file format")


def clean_json_response(response_text):
    """
    Clean AI response and extract JSON object.
    """

    if not response_text:
        raise ValueError("AI returned an empty response.")

    response_text = response_text.strip()

    # Remove markdown code fences
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

    # Find JSON object
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON object found in AI response.")

    return response_text[start:end + 1]


def repair_json_with_ai(client, model_name, broken_response):
    """
    Ask Groq to repair malformed JSON.
    """

    repair_prompt = f"""
You are a JSON repair assistant.

The following text is supposed to be a JSON object but contains
syntax errors.

Repair it and return ONLY valid JSON.

Do not add explanations.
Do not use Markdown.
Do not add code fences.
Do not change the meaning of the data.

BROKEN JSON:

{broken_response}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You repair malformed JSON and return only valid JSON."
            },
            {
                "role": "user",
                "content": repair_prompt
            }
        ],
        temperature=0
    )

    repaired = response.choices[0].message.content

    cleaned = clean_json_response(repaired)

    return json.loads(cleaned)


def parse_ai_json(client, model_name, response_text):
    """
    Parse AI JSON.
    If malformed, automatically request JSON repair.
    """

    cleaned = clean_json_response(response_text)

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        return repair_json_with_ai(
            client,
            model_name,
            cleaned
        )


def analyze_cv(client, model_name, cv_text, job_info):
    """
    Analyze one CV using Groq.
    """

    keywords = job_info["keywords"]

    prompt = f"""
You are an AI recruitment screening assistant.

Analyze the candidate CV against the job requirements.

IMPORTANT RULES:

1. Use ONLY information present in the CV.
2. Do NOT invent qualifications, experience, skills, or certifications.
3. years_relevant_experience must be a number.
4. Identify only experience relevant to the specified job.
5. Match required keywords based on the CV.
6. Return ONLY ONE valid JSON object.
7. Do not return Markdown.
8. Do not use ```json.
9. Do not provide explanations outside the JSON.
10. Make sure every JSON field is properly separated by commas.

JOB INFORMATION

Job Title:
{job_info["job_title"]}

Minimum Relevant Experience:
{job_info["min_experience"]} years

Job Description:
{job_info["job_description"]}

Required Keywords:
{", ".join(keywords)}

REQUIRED JSON FORMAT:

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
                    "You are a professional recruitment screening assistant. "
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

    result_text = response.choices[0].message.content

    return parse_ai_json(
        client,
        model_name,
        result_text
    )


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def calculate_experience_score(years, minimum):
    """
    Experience score based on relevant years.
    """

    try:
        years = float(years)
        minimum = float(minimum)
    except:
        return 0

    if years >= minimum + 4:
        return 100

    elif years >= minimum + 2:
        return 90

    elif years >= minimum:
        return 75

    elif years >= max(0, minimum - 1):
        return 50

    elif years > 0:
        return 30

    return 0


def calculate_keyword_score(found, required):
    """
    Score based on percentage of required keywords matched.
    """

    required = [str(x).strip().lower() for x in required if str(x).strip()]
    found = [str(x).strip().lower() for x in found if str(x).strip()]

    if not required:
        return 100

    matched = set(found).intersection(set(required))

    return round(
        (len(matched) / len(set(required))) * 100,
        2
    )


def calculate_education_score(education):
    """
    Education relevance score.
    """

    if not education:
        return 0

    text = " ".join(
        [str(x) for x in education]
    ).lower()

    score = 50

    if "water resources" in text:
        score = 95

    elif "hydrology" in text:
        score = 95

    elif "hydraulic" in text:
        score = 95

    elif "civil engineering" in text:
        score = 80

    elif "environmental engineering" in text:
        score = 75

    elif "engineering" in text:
        score = 70

    if "master" in text or "msc" in text or "m.sc" in text:
        score = min(score + 5, 100)

    return score


def calculate_certification_score(certifications):
    """
    Certification score.
    """

    if not certifications:
        return 0

    valid_certifications = [
        str(x).strip()
        for x in certifications
        if str(x).strip()
        and str(x).strip().lower() not in [
            "not specified",
            "none",
            "n/a",
            "na"
        ]
    ]

    if valid_certifications:
        return 100

    return 0


def calculate_candidate_scores(candidate, job_info, weights):
    """
    Calculate all component scores and final weighted score.

    THIS FUNCTION IS LOCAL PYTHON ONLY.
    IT DOES NOT CALL GROQ.
    """

    minimum_experience = job_info["min_experience"]
    required_keywords = job_info["keywords"]

    experience_score = calculate_experience_score(
        candidate.get("years_relevant_experience", 0),
        minimum_experience
    )

    # Combine technical skills + software tools
    technical_items = (
        candidate.get("technical_skills", [])
        + candidate.get("software_tools", [])
    )

    technical_score = calculate_keyword_score(
        technical_items,
        required_keywords
    )

    education_score = calculate_education_score(
        candidate.get("education", [])
    )

    requirement_score = calculate_keyword_score(
        candidate.get("required_keywords_found", []),
        required_keywords
    )

    certification_score = calculate_certification_score(
        candidate.get("certifications", [])
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

    weighted_experience = (
        experience_score * weights["experience"]
    )

    weighted_technical = (
        technical_score * weights["technical"]
    )

    weighted_education = (
        education_score * weights["education"]
    )

    weighted_requirements = (
        requirement_score * weights["requirements"]
    )

    weighted_certification = (
        certification_score * weights["certification"]
    )

    overall_score = (
        weighted_experience
        + weighted_technical
        + weighted_education
        + weighted_requirements
        + weighted_certification
    ) / total_weight

    return {
        "experience_score": round(experience_score, 2),
        "technical_score": round(technical_score, 2),
        "education_score": round(education_score, 2),
        "requirement_score": round(requirement_score, 2),
        "certification_score": round(certification_score, 2),

        "experience_contribution": round(
            weighted_experience / total_weight,
            2
        ),

        "technical_contribution": round(
            weighted_technical / total_weight,
            2
        ),

        "education_contribution": round(
            weighted_education / total_weight,
            2
        ),

        "requirements_contribution": round(
            weighted_requirements / total_weight,
            2
        ),

        "certification_contribution": round(
            weighted_certification / total_weight,
            2
        ),

        "overall_score": round(overall_score, 2)
    }


def recalculate_results():
    """
    Recalculate ranking using stored AI results.

    IMPORTANT:
    This does NOT call Groq.
    """

    results = []

    weights = {
        "experience": st.session_state.weight_experience,
        "technical": st.session_state.weight_technical,
        "education": st.session_state.weight_education,
        "requirements": st.session_state.weight_requirements,
        "certification": st.session_state.weight_certification
    }

    for candidate in st.session_state.candidate_data:

        scores = calculate_candidate_scores(
            candidate,
            st.session_state.job_info,
            weights
        )

        result = candidate.copy()
        result.update(scores)

        results.append(result)

    results.sort(
        key=lambda x: x["overall_score"],
        reverse=True
    )

    for i, candidate in enumerate(results, start=1):
        candidate["rank"] = i

    return results


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ AI Settings")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Enter your Groq API key."
)

model_name = st.sidebar.selectbox(
    "AI Model",
    [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b"
    ]
)

st.sidebar.markdown("---")

st.sidebar.header("⚖️ Scoring Weightage")

st.sidebar.caption(
    "Change these weights to control how much each factor contributes "
    "to the final candidate score."
)

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


st.sidebar.slider(
    "Relevant Experience (%)",
    min_value=0,
    max_value=100,
    step=5,
    key="weight_experience"
)

st.sidebar.slider(
    "Technical Skills (%)",
    min_value=0,
    max_value=100,
    step=5,
    key="weight_technical"
)

st.sidebar.slider(
    "Education (%)",
    min_value=0,
    max_value=100,
    step=5,
    key="weight_education"
)

st.sidebar.slider(
    "Job Requirements (%)",
    min_value=0,
    max_value=100,
    step=5,
    key="weight_requirements"
)

st.sidebar.slider(
    "Certifications (%)",
    min_value=0,
    max_value=100,
    step=5,
    key="weight_certification"
)


total_weight_display = (
    st.session_state.weight_experience
    + st.session_state.weight_technical
    + st.session_state.weight_education
    + st.session_state.weight_requirements
    + st.session_state.weight_certification
)

st.sidebar.markdown(
    f"### Total Weight: **{total_weight_display}%**"
)

st.sidebar.info(
    "The scoring engine automatically normalizes the weights. "
    "Therefore, the total does not have to equal exactly 100%."
)


# ============================================================
# JOB REQUIREMENTS
# ============================================================

st.header("1️⃣ Job Requirements")

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
        placeholder="HEC-RAS, HEC-HMS, ArcGIS, Hydrology"
    )

    job_description = st.text_area(
        "Job Description",
        height=150,
        placeholder=(
            "Describe the responsibilities, required qualifications, "
            "technical skills and experience."
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

st.header("2️⃣ Upload Candidate CVs")

uploaded_files = st.file_uploader(
    "Upload multiple CVs",
    type=["pdf", "docx"],
    accept_multiple_files=True
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.header("3️⃣ Analyze Candidates")

analyze_button = st.button(
    "🚀 Analyze CVs",
    type="primary",
    use_container_width=True
)


if analyze_button:

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")

    elif not uploaded_files:
        st.error("Please upload at least one CV.")

    elif not job_title:
        st.error("Please enter the Job Title.")

    else:

        try:

            client = Groq(api_key=api_key)

            job_info = {
                "job_title": job_title,
                "min_experience": min_experience,
                "job_description": job_description,
                "keywords": required_keywords
            }

            st.session_state.job_info = job_info

            # Clear previous results
            st.session_state.candidate_data = []
            st.session_state.analysis_errors = []

            progress = st.progress(0)

            status = st.empty()

            total_files = len(uploaded_files)

            for index, uploaded_file in enumerate(uploaded_files):

                status.info(
                    f"Analyzing {uploaded_file.name}..."
                )

                try:

                    cv_text = extract_text(uploaded_file)

                    if not cv_text.strip():
                        raise Exception(
                            "No readable text found in the CV."
                        )

                    candidate = analyze_cv(
                        client,
                        model_name,
                        cv_text,
                        job_info
                    )

                    # Store original file name
                    candidate["cv_file"] = uploaded_file.name

                    # Store raw CV text
                    candidate["cv_text"] = cv_text

                    st.session_state.candidate_data.append(
                        candidate
                    )

                except Exception as e:

                    st.session_state.analysis_errors.append(
                        {
                            "file": uploaded_file.name,
                            "error": str(e)
                        }
                    )

                progress.progress(
                    (index + 1) / total_files
                )

            st.session_state.processed = True

            status.success(
                "CV analysis completed."
            )

        except Exception as e:

            st.error(
                f"Could not initialize Groq client: {str(e)}"
            )


# ============================================================
# DISPLAY ERRORS
# ============================================================

if st.session_state.analysis_errors:

    st.warning(
        "Some CVs could not be processed."
    )

    for error in st.session_state.analysis_errors:

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

    # --------------------------------------------------------
    # RECALCULATE LOCALLY
    # --------------------------------------------------------

    results = recalculate_results()

    st.markdown("---")

    st.header("📊 Candidate Ranking")

    # --------------------------------------------------------
    # CURRENT WEIGHTS
    # --------------------------------------------------------

    st.subheader("Current Scoring Weights")

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
        "Change the sliders in the sidebar. The candidate scores and ranking "
        "will update immediately without re-running the AI analysis."
    )


    # --------------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    average_score = sum(
        x["overall_score"] for x in results
    ) / len(results)

    highest_score = max(
        x["overall_score"] for x in results
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


    # --------------------------------------------------------
    # RANKING TABLE
    # --------------------------------------------------------

    ranking_data = []

    for candidate in results:

        ranking_data.append(
            {
                "Rank": candidate["rank"],
                "Candidate": candidate.get(
                    "candidate_name",
                    "Not specified"
                ),
                "CV File": candidate["cv_file"],
                "Overall Score": candidate["overall_score"],
                "Experience": candidate["experience_score"],
                "Technical Skills": candidate["technical_score"],
                "Education": candidate["education_score"],
                "Job Requirements": candidate["requirement_score"],
                "Certifications": candidate["certification_score"],
                "Matched Keywords": ", ".join(
                    candidate.get(
                        "required_keywords_found",
                        []
                    )
                ),
                "Missing Keywords": ", ".join(
                    candidate.get(
                        "required_keywords_missing",
                        []
                    )
                )
            }
        )

    ranking_df = pd.DataFrame(ranking_data)

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # SCORE COMPARISON
    # --------------------------------------------------------

    st.subheader("📈 Candidate Score Comparison")

    chart_df = ranking_df[
        ["Candidate", "Overall Score"]
    ].copy()

    chart_df = chart_df.set_index(
        "Candidate"
    )

    st.bar_chart(
        chart_df
    )


    # --------------------------------------------------------
    # WEIGHT CONTRIBUTION
    # --------------------------------------------------------

    st.subheader("⚖️ Weight Contribution Analysis")

    st.caption(
        "This shows exactly how each scoring factor contributes to the "
        "candidate's final score."
    )

    contribution_rows = []

    for candidate in results:

        contribution_rows.append(
            {
                "Candidate": candidate.get(
                    "candidate_name",
                    "Not specified"
                ),
                "Experience Contribution":
                    candidate["experience_contribution"],
                "Technical Contribution":
                    candidate["technical_contribution"],
                "Education Contribution":
                    candidate["education_contribution"],
                "Requirements Contribution":
                    candidate["requirements_contribution"],
                "Certification Contribution":
                    candidate["certification_contribution"],
                "Overall Score":
                    candidate["overall_score"]
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


    # --------------------------------------------------------
    # DETAILED CANDIDATE ANALYSIS
    # --------------------------------------------------------

    st.markdown("---")

    st.header("🔎 Detailed Candidate Analysis")

    candidate_names = [
        f"{x['rank']}. {x.get('candidate_name', 'Not specified')}"
        for x in results
    ]

    selected_candidate_name = st.selectbox(
        "Select Candidate",
        candidate_names
    )

    selected_index = candidate_names.index(
        selected_candidate_name
    )

    candidate = results[selected_index]


    # --------------------------------------------------------
    # CANDIDATE HEADER
    # --------------------------------------------------------

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
        f"{candidate.get('years_relevant_experience', 0)} years"
    )

    st.write(
        f"**CV File:** {candidate['cv_file']}"
    )


    # --------------------------------------------------------
    # SCORE BREAKDOWN
    # --------------------------------------------------------

    st.subheader("Score Breakdown")

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
                candidate["experience_score"],
                candidate["technical_score"],
                candidate["education_score"],
                candidate["requirement_score"],
                candidate["certification_score"]
            ],
            "Weight (%)": [
                st.session_state.weight_experience,
                st.session_state.weight_technical,
                st.session_state.weight_education,
                st.session_state.weight_requirements,
                st.session_state.weight_certification
            ],
            "Contribution": [
                candidate["experience_contribution"],
                candidate["technical_contribution"],
                candidate["education_contribution"],
                candidate["requirements_contribution"],
                candidate["certification_contribution"]
            ]
        }
    )

    st.dataframe(
        score_breakdown,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # CANDIDATE INFORMATION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎓 Education")

        education = candidate.get(
            "education",
            []
        )

        if education:
            for item in education:
                st.write(f"• {item}")
        else:
            st.write("Not specified")


        st.subheader("🛠 Technical Skills")

        skills = candidate.get(
            "technical_skills",
            []
        )

        if skills:
            for item in skills:
                st.write(f"• {item}")
        else:
            st.write("Not specified")


        st.subheader("💻 Software / Tools")

        software = candidate.get(
            "software_tools",
            []
        )

        if software:
            for item in software:
                st.write(f"• {item}")
        else:
            st.write("Not specified")


        st.subheader("📜 Certifications")

        certifications = candidate.get(
            "certifications",
            []
        )

        if certifications:
            for item in certifications:
                st.write(f"• {item}")
        else:
            st.write("Not specified")


    with col2:

        st.subheader("✅ Matched Requirements")

        matched = candidate.get(
            "required_keywords_found",
            []
        )

        if matched:
            for item in matched:
                st.success(item)
        else:
            st.write("None")


        st.subheader("❌ Missing Requirements")

        missing = candidate.get(
            "required_keywords_missing",
            []
        )

        if missing:
            for item in missing:
                st.error(item)
        else:
            st.write("None")


        st.subheader("💼 Previous Roles")

        roles = candidate.get(
            "previous_roles",
            []
        )

        if roles:
            for item in roles:
                st.write(f"• {item}")
        else:
            st.write("Not specified")


    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    st.subheader("📌 Relevant Experience Evidence")

    evidence = candidate.get(
        "relevant_experience_evidence",
        []
    )

    if evidence:

        for item in evidence:
            st.write(f"• {item}")

    else:

        st.write(
            "No specific evidence provided."
        )


    # --------------------------------------------------------
    # STRENGTHS & GAPS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("💪 Strengths")

        strengths = candidate.get(
            "strengths",
            []
        )

        if strengths:

            for item in strengths:
                st.success(item)

        else:

            st.write("No specific strengths identified.")


    with col2:

        st.subheader("⚠️ Potential Gaps")

        gaps = candidate.get(
            "potential_gaps",
            []
        )

        if gaps:

            for item in gaps:
                st.warning(item)

        else:

            st.write("No significant gaps identified.")


    # ========================================================
    # EXCEL EXPORT
    # ========================================================

    st.markdown("---")

    st.header("📥 Export Results")

    export_ranking = ranking_df.copy()

    export_details = []

    for candidate in results:

        export_details.append(
            {
                "Candidate": candidate.get(
                    "candidate_name",
                    "Not specified"
                ),
                "CV File": candidate["cv_file"],
                "Overall Score": candidate["overall_score"],
                "Rank": candidate["rank"],
                "Relevant Experience": candidate.get(
                    "years_relevant_experience",
                    0
                ),
                "Experience Score": candidate["experience_score"],
                "Technical Score": candidate["technical_score"],
                "Education Score": candidate["education_score"],
                "Requirement Score": candidate["requirement_score"],
                "Certification Score": candidate["certification_score"],
                "Experience Contribution":
                    candidate["experience_contribution"],
                "Technical Contribution":
                    candidate["technical_contribution"],
                "Education Contribution":
                    candidate["education_contribution"],
                "Requirements Contribution":
                    candidate["requirements_contribution"],
                "Certification Contribution":
                    candidate["certification_contribution"],
                "Matched Keywords": ", ".join(
                    candidate.get(
                        "required_keywords_found",
                        []
                    )
                ),
                "Missing Keywords": ", ".join(
                    candidate.get(
                        "required_keywords_missing",
                        []
                    )
                ),
                "Education": "; ".join(
                    candidate.get(
                        "education",
                        []
                    )
                ),
                "Technical Skills": "; ".join(
                    candidate.get(
                        "technical_skills",
                        []
                    )
                ),
                "Software Tools": "; ".join(
                    candidate.get(
                        "software_tools",
                        []
                    )
                ),
                "Certifications": "; ".join(
                    candidate.get(
                        "certifications",
                        []
                    )
                ),
                "Previous Roles": "; ".join(
                    candidate.get(
                        "previous_roles",
                        []
                    )
                ),
                "Strengths": "; ".join(
                    candidate.get(
                        "strengths",
                        []
                    )
                ),
                "Potential Gaps": "; ".join(
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

        weight_export.to_excel(
            writer,
            sheet_name="Scoring Weights",
            index=False
        )

    excel_buffer.seek(0)

    st.download_button(
        label="📊 Download Excel Ranking",
        data=excel_buffer,
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
    "HireTech AI | AI-assisted recruitment screening and "
    "transparent candidate scoring"
)
