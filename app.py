import streamlit as st
import pandas as pd
import json
import re
import io

from pypdf import PdfReader
from docx import Document
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HireTech AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 HireTech AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered CV Screening and Candidate Matching Platform'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = []

if "processed" not in st.session_state:
    st.session_state.processed = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ HireTech AI Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key."
    )

    model_name = st.selectbox(
        "AI Model",
        [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b"
        ],
        index=0
    )

    st.divider()

    st.subheader("Scoring Weights")

    experience_weight = st.slider(
        "Relevant Experience",
        0,
        100,
        30
    )

    technical_weight = st.slider(
        "Technical Skills",
        0,
        100,
        30
    )

    education_weight = st.slider(
        "Education",
        0,
        100,
        15
    )

    requirements_weight = st.slider(
        "Job Requirements",
        0,
        100,
        15
    )

    certification_weight = st.slider(
        "Certifications",
        0,
        100,
        10
    )

    total_weight = (
        experience_weight
        + technical_weight
        + education_weight
        + requirements_weight
        + certification_weight
    )

    st.info(
        f"Total scoring weight: {total_weight}%"
    )

    st.divider()

    st.caption(
        "HireTech AI is an AI-assisted recruitment "
        "screening tool. Final recruitment decisions "
        "should be reviewed by qualified HR personnel."
    )


# ============================================================
# JOB REQUIREMENTS
# ============================================================

st.header("1️⃣ Job Requirements")

col1, col2 = st.columns([1, 2])

with col1:

    job_title = st.text_input(
        "Job Title",
        value="Hydrologist"
    )

    minimum_experience = st.number_input(
        "Minimum Relevant Experience (Years)",
        min_value=0,
        max_value=50,
        value=3
    )


with col2:

    job_description = st.text_area(
        "Job Description / Requirements",
        height=230,
        value="""We are looking for a Hydrologist for a water resources engineering position.

The candidate should have experience in:

- Hydrological modelling
- Flood modelling
- HEC-HMS
- HEC-RAS
- GIS
- Rainfall-runoff analysis
- Flood frequency analysis
- Hydrological data analysis

Educational requirement:

Bachelor's degree in Civil Engineering, Water Resources Engineering,
Hydrology, Hydraulics, Environmental Engineering or a related field.

Preferred:

Master's degree in Water Resources Engineering, Hydrology,
Hydraulics or a related field."""
    )


required_keywords_text = st.text_input(
    "Required Keywords",
    value=(
        "HEC-HMS, HEC-RAS, Hydrology, Flood modelling, GIS, "
        "Rainfall-runoff, Flood frequency analysis, Civil Engineering"
    ),
    help="Enter keywords separated by commas."
)

required_keywords = [
    keyword.strip()
    for keyword in required_keywords_text.split(",")
    if keyword.strip()
]


# ============================================================
# CV UPLOAD
# ============================================================

st.header("2️⃣ Upload Candidate CVs")

uploaded_files = st.file_uploader(
    "Upload PDF or Word CVs",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Upload multiple candidate CVs for analysis."
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} CV(s) uploaded successfully."
    )

    uploaded_df = pd.DataFrame({
        "CV File": [
            file.name
            for file in uploaded_files
        ]
    })

    st.dataframe(
        uploaded_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_docx_text(file):

    doc = Document(file)

    text = []

    # Normal paragraphs
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

                if cell.text.strip():

                    row_text.append(
                        cell.text.strip()
                    )

            if row_text:

                text.append(
                    " | ".join(row_text)
                )

    return "\n".join(text)


# ============================================================
# UNIVERSAL CV TEXT EXTRACTION
# ============================================================

def extract_cv_text(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_pdf_text(file)

    elif filename.endswith(".docx"):

        return extract_docx_text(file)

    else:

        raise ValueError(
            "Unsupported file type. "
            "Only PDF and DOCX are supported."
        )


# ============================================================
# CLEAN AI JSON RESPONSE
# ============================================================

def clean_json_response(response_text):

    if not response_text:

        raise ValueError(
            "AI returned an empty response."
        )

    response_text = response_text.strip()

    # Remove markdown JSON fences
    response_text = re.sub(
        r"```json",
        "",
        response_text,
        flags=re.IGNORECASE
    )

    response_text = re.sub(
        r"```",
        "",
        response_text
    )

    response_text = response_text.strip()

    # Locate JSON object
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "No JSON object found in AI response."
        )

    return response_text[
        start:end + 1
    ]


# ============================================================
# AI CANDIDATE ANALYSIS
# ============================================================

def analyze_candidate(
    client,
    model_name,
    cv_text,
    job_title,
    job_description,
    required_keywords,
    minimum_experience
):

    prompt = f"""
You are HireTech AI, an AI-powered HR CV screening assistant.

Analyze the candidate CV against the job requirements.

JOB TITLE:
{job_title}

MINIMUM RELEVANT EXPERIENCE:
{minimum_experience} years

JOB DESCRIPTION:
{job_description}

REQUIRED KEYWORDS:
{json.dumps(required_keywords)}

CANDIDATE CV:
{cv_text}

Your task is to extract factual information from the CV.

IMPORTANT RULES:

1. Use ONLY information contained in the CV.
2. Do NOT invent information.
3. Do NOT assume qualifications or experience.
4. Estimate relevant experience only from documented employment history.
5. Identify technical skills actually mentioned in the CV.
6. Identify software/tools actually mentioned in the CV.
7. Match required keywords against the actual CV.
8. Put matched keywords in required_keywords_found.
9. Put missing keywords in required_keywords_missing.
10. Identify evidence supporting relevant experience.
11. If information is unavailable, use "Not specified".
12. years_relevant_experience must be a number.
13. Return ONLY ONE JSON OBJECT.
14. Do NOT use Markdown.
15. Do NOT use ```json.
16. Do NOT include explanations outside the JSON.

Use EXACTLY this structure:

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

Return the JSON object now.
"""

    response = client.chat.completions.create(

        model=model_name,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are HireTech AI. "
                    "You analyze CVs accurately. "
                    "Return one valid JSON object only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# EXPERIENCE SCORE
# ============================================================

def calculate_experience_score(
    years,
    minimum_required
):

    try:

        years = float(years)

    except:

        years = 0

    if years >= minimum_required + 4:

        return 100

    elif years >= minimum_required + 2:

        return 90

    elif years >= minimum_required:

        return 75

    elif years >= max(
        minimum_required - 1,
        0
    ):

        return 50

    elif years > 0:

        return 30

    else:

        return 0


# ============================================================
# TECHNICAL SKILLS SCORE
# ============================================================

def calculate_technical_score(
    candidate,
    required_keywords
):

    technical_text = " ".join(

        candidate.get(
            "technical_skills",
            []
        )

        + candidate.get(
            "software_tools",
            []
        )

    ).lower()

    if not required_keywords:

        return 0

    matched = 0

    for keyword in required_keywords:

        if keyword.lower() in technical_text:

            matched += 1

    return (
        matched
        / len(required_keywords)
        * 100
    )


# ============================================================
# EDUCATION SCORE
# ============================================================

def calculate_education_score(
    education
):

    if not education:

        return 0

    education_text = " ".join(
        str(x)
        for x in education
    ).lower()

    score = 0

    if "civil engineering" in education_text:

        score = max(
            score,
            80
        )

    if "water resources" in education_text:

        score = max(
            score,
            95
        )

    if "hydrology" in education_text:

        score = max(
            score,
            95
        )

    if "hydraulics" in education_text:

        score = max(
            score,
            95
        )

    if "environmental engineering" in education_text:

        score = max(
            score,
            75
        )

    if "master" in education_text:

        score = min(
            100,
            score + 5
        )

    return score


# ============================================================
# JOB REQUIREMENT SCORE
# ============================================================

def calculate_requirement_score(
    candidate,
    required_keywords
):

    if not required_keywords:

        return 0

    found = candidate.get(
        "required_keywords_found",
        []
    )

    found_lower = {
        str(x).lower()
        for x in found
    }

    matched = 0

    for keyword in required_keywords:

        if keyword.lower() in found_lower:

            matched += 1

    return (
        matched
        / len(required_keywords)
        * 100
    )


# ============================================================
# CERTIFICATION SCORE
# ============================================================

def calculate_certification_score(
    certifications
):

    if not certifications:

        return 0

    if certifications == [
        "Not specified"
    ]:

        return 0

    return 100


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    candidate,
    required_keywords,
    minimum_required,
    weights
):

    experience_score = (
        calculate_experience_score(
            candidate.get(
                "years_relevant_experience",
                0
            ),
            minimum_required
        )
    )

    technical_score = (
        calculate_technical_score(
            candidate,
            required_keywords
        )
    )

    education_score = (
        calculate_education_score(
            candidate.get(
                "education",
                []
            )
        )
    )

    requirement_score = (
        calculate_requirement_score(
            candidate,
            required_keywords
        )
    )

    certification_score = (
        calculate_certification_score(
            candidate.get(
                "certifications",
                []
            )
        )
    )

    total_weight = sum(
        weights.values()
    )

    if total_weight == 0:

        total_weight = 100

    # Normalize weights
    experience_w = (
        weights["experience"]
        / total_weight
    )

    technical_w = (
        weights["technical"]
        / total_weight
    )

    education_w = (
        weights["education"]
        / total_weight
    )

    requirement_w = (
        weights["requirements"]
        / total_weight
    )

    certification_w = (
        weights["certifications"]
        / total_weight
    )

    overall_score = (

        experience_score
        * experience_w

        + technical_score
        * technical_w

        + education_score
        * education_w

        + requirement_score
        * requirement_w

        + certification_score
        * certification_w
    )

    return {

        "Experience Score":
            round(
                experience_score,
                2
            ),

        "Technical Skills Score":
            round(
                technical_score,
                2
            ),

        "Education Score":
            round(
                education_score,
                2
            ),

        "Job Requirements Score":
            round(
                requirement_score,
                2
            ),

        "Certification Score":
            round(
                certification_score,
                2
            ),

        "Overall Score":
            round(
                overall_score,
                2
            )
    }


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.header("3️⃣ Analyze Candidates")

analyze_button = st.button(
    "🚀 Analyze & Rank Candidates",
    type="primary",
    use_container_width=True
)


if analyze_button:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not groq_api_key:

        st.error(
            "Please enter your Groq API key in the sidebar."
        )

        st.stop()

    if not uploaded_files:

        st.error(
            "Please upload at least one CV."
        )

        st.stop()

    if not job_description.strip():

        st.error(
            "Please enter a job description."
        )

        st.stop()

    # --------------------------------------------------------
    # INITIALIZE GROQ
    # --------------------------------------------------------

    try:

        client = Groq(
            api_key=groq_api_key
        )

    except Exception as e:

        st.error(
            f"Could not initialize Groq: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # WEIGHTS
    # --------------------------------------------------------

    weights = {

        "experience":
            experience_weight,

        "technical":
            technical_weight,

        "education":
            education_weight,

        "requirements":
            requirements_weight,

        "certifications":
            certification_weight
    }

    results = []

    progress_bar = st.progress(0)

    status_text = st.empty()

    # --------------------------------------------------------
    # PROCESS EACH CV
    # --------------------------------------------------------

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        filename = uploaded_file.name

        status_text.write(
            f"🔄 Analyzing **{filename}**..."
        )

        try:

            # Extract CV
            cv_text = extract_cv_text(
                uploaded_file
            )

            if not cv_text.strip():

                raise ValueError(
                    "No readable text found in CV. "
                    "The PDF may be scanned/image-based."
                )

            # AI analysis
            ai_result = analyze_candidate(

                client,

                model_name,

                cv_text,

                job_title,

                job_description,

                required_keywords,

                minimum_experience
            )

            # Debug information is hidden
            # unless an error occurs.

            # Clean JSON
            cleaned_result = (
                clean_json_response(
                    ai_result
                )
            )

            # Convert JSON to Python dictionary
            candidate = json.loads(
                cleaned_result
            )

            # Calculate scores
            scores = (
                calculate_overall_score(

                    candidate,

                    required_keywords,

                    minimum_experience,

                    weights
                )
            )

            # Store filename
            candidate["filename"] = filename

            # Add scores
            candidate.update(
                scores
            )

            # Store result
            results.append(
                candidate
            )

        except Exception as e:

            st.error(
                f"Could not process "
                f"{filename}: {str(e)}"
            )

            # Show AI response for debugging
            if "ai_result" in locals():

                with st.expander(
                    f"🔧 Debug response - {filename}"
                ):

                    st.code(
                        str(ai_result)
                    )

        progress_bar.progress(
            (index + 1)
            / len(uploaded_files)
        )

    status_text.write(
        "✅ Candidate analysis completed."
    )

    # Save results
    st.session_state.results = results

    st.session_state.processed = True


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.processed:

    results = st.session_state.results

    if not results:

        st.error(
            "No candidates were successfully processed."
        )

        st.stop()

    # ========================================================
    # CREATE RANKING DATA
    # ========================================================

    ranking_data = []

    for candidate in results:

        ranking_data.append({

            "Candidate":
                candidate.get(
                    "candidate_name",
                    "Not specified"
                ),

            "CV File":
                candidate.get(
                    "filename",
                    ""
                ),

            "Overall Score":
                candidate.get(
                    "Overall Score",
                    0
                ),

            "Experience":
                candidate.get(
                    "years_relevant_experience",
                    0
                ),

            "Technical Skills":
                candidate.get(
                    "Technical Skills Score",
                    0
                ),

            "Education":
                candidate.get(
                    "Education Score",
                    0
                ),

            "Job Requirements":
                candidate.get(
                    "Job Requirements Score",
                    0
                ),

            "Certifications":
                candidate.get(
                    "Certification Score",
                    0
                ),

            "Matched Keywords":
                ", ".join(
                    map(
                        str,
                        candidate.get(
                            "required_keywords_found",
                            []
                        )
                    )
                ),

            "Missing Keywords":
                ", ".join(
                    map(
                        str,
                        candidate.get(
                            "required_keywords_missing",
                            []
                        )
                    )
                )
        })

    ranking_df = pd.DataFrame(
        ranking_data
    )

    # Sort by score
    ranking_df = ranking_df.sort_values(
        by="Overall Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # Add ranking
    ranking_df.insert(
        0,
        "Rank",
        range(
            1,
            len(ranking_df) + 1
        )
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    st.header("4️⃣ Candidate Ranking Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Candidates Analyzed",
            len(results)
        )

    with col2:

        st.metric(
            "Average Score",
            f"{ranking_df['Overall Score'].mean():.1f}"
        )

    with col3:

        st.metric(
            "Highest Score",
            f"{ranking_df['Overall Score'].max():.1f}"
        )

    with col4:

        st.metric(
            "Required Keywords",
            len(required_keywords)
        )

    # ========================================================
    # RANKING TABLE
    # ========================================================

    st.subheader(
        "📊 Candidate Ranking"
    )

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # SCORE CHART
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
    # CANDIDATE DETAILS
    # ========================================================

    st.subheader(
        "🔎 Detailed Candidate Analysis"
    )

    candidate_options = []

    for candidate in results:

        name = candidate.get(
            "candidate_name",
            "Not specified"
        )

        filename = candidate.get(
            "filename",
            ""
        )

        candidate_options.append(
            f"{name} — {filename}"
        )

    selected_option = st.selectbox(
        "Select candidate",
        candidate_options
    )

    selected_index = candidate_options.index(
        selected_option
    )

    selected = results[
        selected_index
    ]

    # ========================================================
    # CANDIDATE HEADER
    # ========================================================

    st.markdown(
        f"### 👤 {selected.get('candidate_name', 'Not specified')}"
    )

    st.caption(
        f"CV: {selected.get('filename', '')}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Overall Score",
            f"{selected.get('Overall Score', 0):.1f}/100"
        )

    with col2:

        st.metric(
            "Relevant Experience",
            f"{selected.get('years_relevant_experience', 0)} years"
        )

    with col3:

        st.metric(
            "Matched Keywords",
            len(
                selected.get(
                    "required_keywords_found",
                    []
                )
            )
        )

    # ========================================================
    # SCORE BREAKDOWN
    # ========================================================

    st.markdown(
        "#### 📊 Score Breakdown"
    )

    score_data = pd.DataFrame({

        "Category": [

            "Experience",

            "Technical Skills",

            "Education",

            "Job Requirements",

            "Certifications"
        ],

        "Score": [

            selected.get(
                "Experience Score",
                0
            ),

            selected.get(
                "Technical Skills Score",
                0
            ),

            selected.get(
                "Education Score",
                0
            ),

            selected.get(
                "Job Requirements Score",
                0
            ),

            selected.get(
                "Certification Score",
                0
            )
        ]
    })

    score_data = score_data.set_index(
        "Category"
    )

    st.bar_chart(
        score_data
    )

    # ========================================================
    # EDUCATION AND SKILLS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### 🎓 Education"
        )

        education = selected.get(
            "education",
            []
        )

        if education:

            for item in education:

                st.write(
                    f"• {item}"
                )

        else:

            st.write(
                "Not specified"
            )

        st.markdown(
            "#### 💻 Technical Skills"
        )

        skills = selected.get(
            "technical_skills",
            []
        )

        if skills:

            for item in skills:

                st.write(
                    f"• {item}"
                )

        else:

            st.write(
                "Not specified"
            )

        st.markdown(
            "#### 🛠 Software Tools"
        )

        tools = selected.get(
            "software_tools",
            []
        )

        if tools:

            for item in tools:

                st.write(
                    f"• {item}"
                )

        else:

            st.write(
                "Not specified"
            )

    # ========================================================
    # REQUIREMENTS AND CERTIFICATIONS
    # ========================================================

    with col2:

        st.markdown(
            "#### 📜 Certifications"
        )

        certifications = selected.get(
            "certifications",
            []
        )

        if certifications:

            for item in certifications:

                st.write(
                    f"• {item}"
                )

        else:

            st.write(
                "Not specified"
            )

        st.markdown(
            "#### ✅ Matched Requirements"
        )

        matched = selected.get(
            "required_keywords_found",
            []
        )

        if matched:

            for item in matched:

                st.success(
                    item
                )

        else:

            st.write(
                "No matched keywords identified."
            )

        st.markdown(
            "#### ⚠️ Missing Requirements"
        )

        missing = selected.get(
            "required_keywords_missing",
            []
        )

        if missing:

            for item in missing:

                st.warning(
                    item
                )

        else:

            st.write(
                "No missing keywords identified."
            )

    # ========================================================
    # PREVIOUS EXPERIENCE
    # ========================================================

    st.markdown(
        "#### 💼 Previous Roles"
    )

    previous_roles = selected.get(
        "previous_roles",
        []
    )

    if previous_roles:

        for item in previous_roles:

            st.write(
                f"• {item}"
            )

    else:

        st.write(
            "Not specified"
        )

    # ========================================================
    # EXPERIENCE EVIDENCE
    # ========================================================

    st.markdown(
        "#### 📌 Relevant Experience Evidence"
    )

    evidence = selected.get(
        "relevant_experience_evidence",
        []
    )

    if evidence:

        for item in evidence:

            st.info(
                item
            )

    else:

        st.write(
            "No specific evidence identified."
        )

    # ========================================================
    # STRENGTHS AND GAPS
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### 💪 Candidate Strengths"
        )

        strengths = selected.get(
            "strengths",
            []
        )

        if strengths:

            for item in strengths:

                st.success(
                    item
                )

        else:

            st.write(
                "No specific strengths identified."
            )

    with col2:

        st.markdown(
            "#### 🔍 Potential Gaps"
        )

        gaps = selected.get(
            "potential_gaps",
            []
        )

        if gaps:

            for item in gaps:

                st.warning(
                    item
                )

        else:

            st.write(
                "No specific gaps identified."
            )

    # ========================================================
    # EXCEL EXPORT
    # ========================================================

    st.subheader(
        "5️⃣ Export Recruitment Report"
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        # Ranking sheet
        ranking_df.to_excel(
            writer,
            index=False,
            sheet_name="Candidate Ranking"
        )

        # Detailed sheet
        detailed_data = []

        for candidate in results:

            detailed_data.append({

                "Candidate":
                    candidate.get(
                        "candidate_name",
                        "Not specified"
                    ),

                "CV File":
                    candidate.get(
                        "filename",
                        ""
                    ),

                "Overall Score":
                    candidate.get(
                        "Overall Score",
                        0
                    ),

                "Relevant Experience":
                    candidate.get(
                        "years_relevant_experience",
                        0
                    ),

                "Education":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "education",
                                []
                            )
                        )
                    ),

                "Technical Skills":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "technical_skills",
                                []
                            )
                        )
                    ),

                "Software Tools":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "software_tools",
                                []
                            )
                        )
                    ),

                "Certifications":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "certifications",
                                []
                            )
                        )
                    ),

                "Matched Keywords":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "required_keywords_found",
                                []
                            )
                        )
                    ),

                "Missing Keywords":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "required_keywords_missing",
                                []
                            )
                        )
                    ),

                "Strengths":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "strengths",
                                []
                            )
                        )
                    ),

                "Potential Gaps":
                    "; ".join(
                        map(
                            str,
                            candidate.get(
                                "potential_gaps",
                                []
                            )
                        )
                    )
            })

        detailed_df = pd.DataFrame(
            detailed_data
        )

        detailed_df.to_excel(
            writer,
            index=False,
            sheet_name="Candidate Details"
        )

    excel_buffer.seek(0)

    st.download_button(

        label="📥 Download Excel Recruitment Report",

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

st.divider()

st.caption(
    "HireTech AI | AI-Assisted Recruitment & CV Screening"
)
