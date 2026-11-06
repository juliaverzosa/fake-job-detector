import joblib
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Multi-Modal Fake Job Detector", page_icon="", layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 800; }
    .sub-header { font-size: 1.2rem; color: #4b5563; margin-bottom: 2rem; }
    .metric-card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; }
    </style
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load('fake_job_lgbm_model.pkl')
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return model, embedder

with st.spinner("Loading AI models and embedding weights..."):
    model, embedder = load_artifacts()

st. markdown('<p class="main-header"> MULTI-MODAL FAKE JOB DETECTOR</p>', unsafe_allow_html=True)
st. markdown('<p class="sub-header">AI system powered by Transformer semantic embeddings (all-MiniLM-L6-v2) and structural metadata analysis via LightGBM</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1.3, 0.7], gap="large")

with col1:
    st.subheader("Job Posting Details")

    title = st.text_input("Job Title", placeholder="e.g., Software Engineer, Web Developer")
    company_profile = st.text_area("Company Profile", placeholder="Brief background of the hiring company...", height=100)
    description = st.text_area("Job Description", placeholder="Main duties, day-to-day responsibilities...", height=140)
    requirements = st.text_area("Requirements & Qualifications", placeholder="Skills, tech stack, experience needed...", height=120)
    benefits = st.text_area("Benefits & Perks", placeholder="Health insurance, stipend...", height=90)

    st.subheader("Structural Metadata Markers")
    mc1 ,mc2 ,mc3 = st.columns(3)
    with mc1:
        has_logo = st.checkbox("Company Logo Provided", value=False)
    with mc2:
        telecommuting = st.checkbox("Telecommuniting / Remote", value=False)
    with mc3:
        has_questions = st.checkbox("Screening Questions", value=False)

    st.markdown("")
    analyze_btn = st.button("Run Multi-Modal Analysis", type="primary", use_container_width=True)

with col2:
    st.subheader("Intelligence Dashboard")

    if analyze_btn:
        if not description.strip() and not title.strip():
            st.warning("Please provide at least a Job Title and Description to analyze.")
        else:
            with st.spinner("Generating transformer embeddings & evaluating model..."):
                combined_text = f"{title} {company_profile} {description} {requirements} {benefits}"
                text_embedding = embedder.encode([combined_text])
                metadata_features = np.array([[int(has_logo), int(telecommuting), int(has_questions)]])
                X_input = np.hstack((text_embedding, metadata_features))

                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]

                fraud_prob = probabilities[1] * 100
                real_prob = probabilities[0] * 100

            st.markdown("")

            if prediction == 1:
                st.error("**HIGH RISK: Potential Fradulent Listing**")
                st.metric(label="Estimated Fraud Probability", value=f"{fraud_prob:.2f}")
                st.markdown("""
                    **Detected Risk Factors:**
                    * Linguistic patterns strongly align with known employment scams.
                    * Semantic profile deviates from verified corporate listings.
                """)    
            else:
                st.success("**LOW RISK: Appears Legitimate**")
                st.metric(label="Estimated Legitimacy Confidence", value=f"{real_prob:.2f}")
                st.markdown("""
                    **Verification Summary:**
                    * Text semantics match standard corporate structures.
                    * Structural metadata checks out normally.
                """)       

            st.markdown("Risk Probability Gauge")
            st.progress(float(probabilities[1]))
            st.caption(f"Fraud Score: {fraud_prob:.2f}%  |  Legitimacy Score: {real_prob:.2f}%") 

    else:
        st.info("Fill out the job postingsections on the left and click **Run Multi-Modal Analysis** to evaluate.")