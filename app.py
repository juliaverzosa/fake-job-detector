import joblib
import streamlit as st

st.set_page_config(
    page_title="Fake Job Detector", page_icon="", layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load('fake_job_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_artifacts()

st.title("AI-Powered Fake Job Posting Detector")
st.write(
    "Paste a job description below to check if it exihibits linguistic patterns common in employement scams"
)

job_input = st.text_area(
    "Paste Job Details (Title, Description, Requirements):", height=250
)

if st.button("Analyze Job Posting", type="primary"):
    if job_input.strip() == "":
        st.warning("Please paste some job text first.")
    else:
        transformed_input = tfidf.transform([job_input])
        prediction = model.predict(transformed_input)
        probability = model.predict_proba(transformed_input)

        if prediction[0] == 1:
            risk_score = probability[0][1] * 100
            st.error(
                f"**HIGH RISK: Potential Fraudulent Listing Detected!** (Confidence:)"
                f" {risk_score:.2f}%)"
            )
            st.write(
                "**WARNING:** This text contains patterns frequently associated with deceptive job ads (e.g., vague requirements, missing corporate details)."
            )
        else:
            safe_score = probability[0][0] * 100
            st.success(
                f"**LOW RISK: Legitimate Listing Appears Safe.** (Confidence:"
                f" {safe_score:.2f}%)"
            )
            st.write(
                "This posting aligns closely with standard, verified corporate job structures"
            )