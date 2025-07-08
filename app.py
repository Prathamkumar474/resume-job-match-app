import streamlit as st
import PyPDF2
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load job descriptions from CSV
@st.cache_data
def load_jobs():
    return pd.read_csv("jobs.csv")

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Clean and preprocess text
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

# Compute similarity between resume and jobs
def compute_similarity(resume_text, jobs_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embedding = model.encode([resume_text])[0]
    job_embeddings = model.encode(jobs_df['Description'].tolist())
    similarities = [cosine_similarity([resume_embedding], [job])[0][0] for job in job_embeddings]
    jobs_df['Match Score'] = similarities
    jobs_df.sort_values(by='Match Score', ascending=False, inplace=True)
    return jobs_df

# Streamlit UI
st.title("📄 Resume Scanner & Job Match Recommender")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    cleaned_resume = clean_text(resume_text)

    st.subheader("Extracted Resume Text")
    st.write(cleaned_resume[:1000] + "...")

    jobs_df = load_jobs()
    matched_jobs = compute_similarity(cleaned_resume, jobs_df)

    st.subheader("Top Job Matches")
    st.dataframe(matched_jobs[['Job Title', 'Match Score']].head(3))

    best_match = matched_jobs.iloc[0]
    st.subheader(f"Suggested Job Role: {best_match['Job Title']}")
    st.write(f"**Required Skills:** {best_match['Description']}")

    user_skills = set(cleaned_resume.split())
    job_skills = set([skill.lower() for skill in best_match['Description'].split(', ')])
    missing_skills = job_skills - user_skills

    if missing_skills:
        st.warning("Missing Skills: " + ", ".join(missing_skills))
    else:
        st.success("You have all required skills for this role!")
