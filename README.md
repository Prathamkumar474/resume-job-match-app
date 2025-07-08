# 📄 Resume Scanner & Job Match Recommender

A web app that allows users to upload their resume and receive job role recommendations based on their skills. It uses **Natural Language Processing (NLP)** and **semantic similarity** to compare resume content with predefined job descriptions and highlights missing skills.

---

## 🔍 Features

- ✅ Upload your resume in PDF format
- ✅ Extracts and cleans resume text using Python
- ✅ Matches resume content to job descriptions using `sentence-transformers`
- ✅ Recommends top 3 job roles with similarity scores
- ✅ Highlights missing skills to improve your job readiness
- ✅ Interactive user interface built with **Streamlit**

---

## 🧠 Technologies Used

- Python
- Streamlit
- PyPDF2
- pandas
- scikit-learn
- sentence-transformers (`all-MiniLM-L6-v2`)

---

## 📦 Installation

1. **Clone the repo**  
```bash
git clone https://github.com/Prathamkumar474/resume-job-match-app.git
cd resume-job-match-app
