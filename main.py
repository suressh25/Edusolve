import streamlit as st
from utils.bootstrap import setup_app, apply_custom_css
from components.sidebar import render_sidebar
from utils.logger import get_logger

logger = get_logger()

# Page configuration
st.set_page_config(
    page_title="EduSolve - AI Answer Generation",
    page_icon="📚",
    layout="wide",
)

# Initialize app
setup_app()
apply_custom_css()
render_sidebar()

# Main content
st.markdown("<div class='main-header'>📚 EduSolve</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>GenAI-Powered Automated Answer Generation System</div>", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📝 Answer Generation")
    st.write("Upload question banks in any format (PDF, DOCX, images) and generate comprehensive answers automatically.")
    st.info("Supports typed, scanned, and handwritten questions")

with col2:
    st.markdown("### 🧠 RAG Module")
    st.write("Upload your study materials to generate personalized answers based on your course content.")
    st.info("Uses Retrieval-Augmented Generation")

with col3:
    st.markdown("### 🎯 QB Generator")
    st.write("Generate custom question banks from your course materials with customizable difficulty and types.")
    st.info("AI-powered question creation")

st.markdown("---")

st.markdown("### 🚀 Features")
features = """
- **Multi-format Support**: PDF, DOCX, TXT, JPG, PNG
- **OCR Capability**: Extracts questions from scanned and handwritten documents
- **Mark-Aware Answers**: Scales answer depth based on marks allocation
- **RAG Integration**: Personalized answers from your study materials
- **Free LLM APIs**: Uses Groq, Gemini, and others
- **Professional Output**: Generate Word/PDF answer booklets
"""
st.markdown(features)

st.markdown("---")
st.markdown("### 📖 Quick Start")
st.markdown("1. Navigate to **Answer Generation** to process question banks")
st.markdown("2. Upload study materials in **RAG Module** for personalized answers")
st.markdown("3. Generate custom questions in **QB Generator**")
st.markdown("4. Configure API keys in **Settings**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "EduSolve v2.1 | Refactored Architecture | Powered by Free LLM APIs"
    "</div>",
    unsafe_allow_html=True,
)
