import streamlit as st
import asyncio
from pathlib import Path
from services.qb_service import QBService
from config.settings import settings
from utils.logger import get_logger
from utils.bootstrap import setup_app, apply_custom_css
from components.sidebar import render_sidebar

st.set_page_config(page_title="QB Generator - EduSolve", page_icon="🎯", layout="wide")
setup_app()
apply_custom_css()
render_sidebar()

logger = get_logger()

def render_qb_generator():
    st.title("🎯 Question Bank Generator")
    st.markdown("Generate custom question banks from course materials")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Course Material", type=["pdf", "docx", "txt"])

    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.markdown("### ⚙️ Generation Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            num_questions = st.number_input("Number of Questions", min_value=5, max_value=100, value=20)
            difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard", "Mixed"])
        with col2:
            question_types = st.multiselect("Question Types", ["Short Answer", "Long Answer", "Numerical", "MCQ"], default=["Short Answer", "Long Answer"])

        st.markdown("**Marks Distribution**")
        col1, col2, col3 = st.columns(3)
        marks_2 = col1.number_input("2-mark questions", 0, 50, 5)
        marks_5 = col2.number_input("5-mark questions", 0, 50, 10)
        marks_10 = col3.number_input("10-mark questions", 0, 50, 5)

        topics_input = st.text_input("Specific Topics (comma-separated, optional)")
        topics = [t.strip() for t in topics_input.split(",")] if topics_input else None

        if st.button("🎯 Generate Question Bank", type="primary"):
            with st.spinner("Generating..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p, text):
                    progress_bar.progress(p)
                    status_text.text(text)

                try:
                    service = QBService(st.session_state.llm_router)
                    result = asyncio.run(service.generate_qb(
                        uploaded_file, num_questions, difficulty, question_types,
                        {"2": marks_2, "5": marks_5, "10": marks_10},
                        topics, progress_callback=update_progress
                    ))
                    
                    st.success(f"✅ Question Bank Generated! Total: {len(result['questions'])}")
                    with open(result['saved_path'], "rb") as f:
                        st.download_button("⬇️ Download Generated QB", f, file_name=Path(result['saved_path']).name)
                    
                    st.markdown("### 👀 Preview")
                    for i, q in enumerate(result['questions'][:5], 1):
                        with st.expander(f"Question {i}"):
                            st.write(q.get('question_text'))
                            st.info(f"Type: {q.get('question_type')} | Marks: {q.get('marks')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"QB generation error: {str(e)}", exc_info=True)

render_qb_generator()
