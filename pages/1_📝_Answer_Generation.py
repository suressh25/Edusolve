import streamlit as st
import asyncio
from pathlib import Path
from services.answer_service import AnswerService
from utils.file_handler import FileHandler
from config.settings import settings
from utils.logger import get_logger
from utils.bootstrap import setup_app, apply_custom_css
from components.sidebar import render_sidebar

st.set_page_config(page_title="Answer Generation - EduSolve", page_icon="📝", layout="wide")
setup_app()
apply_custom_css()
render_sidebar()

logger = get_logger()

def render_answer_generation():
    st.title("📝 Answer Generation Module")
    st.markdown("Upload question banks and generate comprehensive answer booklets")
    st.markdown("---")

    # Step 1: File Upload
    st.markdown("### 📤 Step 1: Upload Question Bank")
    
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
    if "file_validated" not in st.session_state:
        st.session_state.file_validated = False

    uploaded_file = st.file_uploader(
        "Upload Question Bank",
        type=settings.ALLOWED_EXTENSIONS,
        help="Supports PDF, DOCX, TXT, and image files",
        key="qb_file_uploader",
    )

    if uploaded_file and st.session_state.uploaded_file_name != uploaded_file.name:
        try:
            FileHandler.validate_file(uploaded_file)
            file_path = asyncio.run(FileHandler.save_uploaded_file(uploaded_file, str(settings.UPLOAD_DIR)))
            
            # Reset state
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_file_path = file_path
            st.session_state.file_validated = True
            
            # Clear previous results
            results_keys = ["processing_complete", "extracted_questions", "cleaned_qb_path", "answer_booklet_path", "generated_answers"]
            for k in results_keys:
                if k in st.session_state: del st.session_state[k]
                
            st.success(f"✅ File validated and saved: {uploaded_file.name}")
        except ValueError as e:
            st.error(f"❌ {str(e)}")
            st.session_state.file_validated = False

    if st.session_state.file_validated and st.session_state.uploaded_file_name:
        st.info(f"📁 **Current File:** {st.session_state.uploaded_file_name}")

    st.markdown("---")

    if st.session_state.file_validated:
        # Step 2: Options
        st.markdown("### ⚙️ Step 2: Configure Processing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            use_rag = st.checkbox(
                "Use RAG (Study Materials)",
                value=st.session_state.get("use_rag_option", False),
                disabled=st.session_state.rag_retriever is None,
                help="Generate answers using your uploaded study materials",
            )
            st.session_state.use_rag_option = use_rag

        with col2:
            output_format = st.selectbox(
                "Output Format",
                ["Word (.docx)", "PDF (.pdf)"],
                index=st.session_state.get("output_format_index", 0),
            )
            st.session_state.output_format_index = 0 if output_format == "Word (.docx)" else 1

        st.markdown("---")

        # Step 3: Process
        st.markdown("### 🚀 Step 3: Process Question Bank")
        
        if not st.session_state.get("processing_complete") and not st.session_state.get("processing_in_progress"):
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                st.session_state.processing_in_progress = True
                st.rerun()

        if st.session_state.get("processing_in_progress"):
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p, text):
                    progress_bar.progress(p)
                    status_text.text(text)

                try:
                    service = AnswerService(st.session_state.llm_router, st.session_state.rag_retriever)
                    result = asyncio.run(service.process_question_bank(
                        st.session_state.uploaded_file_path,
                        st.session_state.use_rag_option,
                        output_format,
                        progress_callback=update_progress
                    ))
                    
                    st.session_state.extracted_questions = result["questions_count"] # Just for summary
                    st.session_state.cleaned_qb_path = result["cleaned_qb_path"]
                    st.session_state.answer_booklet_path = result["answer_booklet_path"]
                    st.session_state.generated_answers = result["answers"]
                    
                    st.session_state.processing_complete = True
                    st.session_state.processing_in_progress = False
                    st.success("🎉 **Processing Complete!**")
                    st.balloons()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Extracted", result["questions_count"])
                    col2.metric("Valid", result["final_questions_count"])
                    col3.metric("Successful", result["success_count"])

                except Exception as e:
                    st.error(f"❌ **Error:** {str(e)}")
                    logger.error(f"Processing error: {str(e)}", exc_info=True)
                    st.session_state.processing_in_progress = False

        st.markdown("---")

        # Step 4: Download
        if st.session_state.get("processing_complete"):
            st.markdown("### 📥 Step 4: Download Your Files")
            cleaned_path = st.session_state.get("cleaned_qb_path")
            booklet_path = st.session_state.get("answer_booklet_path")

            if cleaned_path and booklet_path and Path(cleaned_path).exists() and Path(booklet_path).exists():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    with open(cleaned_path, "rb") as f:
                        st.download_button("📄 Download Cleaned QB", f, file_name=Path(cleaned_path).name, use_container_width=True)
                
                with col2:
                    with open(booklet_path, "rb") as f:
                        st.download_button("📘 Download Answer Booklet", f, file_name=Path(booklet_path).name, use_container_width=True)
                
                with col3:
                    if st.button("🔄 Process New", use_container_width=True):
                        for k in ["processing_complete", "processing_in_progress", "uploaded_file_name", "file_validated"]:
                            if k in st.session_state: del st.session_state[k]
                        st.rerun()

    else:
        st.info("👆 **Please upload a question bank file to get started**")

render_answer_generation()
