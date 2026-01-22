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

def init_state():
    """Initialize or reset session state variables"""
    defaults = {
        "raw_questions": None,
        "extracted_filename": None,
        "qb_processing": False,
        "booklet_processing": False,
        "cleaned_qb_path": None,
        "generated_answers": None,
        "answer_booklet_path": None,
        "processing_error": None,
        "extraction_method": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_answer_generation():
    init_state()
    st.title("📝 Answer Generation Module")
    st.markdown("Professional processing for Question Banks and Answer Booklets")
    st.markdown("---")

    service = AnswerService(st.session_state.llm_router, st.session_state.rag_retriever)

    # Step 1: Provide Input
    st.markdown("### 📥 Step 1: Provide Input")
    col_file, col_text = st.columns(2)

    with col_file:
        st.markdown("#### 📁 Upload Document")
        st.info("💡 Best for scanned PDFs, images, or formatted Word documents.")
        uploaded_file = st.file_uploader(
            "Select Question Bank file",
            type=settings.ALLOWED_EXTENSIONS,
            key="qb_uploader"
        )
        if uploaded_file:
            if st.button("🔍 Extract from File", type="primary", key="btn_extract_file"):
                with st.spinner("Extracting questions..."):
                    try:
                        file_path = asyncio.run(FileHandler.save_uploaded_file(uploaded_file, str(settings.UPLOAD_DIR)))
                        questions, method = asyncio.run(service.extract_questions_from_file(file_path))
                        st.session_state.raw_questions = questions
                        st.session_state.extracted_filename = Path(file_path).stem
                        st.session_state.extraction_method = method
                        st.session_state.generated_answers = None # Reset answers on new input
                        st.success(f"✅ Extracted {len(questions)} questions")
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {str(e)}")

    with col_text:
        st.markdown("#### ✍️ Type/Paste Questions")
        st.info("💡 Best for snippets or manually entered academic questions.")
        raw_text = st.text_area("Type questions directly here...", height=110, placeholder="Q1. What is AI?\nQ2. Explain RAG with examples.", key="qb_manual_text")
        if st.button("🔍 Process Manual Text", type="primary", key="btn_extract_text"):
            if raw_text.strip():
                with st.spinner("Processing text..."):
                    try:
                        questions, method = asyncio.run(service.extract_questions_from_text(raw_text))
                        st.session_state.raw_questions = questions
                        st.session_state.extracted_filename = "Manual_Entry"
                        st.session_state.extraction_method = method
                        st.session_state.generated_answers = None
                        st.success("✅ Extracted questions from text")
                    except Exception as e:
                        st.error(f"❌ Extraction failed: {str(e)}")
            else:
                st.warning("⚠️ Please provide some text first.")

    # Show Current State summary
    if st.session_state.raw_questions:
        st.markdown("---")
        method_str = f" via **{st.session_state.extraction_method}**" if st.session_state.extraction_method else ""
        st.markdown(f"📊 **Current Context:** {st.session_state.extracted_filename} ({len(st.session_state.raw_questions)} questions loaded {method_str})")
        
        with st.expander("👁️ Preview Extracted Questions"):
            for q in st.session_state.raw_questions:
                st.markdown(f"**Q{q.get('question_number', '?')}** ({q.get('marks', '0')} marks): {q.get('question_text', 'No text')}")

        st.markdown("---")

        # Step 2: Options
        st.markdown("### ⚙️ Step 2: Configure Options")
        col1, _ = st.columns([1, 1])
        with col1:
            use_rag = st.checkbox(
                "Use RAG (Study Materials)",
                value=st.session_state.get("use_rag_option", False),
                disabled=st.session_state.rag_retriever is None,
                help="Requires study materials indexed in RAG Module",
            )
            st.session_state.use_rag_option = use_rag

        st.markdown("---")

        # Step 3: Independent Flows
        st.markdown("### 🚀 Step 3: Processing Streams")
        
        flow_col1, flow_col2 = st.columns(2)

        # Flow A: Cleaned QB
        with flow_col1:
            st.markdown("#### 🧹 Flow A: Cleaned QB")
            st.caption("Standardizes formatting and validates marks alignment.")
            if st.button("🧹 Generate Cleaned QB", use_container_width=True):
                with st.spinner("Compiling QB..."):
                    try:
                        path = asyncio.run(service.generate_cleaned_qb(st.session_state.raw_questions, st.session_state.extracted_filename))
                        st.session_state.cleaned_qb_path = path
                        st.success("✅ Cleaned QB Ready")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")

            if st.session_state.cleaned_qb_path:
                st.info("📦 Output Available")
                qb_fmt = st.radio("Download as:", ["DOCX", "PDF"], key="qb_fmt_sel", horizontal=True)
                
                if st.button("📥 Prepare & Download QB", key="btn_dl_qb", use_container_width=True):
                    with st.spinner("Preparing..."):
                        try:
                            if qb_fmt == "PDF":
                                # Convert questions to dummy answers for the booklet compiler
                                dummy_answers = [{
                                    "question_number": q.get("question_number", ""),
                                    "question_text": q.get("question_text", ""),
                                    "marks": q.get("marks", "0"),
                                    "answer": "________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________"
                                } for q in st.session_state.raw_questions]
                                path = asyncio.run(service.compile_booklet(dummy_answers, st.session_state.extracted_filename + " (QB)", "PDF"))
                            else:
                                path = st.session_state.cleaned_qb_path
                            
                            with open(path, "rb") as f:
                                st.download_button("⬇️ Click to Save QB", f, file_name=Path(path).name, use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ preparation failed: {str(e)}")

        # Flow B: Answer Booklet
        with flow_col2:
            st.markdown("#### 📘 Flow B: Answer Booklet")
            st.caption("AI-powered answer generation with RAG support.")
            if st.button("📘 Generate Answer Booklet", use_container_width=True, type="primary", key="btn_gen_booklet"):
                st.session_state.booklet_processing = True
                st.rerun()

            if st.session_state.booklet_processing:
                progress_bar = st.progress(0)
                status = st.empty()
                async def update_p(p):
                    progress_bar.progress(p)
                    status.text(f"Generating answers... {int(p*100)}%")
                
                try:
                    # Validate and prepare questions first
                    prep_questions = service.validate_and_prepare_questions(st.session_state.raw_questions)
                    answers = asyncio.run(service.generate_answers(prep_questions, st.session_state.use_rag_option, progress_callback=update_p))
                    st.session_state.generated_answers = answers
                    st.session_state.booklet_processing = False
                    st.success("✅ Answers Generated Successfully")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.booklet_processing = False

            if st.session_state.generated_answers:
                with st.expander("💡 Preview Answers"):
                    for a in st.session_state.generated_answers:
                        st.markdown(f"**Q{a.get('question_number')}**")
                        st.write(a.get('answer', 'Failed to generate'))
                        st.markdown("---")
                
                st.markdown("##### 📥 Download Booklet")
                booklet_fmt = st.selectbox("Choose Format", ["Word (.docx)", "PDF (.pdf)"], key="booklet_fmt_sel")
                
                if st.button("🔨 Compile & Download Booklet", type="primary", use_container_width=True, key="btn_dl_booklet"):
                    with st.spinner("Compiling..."):
                        try:
                            final_fmt = "PDF" if "PDF" in booklet_fmt else "DOCX"
                            final_path = asyncio.run(service.compile_booklet(st.session_state.generated_answers, st.session_state.extracted_filename, final_fmt))
                            with open(final_path, "rb") as f:
                                st.download_button("⬇️ Click to Save Booklet", f, file_name=Path(final_path).name, use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Compilation failed: {str(e)}")

    else:
        st.info("👈 **Provide Step 1 input to begin.**")
        st.image("https://img.freepik.com/free-vector/academic-success-concept-illustration_114360-124.jpg", width=400)

render_answer_generation()
