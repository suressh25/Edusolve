"""
EduSolve - Main Streamlit Application
GenAI-based Automated Answer Generation System
"""

import streamlit as st
import asyncio
from pathlib import Path
import sys
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings

# Import logger ONCE using get_logger
from utils.logger import get_logger

logger = get_logger()

from utils.file_handler import FileHandler

# Import ALL API clients
from api.groq_client import GroqClient
from api.gemini_client import GeminiClient
from api.cerebras_client import CerebrasClient
from api.mistral_client import MistralClient
from api.openrouter_client import OpenRouterClient
from api.cohere_client import CohereClient
from api.llm_router import LLMRouter

# Import extraction modules
from extraction.text_extractor import TextExtractor
from extraction.image_extractor import ImageExtractor
from extraction.qb_cleaner import QuestionBankCleaner

# Import generation modules
from generation.answer_generator import AnswerGenerator
from generation.booklet_compiler import BookletCompiler

# Import RAG modules
from rag.document_processor import DocumentProcessor
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever

# Import question generation
from question_generation.qb_generator import QuestionBankGenerator

# Page configuration
st.set_page_config(
    page_title="EduSolve - AI Answer Generation",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS (keep existing)
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def init_session_state():
    """Initialize session state with multi-provider support"""
    if "llm_router" not in st.session_state:
        # Initialize all clients
        groq_client = GroqClient()
        gemini_client = GeminiClient()

        # Initialize optional clients
        cerebras_client = None
        mistral_client = None
        openrouter_client = None
        cohere_client = None

        try:
            cerebras_client = CerebrasClient()
        except Exception as e:
            logger.warning(f"Cerebras client initialization failed: {e}")

        try:
            mistral_client = MistralClient()
        except Exception as e:
            logger.warning(f"Mistral client initialization failed: {e}")

        try:
            openrouter_client = OpenRouterClient()
        except Exception as e:
            logger.warning(f"OpenRouter client initialization failed: {e}")

        try:
            cohere_client = CohereClient()
        except Exception as e:
            logger.warning(f"Cohere client initialization failed: {e}")

        # Initialize router with all available clients
        st.session_state.llm_router = LLMRouter(
            groq_client,
            gemini_client,
            cerebras_client,
            mistral_client,
            openrouter_client,
            cohere_client,
        )

    if "rag_retriever" not in st.session_state:
        st.session_state.rag_retriever = None

    if "extracted_questions" not in st.session_state:
        st.session_state.extracted_questions = None

    if "generated_answers" not in st.session_state:
        st.session_state.generated_answers = None


init_session_state()

# Sidebar with enhanced API status
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x80/1f77b4/ffffff?text=EduSolve",
        width="stretch",
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📝 Answer Generation",
            "🧠 RAG Module",
            "🎯 QB Generator",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📊 API Status")

    # Get router stats
    if "llm_router" in st.session_state:
        stats = st.session_state.llm_router.get_stats()

        st.markdown("**Available Providers:**")
        for provider in stats["available_providers"]:
            usage = stats["usage_stats"].get(provider, 0)
            st.markdown(f"✅ {provider.capitalize()} ({usage} used)")

        if stats["total_requests"] > 0:
            st.markdown(f"**Total Requests:** {stats['total_requests']}")
    else:
        st.markdown("❌ No providers available")

    # RAG status
    if (
        st.session_state.rag_retriever
        and st.session_state.rag_retriever.vector_store.index
    ):
        st.markdown("✅ RAG Enabled")
    else:
        st.markdown("⭕ RAG Disabled")

# Main content
if page == "🏠 Home":
    st.markdown("<div class='main-header'>📚 EduSolve</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>GenAI-Powered Automated Answer Generation System</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📝 Answer Generation")
        st.write(
            "Upload question banks in any format (PDF, DOCX, images) and generate comprehensive answers automatically."
        )
        st.info("Supports typed, scanned, and handwritten questions")

    with col2:
        st.markdown("### 🧠 RAG Module")
        st.write(
            "Upload your study materials to generate personalized answers based on your course content."
        )
        st.info("Uses Retrieval-Augmented Generation")

    with col3:
        st.markdown("### 🎯 QB Generator")
        st.write(
            "Generate custom question banks from your course materials with customizable difficulty and types."
        )
        st.info("AI-powered question creation")

    st.markdown("---")

    st.markdown("### 🚀 Features")

    features = """
    - **Multi-format Support**: PDF, DOCX, TXT, JPG, PNG
    - **OCR Capability**: Extracts questions from scanned and handwritten documents
    - **Mark-Aware Answers**: Scales answer depth based on marks allocation
    - **RAG Integration**: Personalized answers from your study materials
    - **Free LLM APIs**: Uses Groq, Gemini, and HuggingFace free tiers
    - **Professional Output**: Generate Word/PDF answer booklets
    """

    st.markdown(features)

    st.markdown("---")
    st.markdown("### 📖 Quick Start")
    st.markdown("1. Navigate to **Answer Generation** to process question banks")
    st.markdown("2. Upload study materials in **RAG Module** for personalized answers")
    st.markdown("3. Generate custom questions in **QB Generator**")
    st.markdown("4. Configure API keys in **Settings**")

elif page == "📝 Answer Generation":
    st.title("📝 Answer Generation Module")
    st.markdown("Upload question banks and generate comprehensive answer booklets")

    st.markdown("---")

    # ============================================================================
    # SECTION 1: FILE UPLOAD (Always Visible)
    # ============================================================================
    st.markdown("### 📤 Step 1: Upload Question Bank")

    # Initialize session state for file upload
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

    # Handle file upload
    if uploaded_file:
        # Check if new file uploaded
        if st.session_state.uploaded_file_name != uploaded_file.name:
            try:
                FileHandler.validate_file(uploaded_file)

                # Save file
                file_path = asyncio.run(
                    FileHandler.save_uploaded_file(
                        uploaded_file, str(settings.UPLOAD_DIR)
                    )
                )

                # Update session state
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_path = file_path
                st.session_state.file_validated = True

                # Clear previous processing results
                if "processing_complete" in st.session_state:
                    del st.session_state.processing_complete
                if "extracted_questions" in st.session_state:
                    del st.session_state.extracted_questions
                if "cleaned_qb_path" in st.session_state:
                    del st.session_state.cleaned_qb_path
                if "answer_booklet_path" in st.session_state:
                    del st.session_state.answer_booklet_path
                if "generated_answers" in st.session_state:
                    del st.session_state.generated_answers

                st.success(f"✅ File validated and saved: {uploaded_file.name}")

            except ValueError as e:
                st.error(f"❌ {str(e)}")
                st.session_state.file_validated = False

    # Show file status
    if st.session_state.file_validated and st.session_state.uploaded_file_name:
        st.info(f"📁 **Current File:** {st.session_state.uploaded_file_name}")

    st.markdown("---")

    # ============================================================================
    # SECTION 2: PROCESSING OPTIONS (Always Visible if file uploaded)
    # ============================================================================
    if st.session_state.file_validated:
        st.markdown("### ⚙️ Step 2: Configure Processing Options")

        col1, col2 = st.columns(2)

        with col1:
            use_rag = st.checkbox(
                "Use RAG (Study Materials)",
                value=st.session_state.get("use_rag_option", False),
                disabled=st.session_state.rag_retriever is None,
                help="Generate answers using your uploaded study materials",
                key="use_rag_checkbox",
            )
            st.session_state.use_rag_option = use_rag

        with col2:
            output_format = st.selectbox(
                "Output Format",
                ["Word (.docx)", "PDF (.pdf)"],
                index=st.session_state.get("output_format_index", 0),
                key="output_format_select",
            )
            st.session_state.output_format_index = (
                0 if output_format == "Word (.docx)" else 1
            )

        st.markdown("---")

        # ============================================================================
        # SECTION 3: PROCESSING BUTTON & PROGRESS (Always Visible if file uploaded)
        # ============================================================================
        st.markdown("### 🚀 Step 3: Process Question Bank")

        # Initialize processing state
        if "processing_complete" not in st.session_state:
            st.session_state.processing_complete = False
        if "processing_in_progress" not in st.session_state:
            st.session_state.processing_in_progress = False

        # Process button
        if (
            not st.session_state.processing_complete
            and not st.session_state.processing_in_progress
        ):
            if st.button(
                "🚀 Start Processing", type="primary", use_container_width=True
            ):
                st.session_state.processing_in_progress = True
                st.rerun()

        # Show processing status
        if st.session_state.processing_in_progress:
            with st.spinner("Processing question bank..."):

                file_path = st.session_state.uploaded_file_path
                file_extension = Path(file_path).suffix.lower()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Step 1: Extract questions
                    status_text.text("📖 Extracting questions...")
                    progress_bar.progress(20)

                    if file_extension in [".pdf", ".docx", ".txt"]:
                        # Text extraction
                        text_extractor = TextExtractor(st.session_state.llm_router)

                        if file_extension == ".pdf":
                            # Check if scanned PDF
                            import fitz

                            doc = fitz.open(file_path)
                            first_page_text = doc[0].get_text().strip()
                            doc.close()

                            if len(first_page_text) < 100:
                                # Likely scanned - use vision API
                                image_extractor = ImageExtractor(GeminiClient())
                                questions = asyncio.run(
                                    image_extractor.extract_from_scanned_pdf(file_path)
                                )
                            else:
                                # Digital PDF
                                raw_text = asyncio.run(
                                    text_extractor.extract_text_from_file(file_path)
                                )
                                questions = asyncio.run(
                                    text_extractor.extract_questions_with_llm(raw_text)
                                )
                        else:
                            raw_text = asyncio.run(
                                text_extractor.extract_text_from_file(file_path)
                            )
                            questions = asyncio.run(
                                text_extractor.extract_questions_with_llm(raw_text)
                            )

                    else:
                        # Image extraction
                        image_extractor = ImageExtractor(GeminiClient())
                        questions = asyncio.run(
                            image_extractor.extract_from_image_file(file_path)
                        )

                    st.session_state.extracted_questions = questions

                    progress_bar.progress(40)
                    status_text.text(f"✅ Extracted {len(questions)} questions")

                    # Step 2: Generate cleaned QB
                    status_text.text("🧹 Generating cleaned question bank...")
                    progress_bar.progress(50)

                    qb_cleaner = QuestionBankCleaner(st.session_state.llm_router)
                    cleaned_qb_path = asyncio.run(
                        qb_cleaner.generate_cleaned_document(
                            questions,
                            str(
                                settings.OUTPUT_DIR
                                / f"{Path(st.session_state.uploaded_file_name).stem} - Cleaned.docx"
                            ),
                            Path(st.session_state.uploaded_file_name).stem,
                        )
                    )

                    st.session_state.cleaned_qb_path = cleaned_qb_path

                    progress_bar.progress(55)

                    # Step 3: Parse questions from cleaned QB
                    status_text.text("📖 Reading questions from cleaned QB...")

                    from docx import Document as DocxDocument
                    import re

                    def parse_cleaned_qb(docx_path: str):
                        """Parse questions directly from cleaned QB DOCX structure"""

                        doc = DocxDocument(docx_path)
                        parsed_questions = []
                        current_question = {}

                        for para in doc.paragraphs:
                            text = para.text.strip()

                            if text.startswith("Question Number:"):
                                # Save previous question if exists
                                if current_question and current_question.get(
                                    "question_text"
                                ):
                                    parsed_questions.append(current_question.copy())

                                # Start new question
                                current_question = {
                                    "question_number": text.replace(
                                        "Question Number:", ""
                                    ).strip(),
                                    "question_text": "",
                                    "marks": "0",
                                }

                            elif text.startswith("Question Text:"):
                                if current_question:
                                    current_question["question_text"] = text.replace(
                                        "Question Text:", ""
                                    ).strip()

                            elif text.startswith("Marks Allocated:"):
                                if current_question:
                                    marks_text = text.replace(
                                        "Marks Allocated:", ""
                                    ).strip()
                                    marks_match = re.search(r"\d+", marks_text)
                                    if marks_match:
                                        current_question["marks"] = marks_match.group()

                        # Add last question
                        if current_question and current_question.get("question_text"):
                            parsed_questions.append(current_question)

                        return parsed_questions

                    final_questions = parse_cleaned_qb(cleaned_qb_path)
                    logger.info(
                        f"Parsed {len(final_questions)} questions from cleaned QB"
                    )

                    # Validate questions
                    status_text.text("🔍 Validating questions...")
                    valid_questions = []
                    for q in final_questions:
                        try:
                            marks = int(q.get("marks", 0))
                            if marks == 0:
                                q_text_length = len(q.get("question_text", "").split())
                                if q_text_length < 10:
                                    q["marks"] = "2"
                                elif q_text_length < 20:
                                    q["marks"] = "5"
                                elif q_text_length < 40:
                                    q["marks"] = "10"
                                else:
                                    q["marks"] = "13"
                            valid_questions.append(q)
                        except (ValueError, TypeError):
                            q["marks"] = "5"
                            valid_questions.append(q)

                    final_questions = valid_questions

                    progress_bar.progress(60)
                    status_text.text(f"✅ Validated {len(final_questions)} questions")

                    # Step 4: Generate answers
                    status_text.text("💡 Generating answers...")

                    answer_generator = AnswerGenerator(
                        st.session_state.llm_router,
                        (
                            st.session_state.rag_retriever
                            if st.session_state.use_rag_option
                            else None
                        ),
                    )

                    async def update_progress(progress):
                        progress_bar.progress(int(60 + (progress * 30)))

                    answers = asyncio.run(
                        answer_generator.generate_batch_answers(
                            final_questions,
                            use_rag=st.session_state.use_rag_option,
                            batch_size=5,
                            progress_callback=update_progress,
                        )
                    )

                    st.session_state.generated_answers = answers

                    progress_bar.progress(90)
                    status_text.text("📄 Compiling answer booklet...")

                    # Step 5: Compile booklet
                    booklet_compiler = BookletCompiler()

                    if "PDF" in output_format:
                        output_path = str(
                            settings.OUTPUT_DIR
                            / f"{Path(st.session_state.uploaded_file_name).stem} Answers.pdf"
                        )
                        final_path = asyncio.run(
                            booklet_compiler.compile_to_pdf(
                                answers,
                                output_path,
                                Path(st.session_state.uploaded_file_name).stem,
                            )
                        )
                    else:
                        output_path = str(
                            settings.OUTPUT_DIR
                            / f"{Path(st.session_state.uploaded_file_name).stem} Answers.docx"
                        )
                        final_path = asyncio.run(
                            booklet_compiler.compile_to_word(
                                answers,
                                output_path,
                                Path(st.session_state.uploaded_file_name).stem,
                            )
                        )

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")

                    # Store final output path
                    st.session_state.answer_booklet_path = final_path

                    # Mark processing as complete
                    st.session_state.processing_complete = True
                    st.session_state.processing_in_progress = False

                    # Success message
                    st.success("🎉 **Processing Complete!**")
                    st.balloons()

                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Questions Extracted", len(questions))
                    with col2:
                        st.metric("Questions in Cleaned QB", len(final_questions))
                    with col3:
                        st.metric(
                            "Answers Generated",
                            len([a for a in answers if a.get("success")]),
                        )

                except Exception as e:
                    st.error(f"❌ **Error during processing:** {str(e)}")
                    logger.error(f"Processing error: {str(e)}")
                    import traceback

                    logger.error(traceback.format_exc())

                    # Reset processing state
                    st.session_state.processing_in_progress = False
                    st.session_state.processing_complete = False

        st.markdown("---")

        # ============================================================================
        # SECTION 4: DOWNLOAD SECTION (Always Visible if processing complete)
        # ============================================================================
        if st.session_state.processing_complete:
            st.markdown("### 📥 Step 4: Download Your Files")

            cleaned_path = st.session_state.get("cleaned_qb_path")
            booklet_path = st.session_state.get("answer_booklet_path")

            # Verify files exist
            if (
                cleaned_path
                and booklet_path
                and Path(cleaned_path).exists()
                and Path(booklet_path).exists()
            ):

                st.success("✅ **Your files are ready for download!**")
                st.info(
                    "💡 **Tip:** You can download both files. Clicking download won't clear this section."
                )

                # Create download buttons in columns
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    # Read cleaned QB file
                    with open(cleaned_path, "rb") as f:
                        cleaned_data = f.read()

                    st.download_button(
                        label="📄 Download Cleaned Question Bank",
                        data=cleaned_data,
                        file_name=Path(cleaned_path).name,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="download_cleaned_qb_persistent",
                        help="Download the standardized question bank",
                        use_container_width=True,
                    )

                with col2:
                    # Read booklet file
                    with open(booklet_path, "rb") as f:
                        booklet_data = f.read()

                    mime_type = (
                        "application/pdf"
                        if booklet_path.endswith(".pdf")
                        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )

                    st.download_button(
                        label="📘 Download Answer Booklet",
                        data=booklet_data,
                        file_name=Path(booklet_path).name,
                        mime=mime_type,
                        key="download_answer_booklet_persistent",
                        help="Download the complete answer booklet",
                        use_container_width=True,
                    )

                with col3:
                    if st.button(
                        "🔄 Process New File",
                        use_container_width=True,
                        help="Clear results and upload a new file",
                    ):
                        # Clear all session state related to processing
                        keys_to_clear = [
                            "processing_complete",
                            "processing_in_progress",
                            "extracted_questions",
                            "cleaned_qb_path",
                            "answer_booklet_path",
                            "generated_answers",
                            "uploaded_file_name",
                            "uploaded_file_path",
                            "file_validated",
                        ]
                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()

                # Show file details in expander
                with st.expander("📊 File Information"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Cleaned Question Bank:**")
                        st.write(f"📝 Name: `{Path(cleaned_path).name}`")
                        st.write(
                            f"📦 Size: `{Path(cleaned_path).stat().st_size / 1024:.1f} KB`"
                        )
                        st.write(f"📁 Location: `{cleaned_path}`")

                    with col2:
                        st.markdown("**Answer Booklet:**")
                        st.write(f"📝 Name: `{Path(booklet_path).name}`")
                        st.write(
                            f"📦 Size: `{Path(booklet_path).stat().st_size / 1024:.1f} KB`"
                        )
                        st.write(f"📁 Location: `{booklet_path}`")

            else:
                st.warning(
                    "⚠️ **Files not found.** They may have been deleted. Please process a new question bank."
                )
                if st.button("🔄 Start Over"):
                    keys_to_clear = [
                        "processing_complete",
                        "processing_in_progress",
                        "extracted_questions",
                        "cleaned_qb_path",
                        "answer_booklet_path",
                        "generated_answers",
                        "uploaded_file_name",
                        "uploaded_file_path",
                        "file_validated",
                    ]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

    else:
        # No file uploaded yet
        st.info("👆 **Please upload a question bank file to get started**")

        # Show helpful tips
        with st.expander("💡 Tips for Best Results"):
            st.markdown(
                """
            ### 📋 Supported File Formats
            - **PDF** - Digital or scanned
            - **Word (.docx)** - Text-based question banks
            - **Text (.txt)** - Plain text format
            - **Images** - JPG, PNG, JPEG (for scanned documents)
            
            ### ✅ Best Practices
            1. **Clear formatting**: Ensure questions are clearly numbered
            2. **Mark allocation**: Include marks for each question (e.g., "5 marks")
            3. **File size**: Keep files under 10MB for optimal processing
            4. **Quality**: For scanned documents, use high-resolution images
            
            ### 🚀 Processing Steps
            1. Upload your question bank file
            2. Configure RAG and output format options
            3. Click "Start Processing"
            4. Wait for completion (usually 1-5 minutes)
            5. Download both cleaned QB and answer booklet
            
            ### 🔧 Troubleshooting
            - **Extraction issues**: Try different file format
            - **Slow processing**: Normal for large files (100+ questions)
            - **Missing questions**: Check original file formatting
            - **API errors**: System will automatically retry with fallback providers
            """
            )

elif page == "🧠 RAG Module":
    st.title("🧠 RAG Module - Study Materials")
    st.markdown("Upload study materials to generate personalized answers")

    st.markdown("---")

    # Collection name
    collection_name = st.text_input(
        "Collection Name",
        value="default",
        help="Name for this collection of study materials",
    )

    # File upload (multiple files)
    uploaded_files = st.file_uploader(
        "Upload Study Materials",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload your course notes, textbooks, or study materials",
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")

        if st.button("🚀 Process & Index Materials", type="primary"):
            with st.spinner("Processing study materials..."):

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Save files
                    status_text.text("💾 Saving uploaded files...")
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = asyncio.run(
                            FileHandler.save_uploaded_file(
                                uploaded_file, str(settings.UPLOAD_DIR)
                            )
                        )
                        file_paths.append(file_path)

                    progress_bar.progress(20)

                    # Process documents
                    status_text.text("📖 Extracting and chunking text...")
                    doc_processor = DocumentProcessor()

                    async def update_progress(progress):
                        progress_bar.progress(int(20 + (progress * 30)))

                    documents = asyncio.run(
                        doc_processor.process_multiple_documents(
                            file_paths, progress_callback=update_progress
                        )
                    )

                    progress_bar.progress(50)
                    status_text.text(f"✅ Processed {len(documents)} chunks")

                    # Generate embeddings
                    status_text.text("🔢 Generating embeddings...")
                    embedder = Embedder()

                    texts = [doc["text"] for doc in documents]
                    embeddings = asyncio.run(embedder.embed_documents(texts))

                    progress_bar.progress(70)

                    # Create vector store
                    status_text.text("💾 Creating vector database...")
                    vector_store = VectorStore(collection_name)

                    asyncio.run(vector_store.create_index(embeddings, documents))
                    asyncio.run(vector_store.save())

                    progress_bar.progress(90)

                    # Initialize retriever
                    st.session_state.rag_retriever = RAGRetriever(collection_name)
                    asyncio.run(st.session_state.rag_retriever.initialize())

                    progress_bar.progress(100)
                    status_text.text("✅ RAG module initialized!")

                    st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                    st.markdown(f"### ✅ RAG Module Ready!")
                    st.markdown(f"- **Files Processed**: {len(uploaded_files)}")
                    st.markdown(f"- **Text Chunks**: {len(documents)}")
                    st.markdown(f"- **Vector Dimension**: {embeddings.shape[1]}")
                    st.markdown(f"- **Collection**: {collection_name}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.info("💡 You can now use RAG in the Answer Generation module!")

                except Exception as e:
                    st.markdown(
                        f"<div class='error-box'>❌ Error: {str(e)}</div>",
                        unsafe_allow_html=True,
                    )
                    logger.error(f"RAG processing error: {str(e)}")

    # Show current RAG status
    st.markdown("---")
    st.markdown("### 📊 Current RAG Status")

    if (
        st.session_state.rag_retriever
        and st.session_state.rag_retriever.vector_store.index
    ):
        retriever = st.session_state.rag_retriever
        st.success(
            f"✅ RAG Active - {retriever.vector_store.index.ntotal} vectors indexed"
        )

        # Test query
        test_query = st.text_input(
            "Test RAG Retrieval", placeholder="Enter a question to test..."
        )
        if test_query:
            with st.spinner("Retrieving context..."):
                context = asyncio.run(retriever.retrieve_context(test_query, k=3))
                if context:
                    st.markdown("**Retrieved Context:**")
                    st.text_area("Context", context, height=300)
                else:
                    st.warning("No relevant context found")
    else:
        st.info("⭕ No RAG collection active. Upload study materials above.")

elif page == "🎯 QB Generator":
    st.title("🎯 Question Bank Generator")
    st.markdown("Generate custom question banks from course materials")

    st.markdown("---")

    # Upload course material
    uploaded_file = st.file_uploader(
        "Upload Course Material",
        type=["pdf", "docx", "txt"],
        help="Upload lecture notes, textbook chapters, or course content",
    )

    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Configuration
        st.markdown("### ⚙️ Generation Settings")

        col1, col2 = st.columns(2)

        with col1:
            num_questions = st.number_input(
                "Number of Questions", min_value=5, max_value=100, value=20
            )
            difficulty = st.selectbox(
                "Difficulty Level", ["Easy", "Medium", "Hard", "Mixed"]
            )

        with col2:
            question_types = st.multiselect(
                "Question Types",
                ["Short Answer", "Long Answer", "Numerical", "MCQ"],
                default=["Short Answer", "Long Answer"],
            )

        # Marks distribution
        st.markdown("**Marks Distribution**")
        col1, col2, col3 = st.columns(3)

        with col1:
            marks_2 = st.number_input(
                "2-mark questions", min_value=0, max_value=50, value=5
            )
        with col2:
            marks_5 = st.number_input(
                "5-mark questions", min_value=0, max_value=50, value=10
            )
        with col3:
            marks_10 = st.number_input(
                "10-mark questions", min_value=0, max_value=50, value=5
            )

        marks_distribution = {"2": marks_2, "5": marks_5, "10": marks_10}

        # Topics (optional)
        topics_input = st.text_input(
            "Specific Topics (comma-separated, optional)",
            placeholder="e.g., Data Structures, Algorithms, Database",
        )
        topics = [t.strip() for t in topics_input.split(",")] if topics_input else None

        # Generate button
        if st.button("🎯 Generate Question Bank", type="primary"):
            with st.spinner("Generating questions..."):

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Save and extract text
                    status_text.text("📖 Reading course material...")
                    file_path = asyncio.run(
                        FileHandler.save_uploaded_file(
                            uploaded_file, str(settings.UPLOAD_DIR)
                        )
                    )

                    doc_processor = DocumentProcessor()
                    course_text = asyncio.run(doc_processor._extract_text(file_path))

                    progress_bar.progress(30)

                    # Generate questions
                    status_text.text("🎯 Generating questions...")
                    qb_generator = QuestionBankGenerator(st.session_state.llm_router)

                    questions = asyncio.run(
                        qb_generator.generate_questions(
                            course_text,
                            num_questions=num_questions,
                            difficulty=difficulty,
                            question_types=question_types,
                            marks_distribution=marks_distribution,
                            topics=topics,
                        )
                    )

                    progress_bar.progress(70)

                    # Save question bank
                    status_text.text("💾 Saving question bank...")
                    output_path = str(
                        settings.OUTPUT_DIR
                        / f"{Path(uploaded_file.name).stem} - Generated QB.docx"
                    )

                    saved_path = asyncio.run(
                        qb_generator.save_question_bank(
                            questions,
                            output_path,
                            f"Generated Question Bank - {Path(uploaded_file.name).stem}",
                        )
                    )

                    progress_bar.progress(100)
                    status_text.text("✅ Generation complete!")

                    # Success message
                    st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                    st.markdown(f"### ✅ Question Bank Generated!")
                    st.markdown(f"- **Total Questions**: {len(questions)}")
                    st.markdown(f"- **Difficulty**: {difficulty}")
                    st.markdown(f"- **Types**: {', '.join(question_types)}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Download button
                    with open(saved_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download Generated QB",
                            f,
                            file_name=Path(saved_path).name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )

                    # Preview
                    st.markdown("### 👀 Preview")
                    for i, q in enumerate(questions[:5], 1):
                        with st.expander(f"Question {i}"):
                            st.markdown(f"**Question**: {q.get('question_text')}")
                            st.markdown(
                                f"**Type**: {q.get('question_type')} | **Marks**: {q.get('marks')} | **Topic**: {q.get('topic')}"
                            )

                    if len(questions) > 5:
                        st.info(f"... and {len(questions) - 5} more questions")

                except Exception as e:
                    st.markdown(
                        f"<div class='error-box'>❌ Error: {str(e)}</div>",
                        unsafe_allow_html=True,
                    )
                    logger.error(f"QB generation error: {str(e)}")

elif page == "⚙️ Settings":
    st.title("⚙️ System Settings")
    st.markdown("Configure API keys and system parameters")

    st.markdown("---")

    # === API CONFIGURATION ===
    st.markdown("### 🔑 API Configuration")

    # Create tabs for different provider tiers
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔥 Primary APIs", "⚡ Secondary APIs", "🔮 Optional APIs", "📊 Status & Info"]
    )

    with tab1:
        st.markdown("#### Primary Providers (Required)")
        st.info("💡 These providers are essential for core functionality")

        # Groq
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Groq API Key",
                value=settings.GROQ_API_KEY,
                type="password",
                help="Primary text generation - Fast inference",
                key="groq_key_display",
            )
        with col2:
            if settings.GROQ_API_KEY:
                st.success("✅ Active")
            else:
                st.error("❌ Missing")

        if not settings.GROQ_API_KEY:
            st.markdown(
                """
            **Get Groq API Key:**
            1. Go to [console.groq.com/keys](https://console.groq.com/keys)
            2. Sign up with email (free)
            3. Create API key
            4. Add to `.env` file: `GROQ_API_KEY=gsk_...`
            
            **Limits:** 1,000-14,400 requests/day depending on model
            """
            )

        st.markdown("---")

        # Gemini
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Google Gemini API Key",
                value=settings.GEMINI_API_KEY,
                type="password",
                help="Vision/OCR for image processing",
                key="gemini_key_display",
            )
        with col2:
            if settings.GEMINI_API_KEY:
                st.success("✅ Active")
            else:
                st.error("❌ Missing")

        if not settings.GEMINI_API_KEY:
            st.markdown(
                """
            **Get Gemini API Key:**
            1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
            2. Sign in with Google account
            3. Create API key
            4. Add to `.env` file: `GEMINI_API_KEY=AIza...`
            
            **Limits:** 15 requests/minute, 50 requests/day for free tier
            """
            )

    with tab2:
        st.markdown("#### Secondary Providers (Highly Recommended)")
        st.info("💡 Add these for better reliability and 30x more capacity")

        # Cerebras
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Cerebras API Key",
                value=settings.CEREBRAS_API_KEY,
                type="password",
                help="Ultra-fast inference - 14,400 req/day",
                key="cerebras_key_display",
            )
        with col2:
            if settings.CEREBRAS_API_KEY:
                st.success("✅ Active")
            else:
                st.warning("⭕ Optional")

        if not settings.CEREBRAS_API_KEY:
            st.markdown(
                """
            **Get Cerebras API Key (5 min setup):**
            1. Go to [cloud.cerebras.ai](https://cloud.cerebras.ai/)
            2. Sign up with email
            3. Navigate to API Keys
            4. Create new key
            5. Add to `.env` file: `CEREBRAS_API_KEY=csk_...`
            
            **Benefit:** 14,400 requests/day, 16x faster than standard, 1M tokens/day
            """
            )

        st.markdown("---")

        # Mistral
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Mistral API Key",
                value=settings.MISTRAL_API_KEY,
                type="password",
                help="High-volume fallback - 1B tokens/month!",
                key="mistral_key_display",
            )
        with col2:
            if settings.MISTRAL_API_KEY:
                st.success("✅ Active")
            else:
                st.warning("⭕ Optional")

        if not settings.MISTRAL_API_KEY:
            st.markdown(
                """
            **Get Mistral API Key (10 min setup - requires phone):**
            1. Go to [console.mistral.ai](https://console.mistral.ai/)
            2. Sign up (phone verification required)
            3. Choose "Experiment" plan (free)
            4. Opt-in to data training for free tier
            5. Create API key
            6. Add to `.env` file: `MISTRAL_API_KEY=...`
            
            **Benefit:** 1 BILLION tokens/month, 60 requests/minute, 500k tokens/minute
            """
            )

    with tab3:
        st.markdown("#### Optional Providers")
        st.info("💡 Additional fallbacks for maximum reliability")

        # OpenRouter
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "OpenRouter API Key",
                value=settings.OPENROUTER_API_KEY,
                type="password",
                help="Multi-model access - 30+ models",
                key="openrouter_key_display",
            )
        with col2:
            if settings.OPENROUTER_API_KEY:
                st.success("✅ Active")
            else:
                st.info("⭕ Optional")

        if not settings.OPENROUTER_API_KEY:
            st.markdown(
                """
            **Get OpenRouter API Key:**
            1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
            2. Sign up with email
            3. Create API key (free)
            4. Add to `.env` file: `OPENROUTER_API_KEY=sk-or-...`
            
            **Limits:** 20 req/min, 50 req/day (upgradeable to 1000 with $10 lifetime)
            """
            )

        st.markdown("---")

        # Cohere
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "Cohere API Key",
                value=settings.COHERE_API_KEY,
                type="password",
                help="RAG embeddings and generation",
                key="cohere_key_display",
            )
        with col2:
            if settings.COHERE_API_KEY:
                st.success("✅ Active")
            else:
                st.info("⭕ Optional")

        if not settings.COHERE_API_KEY:
            st.markdown(
                """
            **Get Cohere API Key:**
            1. Go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)
            2. Sign up with email
            3. Create API key
            4. Add to `.env` file: `COHERE_API_KEY=...`
            
            **Benefit:** High-quality embeddings for RAG, 1000 requests/month
            """
            )

        st.markdown("---")

        # HuggingFace (Legacy)
        st.markdown("#### Legacy Providers (Deprecated)")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_input(
                "HuggingFace API Key (Legacy)",
                value=settings.HUGGINGFACE_API_KEY,
                type="password",
                help="Legacy support - Use Cerebras/Mistral instead",
                key="hf_key_display",
            )
        with col2:
            if settings.HUGGINGFACE_API_KEY:
                st.warning("⚠️ Legacy")
            else:
                st.info("⭕ Optional")

        st.warning(
            "⚠️ **Note:** HuggingFace Inference API has limited compatibility. Recommended to use Cerebras or Mistral instead."
        )

    with tab4:
        st.markdown("#### 📊 API Status & Capacity")

        # Get current provider status
        if "llm_router" in st.session_state:
            stats = st.session_state.llm_router.get_stats()

            st.markdown("##### Active Providers")

            # Create metrics
            cols = st.columns(
                len(stats["available_providers"]) if stats["available_providers"] else 1
            )

            for idx, provider in enumerate(stats["available_providers"]):
                with cols[idx]:
                    usage = stats["usage_stats"].get(provider, 0)
                    st.metric(
                        label=provider.capitalize(),
                        value="Active ✅",
                        delta=f"{usage} requests used",
                    )

            st.markdown("---")

            # Capacity analysis
            st.markdown("##### 📈 Current vs. Potential Capacity")

            current_providers = len(stats["available_providers"])

            if current_providers >= 4:
                capacity_status = "🔥 **Excellent** - Production Ready"
                capacity_color = "success"
                daily_capacity = "60,000+"
                concurrent_users = "100+"
            elif current_providers >= 2:
                capacity_status = "✅ **Good** - Reliable for testing"
                capacity_color = "normal"
                daily_capacity = "15,000-30,000"
                concurrent_users = "20-50"
            elif current_providers >= 1:
                capacity_status = "⚠️ **Basic** - Single point of failure"
                capacity_color = "warning"
                daily_capacity = "1,000-14,400"
                concurrent_users = "5-10"
            else:
                capacity_status = "❌ **None** - No providers configured"
                capacity_color = "error"
                daily_capacity = "0"
                concurrent_users = "0"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", capacity_status)

            with col2:
                st.metric("Daily Capacity", f"{daily_capacity} requests")

            with col3:
                st.metric("Concurrent Users", concurrent_users)

            st.markdown("---")

            # Recommendations
            st.markdown("##### 💡 Recommendations")

            missing_providers = []

            if not settings.GROQ_API_KEY:
                missing_providers.append(
                    "❗ **Groq** - REQUIRED for core functionality"
                )

            if not settings.GEMINI_API_KEY:
                missing_providers.append(
                    "❗ **Gemini** - REQUIRED for image/OCR processing"
                )

            if not settings.CEREBRAS_API_KEY:
                missing_providers.append(
                    "⭐ **Cerebras** - Highly recommended (14.4k req/day, 5 min setup)"
                )

            if not settings.MISTRAL_API_KEY:
                missing_providers.append(
                    "⭐ **Mistral** - Highly recommended (1B tokens/month, 10 min setup)"
                )

            if missing_providers:
                st.info("**Add these providers to improve reliability:**")
                for provider in missing_providers:
                    st.markdown(f"- {provider}")
            else:
                st.success(
                    "🎉 **All recommended providers configured!** Your setup is production-ready."
                )

        else:
            st.error("LLM Router not initialized. Please restart the application.")

    st.markdown("---")

    # === RAG CONFIGURATION ===
    st.markdown("### 🧠 RAG Configuration")

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=settings.CHUNK_SIZE,
            step=100,
            help="Size of text chunks for RAG processing",
        )

        top_k = st.number_input(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=settings.TOP_K_RESULTS,
            help="Number of relevant chunks to retrieve",
        )

    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=settings.CHUNK_OVERLAP,
            step=50,
            help="Overlap between consecutive chunks",
        )

        st.text_input(
            "Embedding Model",
            value=settings.HF_EMBEDDING_MODEL,
            help="HuggingFace model for embeddings",
            disabled=True,
        )

    st.info("💡 Changes require application restart to take effect")

    st.markdown("---")

    # === FILE SETTINGS ===
    st.markdown("### 📁 File Upload Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=50,
            value=settings.MAX_FILE_SIZE_MB,
            help="Maximum allowed file size",
        )

    with col2:
        st.text_input(
            "Allowed Extensions",
            value=", ".join(settings.ALLOWED_EXTENSIONS),
            help="Comma-separated list of allowed file extensions",
            disabled=True,
        )

    st.markdown("---")

    # === SYSTEM MAINTENANCE ===
    st.markdown("### 🗑️ System Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧹 Clear Upload Files (24h+)", use_container_width=True):
            asyncio.run(
                FileHandler.cleanup_old_files(
                    str(settings.UPLOAD_DIR), max_age_hours=24
                )
            )
            st.success("✅ Old upload files cleared (24h+)")

    with col2:
        if st.button("🧹 Clear Output Files (24h+)", use_container_width=True):
            asyncio.run(
                FileHandler.cleanup_old_files(
                    str(settings.OUTPUT_DIR), max_age_hours=24
                )
            )
            st.success("✅ Old output files cleared (24h+)")

    st.markdown("---")

    # === USAGE GUIDELINES ===
    with st.expander("📖 API Usage Guidelines"):
        st.markdown(
            """
        ### Best Practices
        
        #### Provider Priority Order
        1. **Groq** - Use for most text generation (fast, reliable)
        2. **Cerebras** - Fallback for text (ultra-fast, high capacity)
        3. **Mistral** - High-volume processing (1B tokens/month)
        4. **OpenRouter** - Emergency fallback (30+ models)
        5. **Gemini** - Image/OCR only (save daily quota)
        
        #### Capacity Planning
        - **Minimum Setup (Groq + Gemini):** 5-10 simultaneous users
        - **Recommended Setup (+ Cerebras + Mistral):** 50+ simultaneous users
        - **Full Setup (All providers):** 100+ simultaneous users
        
        #### Cost Optimization
        - All providers offer generous free tiers
        - No credit card required for any free tier
        - Mistral requires phone verification (one-time)
        - Total cost: $0/month for all free tiers combined
        
        #### Rate Limits
        - Groq: 30 req/min, 1,000 req/day (Llama 3.3 70B)
        - Cerebras: 30 req/min, 14,400 req/day per model
        - Gemini: 15 req/min, 50 req/day (vision models)
        - Mistral: 60 req/min, 500k tokens/min, 1B tokens/month
        - OpenRouter: 20 req/min, 50 req/day
        
        #### Troubleshooting
        - If one provider fails, the system automatically tries the next
        - Check "API Status" tab to see which providers are active
        - Add more providers to improve reliability
        - Restart app after adding new API keys
        """
        )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "EduSolve v2.0 | Multi-Provider Architecture | Powered by Free LLM APIs"
    "</div>",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    # Only log on first run
    if "app_initialized" not in st.session_state:
        logger.info("EduSolve application started")
        st.session_state.app_initialized = True
