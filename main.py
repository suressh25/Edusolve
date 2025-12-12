"""
EduSolve - Main Streamlit Application
GenAI-based Automated Answer Generation System
"""

import streamlit as st
import asyncio
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import settings
from utils.logger import logger
from utils.file_handler import FileHandler

# Import API clients
from api.groq_client import GroqClient
from api.gemini_client import GeminiClient
from api.huggingface_client import HuggingFaceClient
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

# Custom CSS
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
    """Initialize session state variables"""
    if "llm_router" not in st.session_state:
        groq_client = GroqClient()
        gemini_client = GeminiClient()
        hf_client = HuggingFaceClient()
        st.session_state.llm_router = LLMRouter(groq_client, gemini_client, hf_client)

    if "rag_retriever" not in st.session_state:
        st.session_state.rag_retriever = None

    if "extracted_questions" not in st.session_state:
        st.session_state.extracted_questions = None

    if "generated_answers" not in st.session_state:
        st.session_state.generated_answers = None


init_session_state()

# Sidebar
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x80/1f77b4/ffffff?text=EduSolve",
        use_container_width=True,
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
    st.markdown("### 📊 System Status")

    # Check API availability
    api_status = []
    if settings.GROQ_API_KEY:
        api_status.append("✅ Groq API")
    else:
        api_status.append("❌ Groq API")

    if settings.GEMINI_API_KEY:
        api_status.append("✅ Gemini API")
    else:
        api_status.append("❌ Gemini API")

    if settings.HUGGINGFACE_API_KEY:
        api_status.append("✅ HuggingFace API")
    else:
        api_status.append("❌ HuggingFace API")

    for status in api_status:
        st.markdown(status)

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

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Question Bank",
        type=settings.ALLOWED_EXTENSIONS,
        help="Supports PDF, DOCX, TXT, and image files",
    )

    if uploaded_file:
        try:
            FileHandler.validate_file(uploaded_file)
            st.success(f"✅ File validated: {uploaded_file.name}")

            # Processing options
            col1, col2 = st.columns(2)

            with col1:
                use_rag = st.checkbox(
                    "Use RAG (Study Materials)",
                    value=False,
                    disabled=st.session_state.rag_retriever is None,
                    help="Generate answers using your uploaded study materials",
                )

            with col2:
                output_format = st.selectbox(
                    "Output Format", ["Word (.docx)", "PDF (.pdf)"]
                )

            # Process button
            if st.button("🚀 Process Question Bank", type="primary"):
                with st.spinner("Processing question bank..."):

                    # Save uploaded file
                    file_path = asyncio.run(
                        FileHandler.save_uploaded_file(
                            uploaded_file, str(settings.UPLOAD_DIR)
                        )
                    )

                    # Determine file type
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
                                        image_extractor.extract_from_scanned_pdf(
                                            file_path
                                        )
                                    )
                                else:
                                    # Digital PDF
                                    raw_text = asyncio.run(
                                        text_extractor.extract_text_from_file(file_path)
                                    )
                                    questions = asyncio.run(
                                        text_extractor.extract_questions_with_llm(
                                            raw_text
                                        )
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
                                    / f"{Path(uploaded_file.name).stem} - Cleaned.docx"
                                ),
                                Path(uploaded_file.name).stem,
                            )
                        )

                        progress_bar.progress(60)

                        # Step 3: Generate answers
                        status_text.text("💡 Generating answers...")

                        answer_generator = AnswerGenerator(
                            st.session_state.llm_router,
                            st.session_state.rag_retriever if use_rag else None,
                        )

                        async def update_progress(progress):
                            progress_bar.progress(int(60 + (progress * 30)))

                        answers = asyncio.run(
                            answer_generator.generate_batch_answers(
                                questions,
                                use_rag=use_rag,
                                batch_size=5,
                                progress_callback=update_progress,
                            )
                        )

                        st.session_state.generated_answers = answers

                        progress_bar.progress(90)
                        status_text.text("📄 Compiling answer booklet...")

                        # Step 4: Compile booklet
                        booklet_compiler = BookletCompiler()

                        if "PDF" in output_format:
                            output_path = str(
                                settings.OUTPUT_DIR
                                / f"{Path(uploaded_file.name).stem} Answers.pdf"
                            )
                            final_path = asyncio.run(
                                booklet_compiler.compile_to_pdf(
                                    answers, output_path, Path(uploaded_file.name).stem
                                )
                            )
                        else:
                            output_path = str(
                                settings.OUTPUT_DIR
                                / f"{Path(uploaded_file.name).stem} Answers.docx"
                            )
                            final_path = asyncio.run(
                                booklet_compiler.compile_to_word(
                                    answers, output_path, Path(uploaded_file.name).stem
                                )
                            )

                        progress_bar.progress(100)
                        status_text.text("✅ Processing complete!")

                        # Success message
                        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                        st.markdown(f"### ✅ Processing Complete!")
                        st.markdown(f"- **Questions Extracted**: {len(questions)}")
                        st.markdown(
                            f"- **Answers Generated**: {len([a for a in answers if a.get('success')])}"
                        )
                        st.markdown(f"- **Cleaned QB**: Available for download")
                        st.markdown(f"- **Answer Booklet**: Ready!")
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Download buttons
                        st.markdown("### 📥 Downloads")

                        col1, col2 = st.columns(2)

                        with col1:
                            with open(cleaned_qb_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Cleaned QB",
                                    f,
                                    file_name=Path(cleaned_qb_path).name,
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                )

                        with col2:
                            with open(final_path, "rb") as f:
                                mime_type = (
                                    "application/pdf"
                                    if final_path.endswith(".pdf")
                                    else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                                st.download_button(
                                    "⬇️ Download Answer Booklet",
                                    f,
                                    file_name=Path(final_path).name,
                                    mime=mime_type,
                                )

                    except Exception as e:
                        st.markdown(
                            f"<div class='error-box'>❌ Error: {str(e)}</div>",
                            unsafe_allow_html=True,
                        )
                        logger.error(f"Processing error: {str(e)}")

        except ValueError as e:
            st.error(f"❌ {str(e)}")

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
    st.title("⚙️ Settings & Configuration")
    st.markdown("Configure API keys and system settings")

    st.markdown("---")

    st.markdown("### 🔑 API Keys Configuration")
    st.info("💡 All APIs have generous free tiers. Get your keys from the links below.")

    # Groq
    st.markdown("#### Groq API")
    st.markdown("[Get Groq API Key](https://console.groq.com) - 30 RPM, 14400 TPM free")
    groq_key = st.text_input(
        "Groq API Key", value=settings.GROQ_API_KEY, type="password"
    )

    # Gemini
    st.markdown("#### Google Gemini API")
    st.markdown(
        "[Get Gemini API Key](https://ai.google.dev) - 15 RPM free, Vision capable"
    )
    gemini_key = st.text_input(
        "Gemini API Key", value=settings.GEMINI_API_KEY, type="password"
    )

    # HuggingFace
    st.markdown("#### HuggingFace API")
    st.markdown(
        "[Get HuggingFace API Key](https://huggingface.co/settings/tokens) - 300 requests/hour"
    )
    hf_key = st.text_input(
        "HuggingFace API Key", value=settings.HUGGINGFACE_API_KEY, type="password"
    )

    if st.button("💾 Save API Keys"):
        # Update .env file
        env_path = Path(__file__).parent / ".env"

        with open(env_path, "w") as f:
            f.write(f"GROQ_API_KEY={groq_key}\n")
            f.write(f"GEMINI_API_KEY={gemini_key}\n")
            f.write(f"HUGGINGFACE_API_KEY={hf_key}\n")

        st.success("✅ API keys saved! Please restart the application.")

    st.markdown("---")

    st.markdown("### 🧹 Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Upload Directory"):
            asyncio.run(
                FileHandler.cleanup_old_files(str(settings.UPLOAD_DIR), max_age_hours=0)
            )
            st.success("✅ Upload directory cleared")

    with col2:
        if st.button("🗑️ Clear Output Directory"):
            asyncio.run(
                FileHandler.cleanup_old_files(str(settings.OUTPUT_DIR), max_age_hours=0)
            )
            st.success("✅ Output directory cleared")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "EduSolve v1.0 | Built with Streamlit | Powered by Free LLM APIs"
    "</div>",
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    # Run cleanup on startup
    asyncio.run(
        FileHandler.cleanup_old_files(str(settings.UPLOAD_DIR), max_age_hours=24)
    )
    logger.info("EduSolve application started")
