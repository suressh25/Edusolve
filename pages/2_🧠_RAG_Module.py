import streamlit as st
import asyncio
from services.rag_service import RAGService
from config.settings import settings
from utils.logger import get_logger
from utils.bootstrap import setup_app, apply_custom_css
from components.sidebar import render_sidebar

st.set_page_config(page_title="RAG Module - EduSolve", page_icon="🧠", layout="wide")
setup_app()
apply_custom_css()
render_sidebar()

logger = get_logger()

def render_rag_module():
    st.title("🧠 RAG Module - Study Materials")
    st.markdown("Upload study materials to generate personalized answers")
    st.markdown("---")

    collection_name = st.text_input("Collection Name", value="default")
    uploaded_files = st.file_uploader(
        "Upload Study Materials",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        if st.button("🚀 Process & Index Materials", type="primary"):
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(p, text):
                    progress_bar.progress(p)
                    status_text.text(text)

                try:
                    cohere_client = st.session_state.llm_router.cohere if hasattr(st.session_state.llm_router, 'cohere') else None
                    service = RAGService(cohere_client=cohere_client)
                    st.session_state.rag_retriever = asyncio.run(service.initialize_rag(
                        uploaded_files, collection_name, progress_callback=update_progress
                    ))
                    st.success("✅ RAG module initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"RAG error: {str(e)}", exc_info=True)

    st.markdown("---")
    st.markdown("### 📊 Current RAG Status")

    if st.session_state.get("rag_retriever") and st.session_state.rag_retriever.vector_store.index:
        retriever = st.session_state.rag_retriever
        st.success(f"✅ RAG Active - {retriever.vector_store.index.ntotal} vectors indexed")
        
        test_query = st.text_input("Test RAG Retrieval")
        if test_query:
            with st.spinner("Retrieving..."):
                context = asyncio.run(retriever.retrieve_context(test_query, k=3))
                if context:
                    st.text_area("Retrieved Context", context, height=300)
                else:
                    st.warning("No relevant context found")
    else:
        st.info("⭕ No RAG collection active. Upload study materials above.")

render_rag_module()
