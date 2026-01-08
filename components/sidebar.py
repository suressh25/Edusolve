import streamlit as st
from config.settings import settings

def render_sidebar():
    """Render the standard EduSolve sidebar"""
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #1f77b4;'>📚 EduSolve</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### 📊 API Status")
        if "llm_router" in st.session_state:
            stats = st.session_state.llm_router.get_stats()
            for provider in stats["available_providers"][:3]: # Show top 3
                st.markdown(f"✅ {provider.capitalize()}")
        else:
            st.markdown("❌ No providers available")
            
        st.markdown("---")
        if st.session_state.get("rag_retriever"):
            st.markdown("✅ RAG Enabled")
        else:
            st.markdown("⭕ RAG Disabled")
        
        st.markdown("---")
        st.caption("EduSolve v2.1 | Refactored")
