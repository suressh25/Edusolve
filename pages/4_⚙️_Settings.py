import streamlit as st
from config.settings import settings
from utils.file_handler import FileHandler
from utils.bootstrap import setup_app, apply_custom_css
from components.sidebar import render_sidebar
import asyncio

st.set_page_config(page_title="Settings - EduSolve", page_icon="⚙️", layout="wide")
setup_app()
apply_custom_css()
render_sidebar()

def render_settings():
    st.title("⚙️ System Settings")
    st.markdown("Configure API keys and system parameters")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Primary APIs", "⚡ Secondary APIs", "🔮 Optional APIs", "📊 Status & Info"])

    with tab1:
        st.markdown("#### Primary Providers (Required)")
        st.text_input("Groq API Key", value=settings.GROQ_API_KEY, type="password", disabled=True)
        st.text_input("Google Gemini API Key", value=settings.GEMINI_API_KEY, type="password", disabled=True)
        st.info("💡 Edit .env file to update keys permanently.")

    with tab2:
        st.markdown("#### Secondary Providers")
        st.text_input("Cerebras API Key", value=settings.CEREBRAS_API_KEY, type="password", disabled=True)
        st.text_input("Mistral API Key", value=settings.MISTRAL_API_KEY, type="password", disabled=True)

    with tab3:
        st.markdown("#### Optional Providers")
        st.text_input("OpenRouter API Key", value=settings.OPENROUTER_API_KEY, type="password", disabled=True)
        st.text_input("Cohere API Key", value=settings.COHERE_API_KEY, type="password", disabled=True)

    with tab4:
        st.markdown("#### 📊 API Status")
        if "llm_router" in st.session_state:
            stats = st.session_state.llm_router.get_stats()
            for provider in stats["available_providers"]:
                st.write(f"✅ {provider.capitalize()}: {stats['usage_stats'].get(provider, 0)} requests used")
        else:
            st.error("Router not initialized")

    st.markdown("---")
    st.markdown("### 🗑️ System Maintenance")
    col1, col2 = st.columns(2)
    if col1.button("🧹 Clear Uploads (24h+)", use_container_width=True):
        asyncio.run(FileHandler.cleanup_old_files(str(settings.UPLOAD_DIR), 24))
        st.success("Cleared uploads")
    if col2.button("🧹 Clear Outputs (24h+)", use_container_width=True):
        asyncio.run(FileHandler.cleanup_old_files(str(settings.OUTPUT_DIR), 24))
        st.success("Cleared outputs")

render_settings()
