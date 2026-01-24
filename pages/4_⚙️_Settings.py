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

# Hide password reveal button (eye icon)
st.markdown(
    """
    <style>
    input[type="password"] ~ button {
        display: none !important;
    }
    button[aria-label="Show password"] {
        display: none !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def get_api_key(key_name: str, default_value: str) -> str:
    """Get API key from session state or default settings"""
    session_key = f"custom_{key_name}"
    if session_key in st.session_state and st.session_state[session_key]:
        return st.session_state[session_key]
    return default_value if default_value else ""


def mask_key(key: str) -> str:
    """Mask API key for display"""
    if not key:
        return "Not configured"
    return f"{key[:10]}{'•' * (len(key) - 10) if len(key) > 10 else ''}"


def render_settings():
    st.title("⚙️ System Settings")
    st.markdown("Configure API keys and system parameters")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔥 Primary APIs", "⚡ Secondary APIs", "🔮 Optional APIs", "📊 Status & Info"]
    )

    with tab1:
        st.markdown("#### Primary Providers (Required)")

        # Groq API Key
        groq_custom = st.text_input(
            "Groq API Key",
            value=st.session_state.get("custom_GROQ_API_KEY", ""),
            type="password",
            key="custom_GROQ_API_KEY",
            help="Enter your own Groq API key or leave blank to use default",
        )
        groq_key = get_api_key("GROQ_API_KEY", settings.GROQ_API_KEY)
        if st.session_state.get("custom_GROQ_API_KEY"):
            st.caption("✅ Using your custom Groq API key")
        else:
            st.caption("� Using default Groq API key")

        # Gemini API Key
        gemini_custom = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.get("custom_GEMINI_API_KEY", ""),
            type="password",
            key="custom_GEMINI_API_KEY",
            help="Enter your own Gemini API key or leave blank to use default",
        )
        gemini_key = get_api_key("GEMINI_API_KEY", settings.GEMINI_API_KEY)
        if st.session_state.get("custom_GEMINI_API_KEY"):
            st.caption("✅ Using your custom Gemini API key")
        else:
            st.caption("� Using default Gemini API key")

        st.info(
            "💡 Leave fields empty to use default API keys. Enter your own keys to override."
        )

    with tab2:
        st.markdown("#### Secondary Providers")

        # Cerebras API Key
        cerebras_custom = st.text_input(
            "Cerebras API Key",
            value=st.session_state.get("custom_CEREBRAS_API_KEY", ""),
            type="password",
            key="custom_CEREBRAS_API_KEY",
            help="Enter your own Cerebras API key or leave blank to use default",
        )
        cerebras_key = get_api_key("CEREBRAS_API_KEY", settings.CEREBRAS_API_KEY)
        if st.session_state.get("custom_CEREBRAS_API_KEY"):
            st.caption("✅ Using your custom Cerebras API key")
        else:
            st.caption("� Using default Cerebras API key")

        # Mistral API Key
        mistral_custom = st.text_input(
            "Mistral API Key",
            value=st.session_state.get("custom_MISTRAL_API_KEY", ""),
            type="password",
            key="custom_MISTRAL_API_KEY",
            help="Enter your own Mistral API key or leave blank to use default",
        )
        mistral_key = get_api_key("MISTRAL_API_KEY", settings.MISTRAL_API_KEY)
        if st.session_state.get("custom_MISTRAL_API_KEY"):
            st.caption("✅ Using your custom Mistral API key")
        else:
            st.caption("� Using default Mistral API key")

    with tab3:
        st.markdown("#### Optional Providers")

        # OpenRouter API Key
        openrouter_custom = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.get("custom_OPENROUTER_API_KEY", ""),
            type="password",
            key="custom_OPENROUTER_API_KEY",
            help="Enter your own OpenRouter API key or leave blank to use default",
        )
        openrouter_key = get_api_key("OPENROUTER_API_KEY", settings.OPENROUTER_API_KEY)
        if st.session_state.get("custom_OPENROUTER_API_KEY"):
            st.caption("✅ Using your custom OpenRouter API key")
        else:
            st.caption("� Using default OpenRouter API key")

        # Cohere API Key
        cohere_custom = st.text_input(
            "Cohere API Key",
            value=st.session_state.get("custom_COHERE_API_KEY", ""),
            type="password",
            key="custom_COHERE_API_KEY",
            help="Enter your own Cohere API key or leave blank to use default",
        )
        cohere_key = get_api_key("COHERE_API_KEY", settings.COHERE_API_KEY)
        if st.session_state.get("custom_COHERE_API_KEY"):
            st.caption("✅ Using your custom Cohere API key")
        else:
            st.caption("� Using default Cohere API key")

    with tab4:
        st.markdown("#### 📊 API Status")
        if "llm_router" in st.session_state:
            stats = st.session_state.llm_router.get_stats()
            for provider in stats["available_providers"]:
                st.write(
                    f"✅ {provider.capitalize()}: {stats['usage_stats'].get(provider, 0)} requests used"
                )
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
