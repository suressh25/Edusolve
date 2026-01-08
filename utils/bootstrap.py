import streamlit as st
import asyncio
from api.groq_client import GroqClient
from api.gemini_client import GeminiClient
from api.cerebras_client import CerebrasClient
from api.mistral_client import MistralClient
from api.openrouter_client import OpenRouterClient
from api.cohere_client import CohereClient
from api.llm_router import LLMRouter
from utils.logger import get_logger

logger = get_logger()

@st.cache_resource
def _get_llm_router():
    """Cached initialization of the LLM Router and its clients"""
    logger.info("Initializing LLM Router (Cached)...")
    
    # Initialize all clients
    groq_client = GroqClient()
    gemini_client = GeminiClient()
    
    optional_clients = {
        "cerebras": CerebrasClient,
        "mistral": MistralClient,
        "openrouter": OpenRouterClient,
        "cohere": CohereClient
    }
    
    initialized = {}
    for name, cls in optional_clients.items():
        try:
            client = cls()
            if hasattr(client, 'configured') and client.configured:
                initialized[name] = client
            else:
                initialized[name] = None
        except Exception as e:
            logger.warning(f"{name.capitalize()} client failed: {e}")
            initialized[name] = None

    return LLMRouter(
        groq_client,
        gemini_client,
        initialized["cerebras"],
        initialized["mistral"],
        initialized["openrouter"],
        initialized["cohere"]
    )


def setup_app():
    """Initializes session state and core components once"""
    
    # Initialize session state using cached router
    if "llm_router" not in st.session_state:
        st.session_state.llm_router = _get_llm_router()
        st.session_state.app_initialized = True

    if "rag_retriever" not in st.session_state:
        st.session_state.rag_retriever = None

def apply_custom_css():
    st.markdown("""
        <style>
            .main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #1f77b4; margin-bottom: 0.5rem; }
            .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
            .success-box { padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
            .error-box { padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        </style>
    """, unsafe_allow_html=True)
