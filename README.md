# EduSolve 🎓🚀

**EduSolve** is a professional, GenAI-powered academic automation suite designed to streamline educational workflows. By orchestrating a multi-provider LLM network and advanced Retrieval-Augmented Generation (RAG), EduSolve automates answer generation, creates structured curriculum materials, and simplifies complex document parsing.

## 🌟 Key Features

### 1. 🧠 Answer Generation Module

- **Hybrid Input System**: Supports both **File Uploads** (PDF, Docx, Images) and **Direct Text Entry** side-by-side.
- **Independent Processing Flows**: Decoupled streams for:
  - **Flow A (Cleaned QB)**: Instantly generates standardized, editable Word/PDF question banks.
  - **Flow B (Answer Booklet)**: AI-synthesis of mark-weighted answers with citations.
- **Dynamic Formatting**: Choose between **Word (.docx)** and **PDF (.pdf)** at the moment of download.

### 2. 📚 Advanced RAG Module

- **Enterprise-Grade Embeddings**: Support for **Cohere Embed v3** for superior retrieval and local academic models (Hugging Face) for privacy.
- **Knowledge Ingestion**: Index textbooks, lecture notes, and research papers into a persistence FAISS vector store.
- **Hallucination Guardrails**: Answers are grounded strictly in uploaded study materials.

### 3. 🎯 QB Generator Module (New)

- **Curriculum Synthesis**: Generate high-quality question banks directly from uploaded study materials.
- **Customizable Difficulty**: Tailor questions based on Bloom's Taxonomy or specific mark distributions.
- **Multi-Type Support**: Create MCQs, short answers, and long descriptive questions automatically.

### 🤖 Intelligent LLM Router

- **Failover Reliability**: Automatically routes between **Groq, Gemini, Cerebras, Mistral, OpenRouter, and Cohere**.
- **Optimized Performance**: Integrated **Streamlit Caching (`@st.cache_resource`)** for instant client initialization and model loading.
- **OCR Hybrid Engine**: Uses Gemini 1.5 Flash for vision extraction and Groq for high-speed text processing.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Core Logic**: Python (Service-Oriented Architecture)
- **Package Management**: [uv](https://github.com/astral-sh/uv) (Recommended) or pip
- **Document Processing**: PyMuPDF, ReportLab, python-docx
- **AI Engine**: Multi-Provider Router (Groq, Gemini, Cohere, Cerebras)

### 🤖 Model Orchestration

| Provider     | Model            | Role          | Priority   |
| :----------- | :--------------- | :------------ | :--------- |
| **Groq**     | `llama-3.1-70b`  | Primary Text  | 1          |
| **Cerebras** | `llama3.3-70b`   | Fast Fallback | 2          |
| **Cohere**   | `command-r-plus` | RAG / Text    | 3          |
| **Gemini**   | `flash-1.5`      | Vision / OCR  | 1 (Vision) |
| **Mistral**  | `mistral-small`  | High-Vol Text | 4          |

## 📂 Project Structure

```bash
Edusolve/
├── api/                     # LLM integration and routing logic
├── extraction/              # OCR and Document parsing (Vision/Text/Cleaner)
├── generation/              # AI Synthesis (Answers/Booklets)
├── question_generation/      # QB Generator logic
├── services/                # Business logic (Decoupled flows)
├── components/              # Reusable UI elements (Sidebar, etc.)
├── pages/                   # Streamlit multi-page application files
├── rag/                     # RAG implementation (Embedder, Retriever, Processor)
├── utils/                   # Performance (Caching, Bootstrapping, Logger)
├── config/                  # Global Settings and Model Configs
├── Research/                # Reference papers and academic materials
├── tests/                   # Verification and unit tests
├── outputs/                 # Final Word/PDF generations
├── vector_stores/           # FAISS persistence layer
└── main.py                  # Entry Point
```

## 🚀 Getting Started

### Installation

We recommend using [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1.  **Clone & Environment**

    ```bash
    git clone https://github.com/yourusername/Edusolve.git
    cd Edusolve
    # Using uv (Recommended)
    uv venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    ```

2.  **Dependencies**

    ```bash
    # Using uv
    uv pip install -r requirements.txt
    # OR using standard pip
    pip install -r requirements.txt
    ```

3.  **API Configuration**
    Create a `.env` file with your keys:
    ```env
    GROQ_API_KEY=...
    GEMINI_API_KEY=...
    COHERE_API_KEY=... # Recommended for RAG
    CEREBRAS_API_KEY=...
    ```

### Running the App

```bash
streamlit run main.py
```

## 🤝 Contribution

Contributions are welcome! This codebase is optimized for cross-platform scalability.

## 📄 License

MIT License.

---

**EduSolve v2.1** | Refactored Architecture | Powered by Free LLM APIs
