# EduSolve 🎓🚀

**EduSolve** is a comprehensive, GenAI-powered educational assistant designed to streamline the academic workflow for students and educators. By orchestating multiple Large Language Models (LLMs) and leveraging Retrieval-Augmented Generation (RAG), EduSolve automates answer generation, handles complex document parsing, and creates study resources from mixed-media question banks.

![EduSolve Flowchart](edusolve_full_workflow_1767722706629.png)

## 🌟 Key Features

### 1. 🧠 Intelligent Answer Generation
*   **Multi-Model Router**: Automatically routes queries to the most efficient LLM (Groq Llama 3, Google Gemini Pro, or HuggingFace) based on complexity and rate limits.
*   **Context-Aware**: Generates precise, curriculum-aligned answers by understanding the context of the question.
*   **Mark-Based Optimization**: Adjusts the depth and length of the answer based on the allocated marks (e.g., brief for 2 marks, detailed for 10 marks).

### 2. 📚 Retrieval-Augmented Generation (RAG)
*   **Document Ingestion**: Upload course materials (PDFs, Images, Text) to build a custom knowledge base.
*   **Vector Search**: Uses **FAISS** and **Sentence-Transformers** to index and retrieve the most relevant study material for every query.
*   **Hallucination Reduction**: Grounded answers cited directly from user-provided documents.

### 3. 📝 Automated Question Bank Processing
*   **Hybrid OCR Engine**: Utilizes **Google Gemini Vision** to extract questions from handwritten notes, scanned PDFs, and complex image layouts.
*   **Format Analysis**: Distinguishes between text, images, and diagrams to parse question banks accurately.
*   **Batch Generation**: Generates answers for entire question banks in bulk and exports them as a formatted PDF.

## 🛠️ Technology Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **LLM Orchestration**: Python (Custom Router), LangChain
### 🤖 AI Model Configuration

| Provider | Model | Role | Limits (Free Tier) |
| :--- | :--- | :--- | :--- |
| **Groq** | `llama-3.1-70b-versatile` | **Primary Text** | 30 RPM, 1,000 RPD |
| **Cerebras** | `llama3.3-70b` | **Fast Fallback** | 30 RPM, 14,400 RPD |
| **Google Gemini** | `gemini-1.5-flash` | **Vision / OCR** | 15 RPM, 50 RPD |
| **Mistral** | `mistral-small-latest` | **High-Vol Fallback** | 60 RPM, 1B Tokens/Mo |
| **OpenRouter** | `llama-3.3-70b:free` | **Emergency** | 20 RPM, 50 RPD |

*   **RAG Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local) or `embed-english-v3.0` (Cohere)
*   **Vector Database**: FAISS (CPU-optimized)

## 📂 Project Structure

```bash
Edusolve/
├── api/                     # LLM integration and routing logic
│   ├── llm_router.py        # Core router for managing Groq, Gemini, etc.
│   └── *_client.py          # Individual provider clients
├── extraction/              # Document parsing modules
│   ├── image_extractor.py   # Gemini Vision integration for OCR
│   ├── text_extractor.py    # Hybrid text/table parsing
│   └── qb_cleaner.py        # Structure standardization module
├── generation/              # Logical modules for answer synthesis
│   ├── answer_generator.py  # Context-aware answer synthesis
│   └── booklet_compiler.py  # PDF/Docx formatting engine
├── qb_gen/                  # Question generation modules
├── rag/                     # RAG implementation
│   ├── document_processor.py# Chunking and text cleaning
│   ├── embedder.py          # Sentence-transformer embedding logic
│   ├── retriever.py         # Semantic search and reranking
│   └── vector_store.py      # FAISS index management
├── utils/                   # Helper functions (PDF generation, logging)
├── vector_stores/           # Persistent storage for FAISS indexes
├── uploads/                 # Temp storage for user uploads
├── outputs/                 # Generated result files
├── main.py                  # Streamlit application entry point
└── requirements.txt         # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   API Keys for at least **Groq** and **Gemini**.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/Edusolve.git
    cd Edusolve
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory. You can copy the template:
    ```bash
    cp .env.example .env
    ```
    Or manually add your keys:
    ```env
    # --- REQUIRED ---
    GROQ_API_KEY=gsk_...
    GEMINI_API_KEY=AIza...

    # --- RECOMMENDED (For better reliability) ---
    CEREBRAS_API_KEY=csk_...
    MISTRAL_API_KEY=...
    
    # --- OPTIONAL ---
    OPENROUTER_API_KEY=...
    COHERE_API_KEY=...
    ```

### Running the Application

Launch the Streamlit interface:
```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

## 📖 Usage Guide

1.  **Home**: Dashboard overview showing active LLM providers and system status.
2.  **📝 Answer Generation (Core)**: 
    *   **Upload**: Submit your Question Bank (PDF, DOCX, or Images).
    *   **Process**: The system automatically:
        *   Parses digital text.
        *   Uses **Gemini Vision** for scanned/handwritten extraction.
        *   Chunks content (>1000 words) for optimal processing.
    *   **Result**: Downloads a comprehensive **Answer Booklet** (Word/PDF) with structured answers scaled to the specific marks (e.g., 2 vs 10 marks).
3.  **🧠 RAG Module (Knowledge Base)**:
    *   **Upload**: Submit your course materials (Textbooks, Lecture Notes).
    *   **Index**: content is vector-embedded into a FAISS index.
    *   **Query**: Ask questions to receive answers grounded *strictly* in your uploaded material (reducing hallucinations).
4.  **🎯 QB Generator**:
    *   **Upload**: Submit raw course content or previous years' papers.
    *   **Generate**: Create new, structured Question Banks based on your specific curriculum preferences.

## 🤝 Contribution

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
