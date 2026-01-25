# EduSolve 
**IEEE MegaProject'25 Hackathon Submission**

---

## Team Details

**Team Name:** Hackwarts

**Team Members:**
- Suresh Kumar K
- Tanusree P

---

## Project Domain

**Educational Technology (EdTech) & Artificial Intelligence**

EduSolve addresses the academic and education sector by leveraging generative AI to automate workflows for educators, students, and academic institutions.

---

## Idea

**Problem Statement:**
Educators spend countless hours manually creating question papers, generating answer keys, and preparing study materials. Students struggle to find relevant information from large volumes of study materials. Traditional methods are time-consuming, error-prone, and lack standardization.

**Solution:**
EduSolve is a professional, GenAI-powered academic automation suite that revolutionizes educational workflows through:
- **AI-Powered Answer Generation**: Automatically generates comprehensive, mark-weighted answers from question papers (PDF, DOCX, Images)
- **Advanced RAG Module**: Enables intelligent document search and retrieval from uploaded textbooks and study materials using vector embeddings
- **Question Bank Generator**: Creates high-quality, curriculum-aligned question banks from study materials with customizable difficulty levels
- **Multi-Modal Document Processing**: Handles text extraction, OCR, and document parsing with hybrid AI models
- **Intelligent LLM Router**: Ensures 99.9% uptime with automatic failover across multiple AI providers (Groq, Gemini, Cohere, Cerebras, Mistral)

**Key Features:**
- **Answer Generation Module**: Hybrid input system supporting file uploads and direct text entry with independent processing flows
- **Advanced RAG Module**: Enterprise-grade embeddings with FAISS vector store for hallucination-free answers
- **QB Generator Module**: Curriculum synthesis with Bloom's Taxonomy-based difficulty customization
- **Intelligent LLM Router**: Automatic failover across 6 AI providers for maximum reliability

---

## Tech Stack Used

**Frontend:**
- Streamlit (Multi-page Web Application)

**Backend:**
- Python 3.10+
- Service-Oriented Architecture (SOA)

**AI/ML:**
- Groq (llama-3.1-70b) - Primary Text Generation
- Cerebras (llama3.3-70b) - Fast Fallback
- Google Gemini (flash-1.5) - Vision/OCR
- Cohere (command-r-plus, embed-v3) - RAG & Embeddings
- Mistral (mistral-small) - High-Volume Text Processing
- OpenRouter - Alternative AI Provider

**Vector Database:**
- FAISS (Facebook AI Similarity Search)

**Document Processing:**
- PyMuPDF - PDF Parsing
- python-docx - Word Document Generation
- ReportLab - PDF Generation
- Pillow - Image Processing

**Package Management:**
- uv

**Additional Libraries:**
- LangChain - RAG Framework
- Hugging Face Transformers - Local Embeddings
- Streamlit Caching - Performance Optimization

---

## How to Execute Your Code

### Prerequisites
- Python 3.10 or higher
- API Keys for: Groq, Gemini, Cohere (optional: Cerebras, Mistral, OpenRouter)
- Internet connection

### Step-by-Step Instructions

**1. Clone the Repository**
```bash
git clone https://github.com/suressh25/Edusolve.git
cd Edusolve
```

**2. Create Virtual Environment**
```bash
# Using uv (Recommended - Lightning Fast)
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# OR using standard Python
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

**3. Install Dependencies**
```bash
# Using uv (Recommended)
uv pip install -r requirements.txt

# OR using standard pip
pip install -r requirements.txt
```

**4. Configure API Keys**
Create a `.env` file in the project root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Recommended for RAG
CEREBRAS_API_KEY=your_cerebras_api_key_here  # Optional
MISTRAL_API_KEY=your_mistral_api_key_here  # Optional
OPENROUTER_API_KEY=your_openrouter_api_key_here  # Optional
```

**5. Run the Application**
```bash
streamlit run main.py
```

**6. Access the Application**
Open your browser and navigate to:
```
http://localhost:8501
```

### Using the Application

**Answer Generation:**
1. Navigate to "Answer Generation" page
2. Upload question paper (PDF/DOCX/Image) or paste text
3. Select output format (Word/PDF)
4. Click "Generate Answers" and download the result

**RAG Module:**
1. Navigate to "RAG Module" page
2. Upload study materials (PDFs, DOCX files)
3. Click "Process Documents" to create vector embeddings
4. Ask questions and get context-aware answers

**QB Generator:**
1. Navigate to "QB Generator" page
2. Upload curriculum/study materials
3. Configure question types and difficulty
4. Generate and download question bank

---

## Code Sample/Link to Hosted Website

**GitHub Repository:**
[https://github.com/suressh25/Edusolve](https://github.com/suressh25/Edusolve)

**Project Demo Video:**
[https://edusolvein.streamlit.app/](https://edusolvein.streamlit.app/)

**Code Sample - LLM Router with Automatic Failover:**
```python
# Intelligent LLM Router with automatic failover
from api.llm_router import get_llm_router

# Initialize router with multiple providers
router = get_llm_router()

# Automatically routes to best available provider
response = router.generate_text(
    prompt="Generate answers for this question paper...",
    use_case="answer_generation"
)

# Fallback chain: Groq → Cerebras → Cohere → Mistral → OpenRouter
```

**Code Sample - RAG Implementation:**
```python
# Advanced RAG with FAISS vector store
from rag.retriever import Retriever
from rag.embedder import Embedder

# Process documents and create embeddings
embedder = Embedder()
retriever = Retriever(collection_name="study_materials")

# Upload and index documents
retriever.add_documents(documents)

# Query with context-aware retrieval
answer = retriever.query(
    question="Explain neural networks",
    top_k=5
)
```

---

## Project Structure

```bash
Edusolve/
├── api/                     # LLM integration and routing logic
├── extraction/              # OCR and Document parsing
├── generation/              # AI Synthesis (Answers/Booklets)
├── question_generation/     # QB Generator logic
├── services/                # Business logic (Decoupled flows)
├── components/              # Reusable UI elements
├── pages/                   # Streamlit multi-page application
├── rag/                     # RAG implementation (FAISS)
├── utils/                   # Utilities and helpers
├── config/                  # Global Settings
├── tests/                   # Verification tests
├── outputs/                 # Generated files
├── vector_stores/           # FAISS persistence
└── main.py                  # Application entry point
```

---

## License

MIT License

---

**EduSolve v2.1** | IEEE MegaProject'25 | Powered by Multi-Provider AI Network

