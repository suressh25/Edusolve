import asyncio
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional
from extraction.text_extractor import TextExtractor
from extraction.image_extractor import ImageExtractor
from extraction.qb_cleaner import QuestionBankCleaner
from generation.answer_generator import AnswerGenerator
from generation.booklet_compiler import BookletCompiler
from api.gemini_client import GeminiClient
from config.settings import settings
from utils.logger import get_logger
from docx import Document as DocxDocument
import re
import fitz

logger = get_logger()

class AnswerService:
    """Service to handle the coordination of answer generation workflows"""
    
    def __init__(self, llm_router: Any, rag_retriever: Any = None):
        self.llm_router = llm_router
        self.rag_retriever = rag_retriever
        
        # Collaborative initialization
        self.text_extractor = TextExtractor(llm_router)
        # Use existing Gemini client from router instead of creating new one
        self.image_extractor = ImageExtractor(llm_router.gemini) 
        self.qb_cleaner = QuestionBankCleaner(llm_router)
        self.booklet_compiler = BookletCompiler()
        # Pre-instantiate generator
        self.generator = AnswerGenerator(self.llm_router, self.rag_retriever)

    def parse_cleaned_qb(self, docx_path: str) -> List[Dict[str, Any]]:
        """Parse questions directly from cleaned QB DOCX structure"""
        doc = DocxDocument(docx_path)
        parsed_questions = []
        current_question = {}

        for para in doc.paragraphs:
            text = para.text.strip()

            if text.startswith("Question Number:"):
                if current_question and current_question.get("question_text"):
                    parsed_questions.append(current_question.copy())

                current_question = {
                    "question_number": text.replace("Question Number:", "").strip(),
                    "question_text": "",
                    "marks": "0",
                }

            elif text.startswith("Question Text:"):
                if current_question:
                    current_question["question_text"] = text.replace("Question Text:", "").strip()

            elif text.startswith("Marks Allocated:"):
                if current_question:
                    marks_text = text.replace("Marks Allocated:", "").strip()
                    marks_match = re.search(r"\d+", marks_text)
                    if marks_match:
                        current_question["marks"] = marks_match.group()

        if current_question and current_question.get("question_text"):
            parsed_questions.append(current_question)

        return parsed_questions

    def validate_marks(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure all questions have reasonable marks allocated"""
        valid_questions = []
        for q in questions:
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
        return valid_questions

    async def process_question_bank(
        self, 
        file_path: str, 
        use_rag: bool, 
        output_format: str, 
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Dict[str, Any]:
        """Full pipeline to process a question bank file"""
        
        file_extension = Path(file_path).suffix.lower()
        file_name_stem = Path(file_path).stem

        # 1. Extraction
        if progress_callback: progress_callback(20, "📖 Extracting questions...")
        
        if file_extension in [".pdf", ".docx", ".txt"]:
            if file_extension == ".pdf":
                doc = fitz.open(file_path)
                first_page_text = doc[0].get_text().strip()
                doc.close()

                if len(first_page_text) < 100:
                    questions = await self.image_extractor.extract_from_scanned_pdf(file_path)
                else:
                    raw_text = await self.text_extractor.extract_text_from_file(file_path)
                    questions = await self.text_extractor.extract_questions_with_llm(raw_text)
            else:
                raw_text = await self.text_extractor.extract_text_from_file(file_path)
                questions = await self.text_extractor.extract_questions_with_llm(raw_text)
        else:
            questions = await self.image_extractor.extract_from_image_file(file_path)

        # 2. Clean QB Generation
        if progress_callback: progress_callback(50, "🧹 Generating cleaned question bank...")
        
        cleaned_qb_path = await self.qb_cleaner.generate_cleaned_document(
            questions,
            str(settings.OUTPUT_DIR / f"{file_name_stem} - Cleaned.docx"),
            file_name_stem
        )

        # 3. Parsing and Validation
        if progress_callback: progress_callback(60, "🔍 Validating questions...")
        final_questions = self.parse_cleaned_qb(cleaned_qb_path)
        final_questions = self.validate_marks(final_questions)

        # 4. Answer Generation
        if progress_callback: progress_callback(65, "💡 Generating answers...")
        
        # Use pre-instantiated generator, updating RAG context if needed
        self.generator.rag_retriever = self.rag_retriever if use_rag else None
        
        async def wrap_progress(p):
            if progress_callback:
                progress_callback(int(65 + (p * 30)), "💡 Generating answers...")

        answers = await self.generator.generate_batch_answers(
            final_questions,
            use_rag=use_rag,
            batch_size=5,
            progress_callback=wrap_progress
        )

        # 5. Compilation
        if progress_callback: progress_callback(95, "📄 Compiling answer booklet...")
        
        if "PDF" in output_format:
            output_path = str(settings.OUTPUT_DIR / f"{file_name_stem} Answers.pdf")
            final_path = await self.booklet_compiler.compile_to_pdf(answers, output_path, file_name_stem)
        else:
            output_path = str(settings.OUTPUT_DIR / f"{file_name_stem} Answers.docx")
            final_path = await self.booklet_compiler.compile_to_word(answers, output_path, file_name_stem)

        return {
            "questions_count": len(questions),
            "final_questions_count": len(final_questions),
            "answers": answers,
            "cleaned_qb_path": cleaned_qb_path,
            "answer_booklet_path": final_path,
            "success_count": len([a for a in answers if a.get("success")])
        }
