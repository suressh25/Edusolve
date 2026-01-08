import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from question_generation.qb_generator import QuestionBankGenerator
from rag.document_processor import DocumentProcessor
from utils.file_handler import FileHandler
from config.settings import settings
from utils.logger import get_logger

logger = get_logger()

class QBService:
    """Service to handle automated question bank generation"""
    
    def __init__(self, llm_router: Any):
        self.llm_router = llm_router
        self.doc_processor = DocumentProcessor()
        self.qb_generator = QuestionBankGenerator(llm_router)

    async def generate_qb(
        self,
        uploaded_file: Any,
        num_questions: int,
        difficulty: str,
        question_types: List[str],
        marks_distribution: Dict[str, int],
        topics: Optional[List[str]] = None,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Full pipeline to generate a question bank from material"""
        
        # 1. Save and extract
        if progress_callback: progress_callback(10, "📖 Reading course material...")
        file_path = await FileHandler.save_uploaded_file(uploaded_file, str(settings.UPLOAD_DIR))
        course_text = await self.doc_processor._extract_text(file_path)
        
        if progress_callback: progress_callback(40, "🎯 Generating questions...")
        
        # 2. Generate questions
        questions = await self.qb_generator.generate_questions(
            course_text,
            num_questions=num_questions,
            difficulty=difficulty,
            question_types=question_types,
            marks_distribution=marks_distribution,
            topics=topics
        )
        
        if progress_callback: progress_callback(80, "💾 Saving question bank...")
        
        # 3. Save
        output_path = str(settings.OUTPUT_DIR / f"{Path(uploaded_file.name).stem} - Generated QB.docx")
        saved_path = await self.qb_generator.save_question_bank(
            questions,
            output_path,
            f"Generated Question Bank - {Path(uploaded_file.name).stem}"
        )
        
        return {
            "questions": questions,
            "saved_path": saved_path
        }
