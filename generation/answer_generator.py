"""
Answer Generator - Generates mark-aware answers for questions
"""

import asyncio
from typing import List, Dict, Any, Optional
from utils.logger import logger


class AnswerGenerator:
    """Generate comprehensive answers scaled by marks allocation"""

    def __init__(self, llm_router, rag_retriever=None):
        self.llm_router = llm_router
        self.rag_retriever = rag_retriever

    def _create_answer_prompt(
        self, question_text: str, marks: int, context: Optional[str] = None
    ) -> str:
        """Create mark-aware prompt for answer generation"""

        base_prompt = f"""You are an expert academic tutor. Answer the following question comprehensively and accurately.

You must scale the depth, structure, and word count based on the marks allocated ({marks} marks).

Marks-to-Length Guidelines:
- 1 mark: 10-20 words (one-line factual answer)
- 2-3 marks: 30-60 words (short explanation)
- 4-6 marks: 80-120 words (concise explanation with reasoning)
- 7-10 marks: 150-250 words (structured answer with intro + points)
- 11-15 marks: 300-450 words (detailed discussion)
- 16+ marks: 500+ words (full essay with introduction, body, conclusion)

Rules:
1. Write in exam-style language: clear, concise, structured
2. For numerical problems, show steps and reasoning
3. For essays, use introduction + 3-5 key points + conclusion
4. Never include meta-commentary about how you formed the answer
5. Output only the answer text

Question ({marks} marks): {question_text}"""

        if context:
            base_prompt = f"""You are an expert academic tutor. Answer the following question using the provided study materials as context.

Context from study materials:
{context}

You must scale the depth, structure, and word count based on the marks allocated ({marks} marks).

Marks-to-Length Guidelines:
- 1 mark: 10-20 words (one-line factual answer)
- 2-3 marks: 30-60 words (short explanation)
- 4-6 marks: 80-120 words (concise explanation with reasoning)
- 7-10 marks: 150-250 words (structured answer with intro + points)
- 11-15 marks: 300-450 words (detailed discussion)
- 16+ marks: 500+ words (full essay with introduction, body, conclusion)

Prioritize information from the provided context. If context is insufficient, supplement with your knowledge.

Question ({marks} marks): {question_text}"""

        return base_prompt

    async def generate_single_answer(
        self, question: Dict[str, Any], use_rag: bool = False
    ) -> Dict[str, Any]:
        """Generate answer for a single question"""

        question_text = question.get("question_text", "")
        question_number = question.get("question_number", "N/A")

        try:
            marks = int(question.get("marks", 0))
        except (ValueError, TypeError):
            marks = 5  # Default to 5 marks if conversion fails

        try:
            # Retrieve context if RAG is enabled
            context = None
            if use_rag and self.rag_retriever:
                context = await self.rag_retriever.retrieve_context(question_text)

            # Generate prompt
            prompt = self._create_answer_prompt(question_text, marks, context)

            # Calculate max_tokens based on marks
            max_tokens = min(100 + (marks * 50), 4096)

            # Generate answer
            response = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                preferred_provider="groq",
            )

            answer_text = response["text"].strip()

            logger.info(f"Generated answer for Q{question_number} ({marks} marks)")

            return {
                "question_number": question_number,
                "question_text": question_text,
                "marks": marks,
                "answer": answer_text,
                "provider": response.get("provider", "unknown"),
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error generating answer for Q{question_number}: {str(e)}")
            return {
                "question_number": question_number,
                "question_text": question_text,
                "marks": marks,
                "answer": f"[Error generating answer: {str(e)}]",
                "success": False,
            }

    async def generate_batch_answers(
        self,
        questions: List[Dict[str, Any]],
        use_rag: bool = False,
        batch_size: int = 5,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Generate answers for multiple questions in batches"""

        all_answers = []
        total_questions = len(questions)

        # Process in batches to avoid rate limits
        for i in range(0, total_questions, batch_size):
            batch = questions[i : i + batch_size]

            logger.info(
                f"Processing batch {i//batch_size + 1}/{(total_questions + batch_size - 1)//batch_size}"
            )

            # Generate answers concurrently within batch
            batch_tasks = [self.generate_single_answer(q, use_rag) for q in batch]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Handle results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch answer generation error: {str(result)}")
                    all_answers.append(
                        {
                            "question_number": "ERROR",
                            "answer": f"[Error: {str(result)}]",
                            "success": False,
                        }
                    )
                else:
                    all_answers.append(result)

            # Update progress
            if progress_callback:
                progress = min((i + batch_size) / total_questions, 1.0)
                await progress_callback(progress)

            # Rate limiting pause between batches
            if i + batch_size < total_questions:
                await asyncio.sleep(2)

        logger.info(f"Completed generating {len(all_answers)} answers")
        return all_answers
