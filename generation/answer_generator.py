"""
Answer Generator - Generates mark-aware answers for questions
Uses centralized prompts from config.prompts
"""

import asyncio
from typing import Callable, List, Dict, Any, Optional
from utils.logger import get_logger
from config.prompts import PromptManager

logger = get_logger()  # Use singleton


class AnswerGenerator:
    """Generate comprehensive answers scaled by marks allocation"""

    def __init__(self, llm_router, rag_retriever=None):
        self.llm_router = llm_router
        self.rag_retriever = rag_retriever

    def _create_answer_prompt(
        self, question_text: str, marks: int, context: Optional[str] = None
    ) -> str:
        """Create mark-aware prompt with STRICT word count enforcement"""
        return PromptManager.get_prompt(
            "generate_answer",
            marks=marks,
            question_text=question_text,
            context=context,
        )

    async def generate_batch_answers(
        self,
        questions: List[Dict[str, Any]],
        use_rag: bool = False,
        batch_size: int = 5,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate answers for multiple questions in batches with reduced logging"""

        answers = []
        total_questions = len(questions)

        # Process in batches
        for i in range(0, total_questions, batch_size):
            batch = questions[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_questions + batch_size - 1) // batch_size

            # Only log batch start (not individual questions)
            logger.info(
                f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} questions)"
            )

            # Generate answers concurrently
            tasks = [self.generate_single_answer(q, use_rag) for q in batch]

            batch_answers = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            success_count = 0
            for j, result in enumerate(batch_answers):
                if isinstance(result, Exception):
                    logger.error(
                        f"❌ Q{batch[j].get('question_number')}: {str(result)[:100]}"
                    )
                    answers.append(
                        {
                            "question_number": batch[j].get("question_number"),
                            "question_text": batch[j].get("question_text"),
                            "marks": batch[j].get("marks"),
                            "answer": f"Error generating answer: {str(result)}",
                            "success": False,
                            "error": str(result),
                        }
                    )
                else:
                    if result.get("success"):
                        success_count += 1
                    answers.append(result)

            # Log batch summary instead of individual successes
            logger.info(
                f"✅ Batch {batch_num} complete: {success_count}/{len(batch)} successful"
            )

            # Update progress
            if progress_callback:
                progress = (i + len(batch)) / total_questions
                await progress_callback(progress)

            # Rate limiting between batches
            await asyncio.sleep(2)

        # Final summary
        total_success = len([a for a in answers if a.get("success")])
        logger.info(
            f"🎉 Completed: {total_success}/{total_questions} answers generated"
        )

        return answers

    async def generate_single_answer(
        self, question: Dict[str, Any], use_rag: bool = False
    ) -> Dict[str, Any]:
        """Generate answer for a single question (NO LOGGING to reduce spam)"""

        question_text = question.get("question_text", "")
        marks = int(question.get("marks", 5))
        question_number = question.get("question_number", "")

        try:
            # Determine question type
            q_lower = question_text.lower()

            if any(
                word in q_lower
                for word in ["code", "program", "implement", "write a function"]
            ):
                task_type = "code"
            elif any(
                word in q_lower for word in ["calculate", "solve", "prove", "derive"]
            ):
                task_type = "math"
            else:
                task_type = "answer_generation"

            # Get RAG context if needed
            context = None
            if use_rag and self.rag_retriever:
                try:
                    rag_results = await self.rag_retriever.retrieve(question_text)
                    if rag_results:
                        context = "\n\n".join(
                            [
                                f"Reference {i+1}:\n{doc['content']}"
                                for i, doc in enumerate(rag_results)
                            ]
                        )
                except Exception as e:
                    # Silently fail RAG, don't spam logs
                    pass

            # Create prompt
            prompt = self._create_answer_prompt(question_text, marks, context)

            # Calculate appropriate max_tokens
            max_tokens = int(min(marks * 80, 2048))

            # Generate answer (LLM Router will handle logging)
            response = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                task_type=task_type,
            )

            answer_text = response["text"]
            word_count = len(answer_text.split())

            # NO LOGGING HERE - reduces spam

            return {
                "question_number": question_number,
                "question_text": question_text,
                "marks": marks,
                "answer": answer_text,
                "word_count": word_count,
                "provider": response.get("provider", "unknown"),
                "success": True,
            }

        except Exception as e:
            # Only log errors
            return {
                "question_number": question_number,
                "question_text": question_text,
                "marks": marks,
                "answer": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
            }
