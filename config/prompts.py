"""
Centralized Prompt Management System
Unified prompts for all question extraction, generation, and processing tasks
Following DRY principle with dynamic prompt construction
"""

from typing import Dict, List, Any, Optional


class PromptManager:
    """Centralized repository for all LLM prompts used in the application"""

    # ==================== QUESTION EXTRACTION PROMPTS ====================

    @staticmethod
    def extract_questions_from_text() -> str:
        """
        Universal prompt for extracting questions from digital text documents
        Handles all question types and mark formats
        """
        return """Extract ALL academic questions from the provided text.

IMPORTANT - Handle ALL question formats:
- Multiple Choice: A/B/C/D, 1/2/3/4, a/b/c/d formats
- True/False: Identify true/false statements
- Fill in blanks: [___], ____, (...) formats
- Short Answer: Questions expecting 1-3 line answers
- Long Answer: Questions expecting paragraph responses
- Match the following: Matching pairs/columns
- Assertion-Reason: If assertion then reason statements
- Numerical: Calculations with specific values
- Subjective: Opinion or descriptive answers

For EACH question extract:
1. question_number: Original numbering (1, Q1, 1.a, etc.)
2. question_text: Complete question text with options if MCQ
3. question_type: Type from list above
4. marks: Allocated marks (parse patterns: [5], (5 marks), 5M, 5 pts, etc.)
5. options: Array of options for MCQ/True-False (empty for other types)

Output ONLY valid JSON (no markdown, no explanation):
[
  {
    "question_number": "1",
    "question_text": "Complete text here with options if MCQ",
    "question_type": "MCQ",
    "marks": "5",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"]
  },
  {
    "question_number": "2",
    "question_text": "Question text",
    "question_type": "Short Answer",
    "marks": "2",
    "options": []
  }
]

RULES:
- Extract EVERY question visible
- If marks missing: estimate 2-15 range based on question complexity
- Preserve exact question numbering and structure
- Keep technical terms and formulas intact
- For MCQ: include all options in question_text AND options array
- Never skip questions even if formatting is poor"""

    @staticmethod
    def extract_questions_from_image() -> str:
        """
        Vision-based OCR prompt for extracting questions from images/scanned documents
        Handles handwritten and printed text
        """
        return """Perform complete OCR on this image and extract ALL academic questions visible.

IMPORTANT - Identify and extract ALL question types:
- Multiple Choice, True/False, Fill in blanks
- Short Answer, Long Answer, Essay type
- Match the following, Assertion-Reason, Numerical
- Subjective/Descriptive questions
- Sub-questions and nested parts (e.g., 1.a, 1.b)

For EACH question extract:
1. question_number: Exact numbering as shown (Q1, 1., (a), etc.)
2. question_text: Complete question with all details and options
3. question_type: Classify question type
4. marks: Look for marks patterns: [5], (5), 5M, 5 marks, 5 pts, etc.
5. options: For MCQ/True-False include all options shown
6. source_page: Page number where found

Output ONLY valid JSON:
[
  {
    "question_number": "1",
    "question_text": "Full question text with options for MCQ",
    "question_type": "Type",
    "marks": "5",
    "options": ["Option 1", "Option 2"],
    "source_page": 1
  }
]

CRITICAL:
- Extract EVERY question even if handwritten/poorly scanned
- If marks not visible, use "0" as default (don't estimate for images)
- Preserve exact formatting and numbering
- Handle sub-questions: if Q1 has parts (a), (b), extract separately
- Keep mathematical formulas and special characters accurate
- Note handwritten areas: mark unclear text with [unclear]"""

    @staticmethod
    def standardize_questions() -> str:
        """
        Standardize and clean extracted questions
        Fixes OCR errors, formatting issues, inconsistencies
        """
        return """Standardize and validate the following extracted questions.

For each question:
1. Fix OCR errors and spelling mistakes
2. Correct grammar while preserving meaning
3. Ensure question text is complete and coherent
4. Validate question numbering consistency
5. Preserve all technical terms, formulas, and special characters exactly
6. Correct mark values (ensure numeric format)
7. Verify all MCQ options are included
8. Check question_type is correct

Validation rules:
- All questions must have: question_number, question_text, marks
- Marks must be numeric (no symbols)
- For MCQ: must have at least 2 options
- question_type must be valid
- No duplicate questions by text content
- Empty options array for non-MCQ types

Output ONLY valid JSON with corrections applied:
[
  {
    "question_number": "1",
    "question_text": "Corrected complete question text",
    "question_type": "Corrected type",
    "marks": "5",
    "options": ["Corrected options"],
    "is_valid": true,
    "issues_fixed": "List of fixes applied"
  }
]"""

    # ==================== QUESTION GENERATION PROMPTS ====================

    @staticmethod
    def generate_questions(
        num_questions: int,
        difficulty: str,
        question_types: List[str],
        marks_distribution: Dict[str, int],
        topics: Optional[List[str]] = None,
    ) -> str:
        """
        Generate new questions from course material
        Supports multiple question types and difficulty levels
        """
        marks_info = ", ".join(
            [f"{count}x {marks} marks" for marks, count in marks_distribution.items()]
        )
        types_info = ", ".join(question_types)
        topics_info = ", ".join(topics) if topics else "All topics"

        return f"""You are an expert academic question paper creator.
Generate {num_questions} high-quality exam questions based on provided course material.

REQUIREMENTS:
- Difficulty Level: {difficulty}
- Question Types: {types_info}
- Marks Distribution: {marks_info}
- Topics to Focus: {topics_info}
- Total Questions: {num_questions}

QUESTION TYPE DEFINITIONS:
- MCQ: Multiple choice with 4 options (A/B/C/D)
- True/False: Binary answer
- Fill in Blank: Complete the statement
- Short Answer: 1-3 lines expected
- Long Answer: Full paragraph expected
- Match the following: Pair items
- Assertion-Reason: If X then Y format
- Numerical: Math problem with calculation
- Essay: Descriptive/subjective response

For EACH question provide:
1. question_number: Sequential (1, 2, 3, etc.)
2. question_text: Complete question text
3. question_type: From list above
4. marks: Allocated marks
5. topic: Concept/topic covered
6. difficulty: Question difficulty level
7. options: For MCQ format as ["A) Option", "B) Option", ...]
8. answer: Expected answer or answer key

Output ONLY valid JSON:
[
  {{
    "question_number": "1",
    "question_text": "Question with options if MCQ",
    "question_type": "Type",
    "marks": "5",
    "topic": "Topic name",
    "difficulty": "{difficulty}",
    "options": ["A) Option1", "B) Option2"],
    "answer": "Expected answer or A for MCQ"
  }}
]

GUIDELINES:
- Test understanding not memorization
- Ensure questions are clear and unambiguous
- Cover diverse concepts from material
- Match difficulty level perfectly
- Distribute marks according to specification
- For MCQ: provide 4 distinct options
- Ensure exactly {num_questions} questions generated"""

    # ==================== FORMATTING & CLEANUP PROMPTS ====================

    @staticmethod
    def reformat_json_output() -> str:
        """
        Convert unstructured LLM output into proper JSON format
        Used as fallback for failed JSON parsing
        """
        return """Convert the following text into valid JSON format.

Extract all questions and format as JSON array:
[
  {
    "question_number": "1",
    "question_text": "Question text",
    "question_type": "Type",
    "marks": "5",
    "options": [],
    "source": "Original source if available"
  }
]

Rules:
- Extract only question-related information
- Ensure valid JSON syntax
- For incomplete data use empty strings ""
- For missing marks use "0"
- options array: empty [] unless MCQ/True-False
- Remove any non-JSON text"""

    @staticmethod
    def validate_and_enhance_questions() -> str:
        """
        Validate question structure and enhance with missing details
        """
        return """Validate and enhance the following questions.

For each question:
1. Verify all required fields present
2. Ensure question_text is clear and grammatically correct
3. Validate question_type is appropriate
4. Ensure marks is numeric
5. For MCQ: verify 4 options present
6. Add missing marks if possible based on question complexity

Missing data rules:
- If marks missing: estimate 2-15 based on complexity
- If question_type missing: infer from question content
- If options missing for MCQ: mark as incomplete

Output validated JSON:
[
  {
    "question_number": "1",
    "question_text": "Validated text",
    "question_type": "Correct type",
    "marks": "5",
    "options": [],
    "validation_status": "Valid/Needs Review",
    "notes": "Any enhancements made"
  }
]"""

    # ==================== ANSWER GENERATION PROMPTS ====================

    @staticmethod
    def generate_answer(
        marks: int, question_text: str, context: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive answers scaled by marks allocation
        Enforces strict word count and depth requirements
        """
        # Calculate exact target word count
        min_words = marks * 35
        max_words = marks * 50
        target_words = (min_words + max_words) // 2

        base_instructions = f"""You are an expert academic tutor. Write a comprehensive exam answer.

QUESTION ({marks} marks): {question_text}

MANDATORY WORD COUNT: Your answer MUST be between {min_words}-{max_words} words. Target: {target_words} words.

WRITING INSTRUCTIONS FOR {marks} MARKS:
"""

        # Specific instructions by mark range
        if marks == 1:
            instructions = """- One concise sentence (15-25 words)
- Direct factual answer only
- No elaboration needed"""

        elif marks == 2:
            instructions = """- 2-3 sentences (40-70 words minimum)
- State the main concept clearly
- Add one supporting detail or example
- Be specific and detailed enough for 2 marks"""

        elif marks in [3, 4, 5]:
            instructions = f"""- Write {marks+1} to {marks+2} sentences ({min_words}-{max_words} words)
- Introduction: State the main concept
- Body: Explain with 2-3 key points
- Add relevant examples or details
- Ensure sufficient depth for {marks} marks"""

        elif marks in [6, 7, 8, 9, 10]:
            instructions = f"""- Write 2-3 paragraphs ({min_words}-{max_words} words)
- Introduction paragraph: Define/introduce the concept (30-40 words)
- Body paragraph(s): Explain 3-4 key points with details (100-180 words)
- Brief conclusion: Summarize or state significance (20-30 words)
- Include examples, formulas, or diagrams if relevant"""

        elif marks in [11, 12, 13, 14, 15]:
            instructions = f"""- Write a comprehensive essay ({min_words}-{max_words} words minimum)
- Introduction: Context and overview (50-70 words)
- Body: 3-4 detailed paragraphs covering:
  * Each major aspect in depth (80-100 words per paragraph)
  * Include examples, comparisons, applications
  * Use technical terminology appropriately
- Conclusion: Summary and implications (50-70 words)
- This is {marks} marks - write extensively with full details"""

        else:  # 16+ marks
            instructions = f"""- Write an extensive essay ({min_words}-{max_words} words minimum)
- Detailed introduction with background (80-100 words)
- Multiple body sections (4-5 paragraphs):
  * Each section thoroughly explores one aspect (100-120 words)
  * Include examples, case studies, comparisons
  * Critical analysis and evaluation
- Comprehensive conclusion (80-100 words)
- This is {marks} marks - write very extensively"""

        full_prompt = f"""{base_instructions}{instructions}

CRITICAL REQUIREMENTS:
✓ Write EXACTLY {target_words} words (±10% acceptable)
✓ Use proper paragraph structure
✓ Write in formal academic language
✓ Include specific details, not generic statements
✓ For {marks} marks, depth and detail are essential

Start writing your {target_words}-word answer now:"""

        if context:
            full_prompt = f"""You are an expert academic tutor using provided study materials.

CONTEXT FROM STUDY MATERIALS:
{context[:2000]}

QUESTION ({marks} marks): {question_text}

MANDATORY WORD COUNT: {min_words}-{max_words} words. Target: {target_words} words.

{full_prompt.split('WRITING INSTRUCTIONS FOR')[1]}"""

        return full_prompt

    # ==================== HELPER METHODS ====================

    @staticmethod
    def get_all_prompts() -> Dict[str, str]:
        """Return dictionary of all available prompts"""
        return {
            "extract_text": PromptManager.extract_questions_from_text(),
            "extract_image": PromptManager.extract_questions_from_image(),
            "standardize": PromptManager.standardize_questions(),
            "reformat": PromptManager.reformat_json_output(),
            "validate": PromptManager.validate_and_enhance_questions(),
        }

    @staticmethod
    def get_prompt(prompt_type: str, **kwargs) -> str:
        """
        Get prompt by type with dynamic parameter substitution

        Args:
            prompt_type: Type of prompt needed
            **kwargs: Parameters for dynamic prompts

        Returns:
            Formatted prompt string
        """
        if prompt_type == "extract_text":
            return PromptManager.extract_questions_from_text()
        elif prompt_type == "extract_image":
            return PromptManager.extract_questions_from_image()
        elif prompt_type == "standardize":
            return PromptManager.standardize_questions()
        elif prompt_type == "generate":
            return PromptManager.generate_questions(
                num_questions=kwargs.get("num_questions", 20),
                difficulty=kwargs.get("difficulty", "Medium"),
                question_types=kwargs.get("question_types", ["Short Answer"]),
                marks_distribution=kwargs.get("marks_distribution", {"2": 5}),
                topics=kwargs.get("topics"),
            )
        elif prompt_type == "generate_answer":
            return PromptManager.generate_answer(
                marks=kwargs.get("marks", 5),
                question_text=kwargs.get("question_text", ""),
                context=kwargs.get("context"),
            )
        elif prompt_type == "reformat":
            return PromptManager.reformat_json_output()
        elif prompt_type == "validate":
            return PromptManager.validate_and_enhance_questions()
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
