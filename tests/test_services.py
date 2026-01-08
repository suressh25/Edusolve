import pytest
import asyncio
from unittest.mock import MagicMock
from services.answer_service import AnswerService

@pytest.mark.asyncio
async def test_parse_cleaned_qb(tmp_path):
    # This is a placeholder test
    router = MagicMock()
    service = AnswerService(router)
    assert hasattr(service, 'parse_cleaned_qb')

@pytest.mark.asyncio
async def test_validate_marks():
    router = MagicMock()
    service = AnswerService(router)
    questions = [{"question_text": "What is AI?", "marks": "0"}]
    validated = service.validate_marks(questions)
    assert validated[0]["marks"] == "2"
