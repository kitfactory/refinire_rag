"""
Data models for refinire-rag
"""

from .document import Document
from .qa_pair import QAPair
from .evaluation_result import EvaluationResult

__all__ = [
    "Document",
    "QAPair",
    "EvaluationResult",
]