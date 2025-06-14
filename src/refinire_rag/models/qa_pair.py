"""
Question-Answer pair model implementation
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class QAPair:
    """
    A class representing a question-answer pair generated from a document.
    文書から生成された質問-回答ペアを表現するクラス
    """
    question: str
    answer: str
    document_id: str
    metadata: Dict[str, Any]
