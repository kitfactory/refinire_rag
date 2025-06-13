"""
Evaluation result model implementation
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EvaluationResult:
    """
    A class representing the evaluation results of a RAG system.
    RAGシステムの評価結果を表現するクラス
    """
    precision: float
    recall: float
    f1_score: float
    metadata: Dict[str, Any] 