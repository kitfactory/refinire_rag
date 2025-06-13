from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class Document:
    """
    A class representing a document with metadata and content.
    メタデータとコンテンツを持つ文書を表現するクラス
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

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