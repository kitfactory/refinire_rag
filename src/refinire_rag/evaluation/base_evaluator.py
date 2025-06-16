"""
Base Evaluator for RAG evaluation metrics

RAG評価メトリクスの基底クラス
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

from ..models.document import Document


@dataclass
class EvaluationScore:
    """
    Evaluation score with metadata
    
    評価スコアとメタデータ
    """
    metric_name: str
    score: float
    details: Dict[str, Any] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BaseEvaluatorConfig(BaseModel):
    """
    Base configuration for evaluators
    
    評価器の基本設定
    """
    name: str = Field(..., description="Name of the evaluator / 評価器の名前")
    enabled: bool = Field(True, description="Whether the evaluator is enabled / 評価器が有効かどうか")
    weight: float = Field(1.0, description="Weight for combining scores / スコア結合時の重み")
    threshold: Optional[float] = Field(None, description="Optional threshold for pass/fail / 合否判定の閾値（オプション）")


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation metrics
    
    すべての評価メトリクスの抽象基底クラス
    """
    
    def __init__(self, config: BaseEvaluatorConfig):
        """
        Initialize the evaluator with configuration
        
        設定で評価器を初期化
        
        Args:
            config: Evaluator configuration / 評価器の設定
        """
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.weight = config.weight
        self.threshold = config.threshold
    
    @abstractmethod
    def evaluate(
        self,
        reference: Union[str, List[str]],
        candidate: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationScore:
        """
        Evaluate a candidate answer against reference(s)
        
        候補の回答を参照と比較して評価
        
        Args:
            reference: Reference answer(s) / 参照回答
            candidate: Candidate answer to evaluate / 評価する候補回答
            context: Optional context information / オプションのコンテキスト情報
            
        Returns:
            EvaluationScore: Evaluation result / 評価結果
        """
        pass
    
    @abstractmethod
    def batch_evaluate(
        self,
        references: List[Union[str, List[str]]],
        candidates: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[EvaluationScore]:
        """
        Evaluate multiple candidate answers in batch
        
        複数の候補回答をバッチで評価
        
        Args:
            references: List of reference answer(s) / 参照回答のリスト
            candidates: List of candidate answers / 候補回答のリスト
            contexts: Optional list of context information / オプションのコンテキスト情報リスト
            
        Returns:
            List[EvaluationScore]: List of evaluation results / 評価結果のリスト
        """
        pass
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before evaluation (can be overridden)
        
        評価前のテキスト前処理（オーバーライド可能）
        
        Args:
            text: Input text / 入力テキスト
            
        Returns:
            str: Preprocessed text / 前処理されたテキスト
        """
        # Default preprocessing: strip whitespace and normalize
        # デフォルト前処理：空白の削除と正規化
        return text.strip().lower()
    
    def validate_inputs(
        self,
        reference: Union[str, List[str]],
        candidate: str
    ) -> None:
        """
        Validate inputs before evaluation
        
        評価前の入力検証
        
        Args:
            reference: Reference answer(s) / 参照回答
            candidate: Candidate answer / 候補回答
            
        Raises:
            ValueError: If inputs are invalid / 入力が無効な場合
        """
        if not candidate or not candidate.strip():
            raise ValueError("Candidate answer cannot be empty / 候補回答は空にできません")
        
        if isinstance(reference, str):
            if not reference or not reference.strip():
                raise ValueError("Reference answer cannot be empty / 参照回答は空にできません")
        elif isinstance(reference, list):
            if not reference or all(not r.strip() for r in reference):
                raise ValueError("At least one reference answer must be non-empty / 少なくとも1つの参照回答は空でない必要があります")
        else:
            raise ValueError("Reference must be string or list of strings / 参照は文字列または文字列のリストである必要があります")
    
    def apply_threshold(self, score: float) -> bool:
        """
        Apply threshold to determine pass/fail
        
        閾値を適用して合否を判定
        
        Args:
            score: Evaluation score / 評価スコア
            
        Returns:
            bool: Whether the score passes the threshold / スコアが閾値を超えているか
        """
        if self.threshold is None:
            return True
        return score >= self.threshold
    
    def get_metric_info(self) -> Dict[str, Any]:
        """
        Get information about this metric
        
        このメトリクスに関する情報を取得
        
        Returns:
            Dict[str, Any]: Metric information / メトリクス情報
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "weight": self.weight,
            "threshold": self.threshold,
            "description": self.__class__.__doc__
        }


class ReferenceBasedEvaluator(BaseEvaluator):
    """
    Base class for evaluators that compare against reference answers
    
    参照回答と比較する評価器の基底クラス
    """
    
    def batch_evaluate(
        self,
        references: List[Union[str, List[str]]],
        candidates: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[EvaluationScore]:
        """
        Default batch evaluation implementation
        
        デフォルトのバッチ評価実装
        """
        if contexts is None:
            contexts = [None] * len(candidates)
        
        if len(references) != len(candidates) or len(references) != len(contexts):
            raise ValueError(
                "Length of references, candidates, and contexts must match / "
                "references、candidates、contextsの長さは一致する必要があります"
            )
        
        results = []
        for ref, cand, ctx in zip(references, candidates, contexts):
            results.append(self.evaluate(ref, cand, ctx))
        
        return results


class ReferenceFreeEvaluator(BaseEvaluator):
    """
    Base class for evaluators that don't require reference answers
    
    参照回答を必要としない評価器の基底クラス
    """
    
    @abstractmethod
    def evaluate_without_reference(
        self,
        candidate: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationScore:
        """
        Evaluate a candidate answer without reference
        
        参照なしで候補回答を評価
        
        Args:
            candidate: Candidate answer to evaluate / 評価する候補回答
            context: Optional context information / オプションのコンテキスト情報
            
        Returns:
            EvaluationScore: Evaluation result / 評価結果
        """
        pass
    
    def evaluate(
        self,
        reference: Union[str, List[str]],
        candidate: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationScore:
        """
        Evaluate method that ignores reference
        
        参照を無視する評価メソッド
        """
        return self.evaluate_without_reference(candidate, context)
    
    def batch_evaluate(
        self,
        references: List[Union[str, List[str]]],
        candidates: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[EvaluationScore]:
        """
        Batch evaluation that ignores references
        
        参照を無視するバッチ評価
        """
        if contexts is None:
            contexts = [None] * len(candidates)
        
        results = []
        for cand, ctx in zip(candidates, contexts):
            results.append(self.evaluate_without_reference(cand, ctx))
        
        return results


class CompositeEvaluator(BaseEvaluator):
    """
    Evaluator that combines multiple evaluation metrics
    
    複数の評価メトリクスを組み合わせる評価器
    """
    
    def __init__(self, config: BaseEvaluatorConfig, evaluators: List[BaseEvaluator]):
        """
        Initialize with multiple evaluators
        
        複数の評価器で初期化
        
        Args:
            config: Composite evaluator configuration / 複合評価器の設定
            evaluators: List of evaluators to combine / 組み合わせる評価器のリスト
        """
        super().__init__(config)
        self.evaluators = [e for e in evaluators if e.enabled]
        
        # Normalize weights
        # 重みを正規化
        total_weight = sum(e.weight for e in self.evaluators)
        if total_weight > 0:
            for e in self.evaluators:
                e.weight = e.weight / total_weight
    
    def evaluate(
        self,
        reference: Union[str, List[str]],
        candidate: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationScore:
        """
        Evaluate using all sub-evaluators and combine scores
        
        すべてのサブ評価器を使用して評価し、スコアを結合
        """
        if not self.evaluators:
            return EvaluationScore(
                metric_name=self.name,
                score=0.0,
                details={"error": "No evaluators enabled / 有効な評価器がありません"}
            )
        
        sub_scores = []
        details = {}
        
        for evaluator in self.evaluators:
            try:
                sub_score = evaluator.evaluate(reference, candidate, context)
                sub_scores.append(sub_score)
                details[evaluator.name] = {
                    "score": sub_score.score,
                    "weight": evaluator.weight,
                    "details": sub_score.details
                }
            except Exception as e:
                details[evaluator.name] = {
                    "error": str(e),
                    "weight": evaluator.weight
                }
        
        # Calculate weighted average
        # 重み付き平均を計算
        if sub_scores:
            weighted_sum = sum(s.score * e.weight for s, e in zip(sub_scores, self.evaluators))
            combined_score = weighted_sum
        else:
            combined_score = 0.0
        
        return EvaluationScore(
            metric_name=self.name,
            score=combined_score,
            details=details
        )
    
    def batch_evaluate(
        self,
        references: List[Union[str, List[str]]],
        candidates: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[EvaluationScore]:
        """
        Batch evaluate using all sub-evaluators
        
        すべてのサブ評価器を使用してバッチ評価
        """
        if contexts is None:
            contexts = [None] * len(candidates)
        
        results = []
        for ref, cand, ctx in zip(references, candidates, contexts):
            results.append(self.evaluate(ref, cand, ctx))
        
        return results