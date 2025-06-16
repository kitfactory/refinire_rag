"""
Evaluation module for RAG system assessment

RAGシステム評価用モジュール

This module provides comprehensive evaluation capabilities including:
- Base evaluator classes for different evaluation types
- Specific metric implementations (QuestEval, BLEU, ROUGE, LLM Judge)
- Composite evaluation for combining multiple metrics
"""

from .base_evaluator import (
    BaseEvaluator,
    ReferenceBasedEvaluator,
    ReferenceFreeEvaluator,
    CompositeEvaluator,
    BaseEvaluatorConfig,
    EvaluationScore
)

from .questeval_evaluator import QuestEvalEvaluator, QuestEvalConfig
from .bleu_evaluator import BLEUEvaluator, BLEUConfig
from .rouge_evaluator import ROUGEEvaluator, ROUGEConfig
from .llm_judge_evaluator import LLMJudgeEvaluator, LLMJudgeConfig

__all__ = [
    # Base classes
    "BaseEvaluator",
    "ReferenceBasedEvaluator", 
    "ReferenceFreeEvaluator",
    "CompositeEvaluator",
    "BaseEvaluatorConfig",
    "EvaluationScore",
    
    # Specific evaluators
    "QuestEvalEvaluator",
    "QuestEvalConfig",
    "BLEUEvaluator",
    "BLEUConfig", 
    "ROUGEEvaluator",
    "ROUGEConfig",
    "LLMJudgeEvaluator",
    "LLMJudgeConfig"
]


def create_comprehensive_evaluator(
    enable_questeval: bool = True,
    enable_bleu: bool = True,
    enable_rouge: bool = True,
    enable_llm_judge: bool = True,
    questeval_config: QuestEvalConfig = None,
    bleu_config: BLEUConfig = None,
    rouge_config: ROUGEConfig = None,
    llm_judge_config: LLMJudgeConfig = None
) -> CompositeEvaluator:
    """
    Create a comprehensive evaluator combining multiple metrics
    
    複数のメトリクスを組み合わせた包括的評価器を作成
    
    Args:
        enable_questeval: Enable QuestEval evaluator
                         QuestEval評価器を有効化
        enable_bleu: Enable BLEU evaluator
                    BLEU評価器を有効化
        enable_rouge: Enable ROUGE evaluator
                     ROUGE評価器を有効化
        enable_llm_judge: Enable LLM Judge evaluator
                         LLM Judge評価器を有効化
        questeval_config: Custom QuestEval configuration
                         カスタムQuestEval設定
        bleu_config: Custom BLEU configuration
                    カスタムBLEU設定
        rouge_config: Custom ROUGE configuration
                     カスタムROUGE設定
        llm_judge_config: Custom LLM Judge configuration
                         カスタムLLM Judge設定
    
    Returns:
        CompositeEvaluator: Combined evaluator with all enabled metrics
                           有効化されたすべてのメトリクスを含む複合評価器
    """
    evaluators = []
    
    if enable_questeval:
        config = questeval_config or QuestEvalConfig()
        evaluators.append(QuestEvalEvaluator(config))
    
    if enable_bleu:
        config = bleu_config or BLEUConfig()
        evaluators.append(BLEUEvaluator(config))
    
    if enable_rouge:
        config = rouge_config or ROUGEConfig()
        evaluators.append(ROUGEEvaluator(config))
    
    if enable_llm_judge:
        config = llm_judge_config or LLMJudgeConfig()
        evaluators.append(LLMJudgeEvaluator(config))
    
    return CompositeEvaluator(evaluators)


def create_quick_evaluator(evaluation_type: str = "comprehensive") -> BaseEvaluator:
    """
    Create a quick evaluator for common use cases
    
    一般的な用途向けのクイック評価器を作成
    
    Args:
        evaluation_type: Type of evaluation to perform
                        実行する評価の種類
                        Options: "comprehensive", "lexical", "semantic", "llm_only"
    
    Returns:
        BaseEvaluator: Configured evaluator
                      設定済み評価器
    """
    if evaluation_type == "comprehensive":
        return create_comprehensive_evaluator()
    
    elif evaluation_type == "lexical":
        # Focus on lexical overlap metrics
        return create_comprehensive_evaluator(
            enable_questeval=False,
            enable_bleu=True,
            enable_rouge=True,
            enable_llm_judge=False
        )
    
    elif evaluation_type == "semantic":
        # Focus on semantic evaluation
        return create_comprehensive_evaluator(
            enable_questeval=True,
            enable_bleu=False,
            enable_rouge=False,
            enable_llm_judge=True
        )
    
    elif evaluation_type == "llm_only":
        # Use only LLM-based evaluation
        return LLMJudgeEvaluator(LLMJudgeConfig())
    
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")