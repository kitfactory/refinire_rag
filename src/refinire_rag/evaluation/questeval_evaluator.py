"""
QuestEval Evaluator Implementation

QuestEval評価器の実装
QuestEvalは質問応答システム専用の評価指標で、生成された回答の品質を多角的に評価します。
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
import json

from .base_evaluator import ReferenceBasedEvaluator, BaseEvaluatorConfig, EvaluationScore


class QuestEvalConfig(BaseEvaluatorConfig):
    """
    Configuration for QuestEval evaluator
    
    QuestEval評価器の設定
    """
    name: str = "QuestEval"
    model_name: str = "gpt-4o-mini"  # LLM for semantic evaluation
    enable_consistency: bool = True  # 一貫性評価の有効化
    enable_answerability: bool = True  # 回答可能性評価の有効化
    enable_source_support: bool = True  # ソース支持度評価の有効化
    enable_fluency: bool = True  # 流暢性評価の有効化
    consistency_weight: float = 0.3
    answerability_weight: float = 0.25
    source_support_weight: float = 0.25
    fluency_weight: float = 0.2


class QuestEvalEvaluator(ReferenceBasedEvaluator):
    """
    QuestEval evaluator for question-answering systems
    
    質問応答システム向けQuestEval評価器
    
    QuestEvalは以下の4つの観点から評価を行います：
    1. Consistency: 回答の一貫性
    2. Answerability: 質問への回答可能性  
    3. Source Support: ソース文書による支持度
    4. Fluency: 回答の流暢性
    """
    
    def __init__(self, config: QuestEvalConfig):
        super().__init__(config)
        self.config: QuestEvalConfig = config
    
    def evaluate(self,
                reference: Union[str, List[str]],
                candidate: str,
                context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """
        Evaluate candidate answer using QuestEval metrics
        
        QuestEvalメトリクスを使用して候補回答を評価
        
        Args:
            reference: Reference answer(s)
                      参照回答
            candidate: Candidate answer to evaluate
                      評価する候補回答
            context: Context including question and source documents
                    質問とソース文書を含むコンテキスト
                    
        Returns:
            EvaluationScore: QuestEval evaluation result
                           QuestEval評価結果
        """
        self.validate_inputs(reference, candidate)
        
        # Preprocess inputs
        candidate = self.preprocess_text(candidate)
        if isinstance(reference, str):
            reference = self.preprocess_text(reference)
        else:
            reference = [self.preprocess_text(ref) for ref in reference]
        
        # Extract context information
        question = context.get("question", "") if context else ""
        source_documents = context.get("source_documents", []) if context else []
        
        # Compute individual QuestEval components
        scores = {}
        details = {}
        
        if self.config.enable_consistency:
            consistency_score = self._evaluate_consistency(reference, candidate, question)
            scores["consistency"] = consistency_score
            details["consistency"] = {"score": consistency_score}
        
        if self.config.enable_answerability:
            answerability_score = self._evaluate_answerability(question, candidate)
            scores["answerability"] = answerability_score
            details["answerability"] = {"score": answerability_score}
        
        if self.config.enable_source_support:
            support_score = self._evaluate_source_support(candidate, source_documents)
            scores["source_support"] = support_score
            details["source_support"] = {"score": support_score}
        
        if self.config.enable_fluency:
            fluency_score = self._evaluate_fluency(candidate)
            scores["fluency"] = fluency_score
            details["fluency"] = {"score": fluency_score}
        
        # Compute weighted overall score
        overall_score = self._compute_weighted_score(scores)
        
        # Detailed breakdown
        details.update({
            "component_scores": scores,
            "weights": {
                "consistency": self.config.consistency_weight,
                "answerability": self.config.answerability_weight,
                "source_support": self.config.source_support_weight,
                "fluency": self.config.fluency_weight
            },
            "overall_computation": f"Weighted average of {len(scores)} components"
        })
        
        return EvaluationScore(
            metric_name=self.config.name,
            score=overall_score,
            details=details,
            confidence=0.85  # QuestEval typically has high confidence
        )
    
    def _evaluate_consistency(self, reference: Union[str, List[str]], candidate: str, question: str) -> float:
        """
        Evaluate consistency between reference and candidate answers
        
        参照回答と候補回答の一貫性を評価
        """
        if isinstance(reference, list):
            # Compare with each reference and take the maximum
            consistency_scores = []
            for ref in reference:
                score = self._semantic_similarity(ref, candidate)
                consistency_scores.append(score)
            return max(consistency_scores) if consistency_scores else 0.0
        else:
            return self._semantic_similarity(reference, candidate)
    
    def _evaluate_answerability(self, question: str, candidate: str) -> float:
        """
        Evaluate how well the candidate answers the given question
        
        候補回答が質問にどの程度答えているかを評価
        """
        if not question or not candidate:
            return 0.0
        
        # Simplified answerability check
        # In a real implementation, this would use more sophisticated NLP
        
        # Check if answer contains question keywords
        question_words = set(self._extract_content_words(question.lower()))
        answer_words = set(self._extract_content_words(candidate.lower()))
        
        # Word overlap score
        overlap_score = len(question_words & answer_words) / len(question_words) if question_words else 0.0
        
        # Length appropriateness (not too short, not too long)
        length_score = self._evaluate_answer_length(candidate, question)
        
        # Question type appropriateness
        type_score = self._evaluate_question_type_match(question, candidate)
        
        return (overlap_score * 0.4 + length_score * 0.3 + type_score * 0.3)
    
    def _evaluate_source_support(self, candidate: str, source_documents: List[str]) -> float:
        """
        Evaluate how well the answer is supported by source documents
        
        ソース文書による回答の支持度を評価
        """
        if not source_documents or not candidate:
            return 0.0
        
        candidate_words = set(self._extract_content_words(candidate.lower()))
        
        # Calculate support from each source document
        support_scores = []
        for source in source_documents:
            source_words = set(self._extract_content_words(source.lower()))
            
            # Word overlap with source
            overlap = len(candidate_words & source_words)
            overlap_score = overlap / len(candidate_words) if candidate_words else 0.0
            
            # Semantic similarity (simplified)
            semantic_score = self._semantic_similarity(candidate, source)
            
            # Combined support score
            support_score = (overlap_score * 0.6 + semantic_score * 0.4)
            support_scores.append(support_score)
        
        # Return maximum support across all sources
        return max(support_scores) if support_scores else 0.0
    
    def _evaluate_fluency(self, candidate: str) -> float:
        """
        Evaluate fluency and readability of the candidate answer
        
        候補回答の流暢性と可読性を評価
        """
        if not candidate.strip():
            return 0.0
        
        # Grammar and structure checks (simplified)
        grammar_score = self._check_basic_grammar(candidate)
        
        # Readability score
        readability_score = self._calculate_readability(candidate)
        
        # Coherence score
        coherence_score = self._evaluate_coherence(candidate)
        
        return (grammar_score * 0.4 + readability_score * 0.3 + coherence_score * 0.3)
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts (simplified implementation)
        
        2つのテキスト間の意味的類似度を計算（簡易実装）
        """
        # This is a simplified implementation
        # In practice, you would use embeddings or pre-trained models
        
        words1 = set(self._extract_content_words(text1.lower()))
        words2 = set(self._extract_content_words(text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_content_words(self, text: str) -> List[str]:
        """Extract content words (remove stop words and punctuation)"""
        # Simplified content word extraction
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 
            'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _evaluate_answer_length(self, answer: str, question: str) -> float:
        """Evaluate appropriateness of answer length"""
        answer_length = len(answer.split())
        
        # Determine expected length based on question type
        if any(word in question.lower() for word in ['what', 'who', 'when', 'where']):
            # Factual questions expect shorter answers
            ideal_range = (5, 30)
        elif any(word in question.lower() for word in ['how', 'why', 'explain']):
            # Explanatory questions expect longer answers
            ideal_range = (15, 100)
        else:
            # General questions
            ideal_range = (10, 50)
        
        min_length, max_length = ideal_range
        
        if min_length <= answer_length <= max_length:
            return 1.0
        elif answer_length < min_length:
            return answer_length / min_length
        else:
            return max_length / answer_length
    
    def _evaluate_question_type_match(self, question: str, answer: str) -> float:
        """Evaluate if answer matches question type"""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Yes/No questions
        if any(word in question_lower for word in ['is', 'are', 'can', 'does', 'do', 'will']):
            if any(word in answer_lower for word in ['yes', 'no', 'true', 'false']):
                return 1.0
            return 0.7  # Partial credit for explanatory answers
        
        # Who questions
        if 'who' in question_lower:
            if any(word in answer_lower for word in ['person', 'people', 'he', 'she', 'they']):
                return 1.0
            return 0.5
        
        # When questions
        if 'when' in question_lower:
            if re.search(r'\d{4}|\b(morning|afternoon|evening|today|yesterday|tomorrow)\b', answer_lower):
                return 1.0
            return 0.5
        
        # Where questions  
        if 'where' in question_lower:
            if any(word in answer_lower for word in ['in', 'at', 'on', 'location', 'place']):
                return 1.0
            return 0.5
        
        # Default case
        return 0.8
    
    def _check_basic_grammar(self, text: str) -> float:
        """Basic grammar checking (simplified)"""
        # This is a very simplified grammar check
        # In practice, you would use proper grammar checking tools
        
        # Check for basic sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:  # Minimum sentence length
                # Check for basic capitalization
                if sentence[0].isupper():
                    valid_sentences += 1
        
        total_sentences = len([s for s in sentences if s.strip()])
        return valid_sentences / total_sentences if total_sentences > 0 else 0.0
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified)"""
        # Simplified readability based on sentence and word length
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Optimal ranges
        ideal_sentence_length = 15  # words per sentence
        ideal_word_length = 5  # characters per word
        
        sentence_score = 1.0 - abs(avg_sentence_length - ideal_sentence_length) / ideal_sentence_length
        word_score = 1.0 - abs(avg_word_length - ideal_word_length) / ideal_word_length
        
        return max(0.0, (sentence_score + word_score) / 2)
    
    def _evaluate_coherence(self, text: str) -> float:
        """Evaluate text coherence (simplified)"""
        # Simplified coherence evaluation
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is trivially coherent
        
        # Check for transition words and pronouns
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 
                          'consequently', 'nevertheless', 'thus', 'hence', 'also', 'furthermore']
        pronouns = ['it', 'this', 'that', 'these', 'those', 'they', 'them']
        
        coherence_indicators = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in transition_words + pronouns):
                coherence_indicators += 1
        
        return min(1.0, coherence_indicators / len(sentences))
    
    def _compute_weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted overall QuestEval score"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        weight_map = {
            "consistency": self.config.consistency_weight,
            "answerability": self.config.answerability_weight,
            "source_support": self.config.source_support_weight,
            "fluency": self.config.fluency_weight
        }
        
        for component, score in scores.items():
            weight = weight_map.get(component, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0