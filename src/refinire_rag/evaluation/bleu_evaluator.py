"""
BLEU Evaluator Implementation

BLEU評価器の実装
BLEUは機械翻訳で広く使用される評価指標で、n-gramの一致度に基づいて評価を行います。
"""

from typing import List, Dict, Any, Optional, Union
from collections import Counter, defaultdict
import math
import re

from .base_evaluator import ReferenceBasedEvaluator, BaseEvaluatorConfig, EvaluationScore


class BLEUConfig(BaseEvaluatorConfig):
    """
    Configuration for BLEU evaluator
    
    BLEU評価器の設定
    """
    name: str = "BLEU"
    max_n: int = 4  # Maximum n-gram order (typically 4 for BLEU-4)
    weights: List[float] = None  # Weights for different n-gram orders
    smoothing_function: str = "epsilon"  # "epsilon", "add_one", "exponential"
    epsilon: float = 0.1  # Epsilon value for smoothing
    case_sensitive: bool = False  # Case sensitivity
    
    def __post_init__(self):
        super().__post_init__()
        if self.weights is None:
            # Uniform weights for BLEU-4 (default)
            self.weights = [0.25, 0.25, 0.25, 0.25]
        
        # Ensure weights match max_n
        if len(self.weights) != self.max_n:
            # Adjust weights to match max_n
            if len(self.weights) < self.max_n:
                # Extend with uniform weights
                remaining = self.max_n - len(self.weights)
                weight_per_gram = 1.0 / self.max_n
                self.weights.extend([weight_per_gram] * remaining)
            else:
                # Truncate weights
                self.weights = self.weights[:self.max_n]
            
            # Normalize weights to sum to 1
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]


class BLEUEvaluator(ReferenceBasedEvaluator):
    """
    BLEU (Bilingual Evaluation Understudy) evaluator
    
    BLEU（双言語評価代替手法）評価器
    
    BLEUは以下の要素を組み合わせて評価します：
    1. N-gram精度 (1-gram～4-gram)
    2. 短文ペナルティ (BP: Brevity Penalty)
    3. スムージング機能
    """
    
    def __init__(self, config: BLEUConfig):
        super().__init__(config)
        self.config: BLEUConfig = config
    
    def evaluate(self,
                reference: Union[str, List[str]],
                candidate: str,
                context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """
        Evaluate candidate answer using BLEU metric
        
        BLEUメトリクスを使用して候補回答を評価
        
        Args:
            reference: Reference answer(s)
                      参照回答
            candidate: Candidate answer to evaluate
                      評価する候補回答
            context: Additional context (not used in BLEU)
                    追加コンテキスト（BLEUでは未使用）
                    
        Returns:
            EvaluationScore: BLEU evaluation result
                           BLEU評価結果
        """
        self.validate_inputs(reference, candidate)
        
        # Preprocess inputs
        candidate = self.preprocess_text(candidate)
        if isinstance(reference, str):
            references = [self.preprocess_text(reference)]
        else:
            references = [self.preprocess_text(ref) for ref in reference]
        
        # Tokenize texts
        candidate_tokens = self._tokenize(candidate)
        reference_token_lists = [self._tokenize(ref) for ref in references]
        
        # Calculate BLEU score
        bleu_score = self._calculate_bleu(reference_token_lists, candidate_tokens)
        
        # Calculate individual n-gram precisions for detailed analysis
        n_gram_precisions = self._calculate_n_gram_precisions(reference_token_lists, candidate_tokens)
        
        # Calculate brevity penalty
        bp = self._calculate_brevity_penalty(reference_token_lists, candidate_tokens)
        
        # Detailed breakdown
        details = {
            "n_gram_precisions": {
                f"{i+1}_gram": precision 
                for i, precision in enumerate(n_gram_precisions)
            },
            "brevity_penalty": bp,
            "candidate_length": len(candidate_tokens),
            "reference_lengths": [len(ref_tokens) for ref_tokens in reference_token_lists],
            "weights": self.config.weights,
            "max_n": self.config.max_n,
            "smoothing_function": self.config.smoothing_function,
            "bleu_computation": f"BP * exp(sum(w_i * log(p_i))) where BP={bp:.4f}"
        }
        
        return EvaluationScore(
            metric_name=self.config.name,
            score=bleu_score,
            details=details,
            confidence=0.9  # BLEU is generally reliable for lexical similarity
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        テキストを単語にトークン化
        """
        if not self.config.case_sensitive:
            text = text.lower()
        
        # Simple word tokenization
        # In practice, you might want to use more sophisticated tokenizers
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _calculate_bleu(self, reference_token_lists: List[List[str]], candidate_tokens: List[str]) -> float:
        """
        Calculate BLEU score
        
        BLEUスコアを計算
        """
        if not candidate_tokens:
            return 0.0
        
        # Calculate n-gram precisions
        n_gram_precisions = self._calculate_n_gram_precisions(reference_token_lists, candidate_tokens)
        
        # Apply smoothing if any precision is 0
        smoothed_precisions = self._apply_smoothing(n_gram_precisions, candidate_tokens)
        
        # Calculate geometric mean of precisions
        if any(p <= 0 for p in smoothed_precisions):
            return 0.0
        
        log_precisions = [math.log(p) for p in smoothed_precisions]
        weighted_log_precision = sum(w * log_p for w, log_p in zip(self.config.weights, log_precisions))
        
        # Calculate brevity penalty
        bp = self._calculate_brevity_penalty(reference_token_lists, candidate_tokens)
        
        # Final BLEU score
        bleu_score = bp * math.exp(weighted_log_precision)
        
        return min(1.0, bleu_score)  # Cap at 1.0
    
    def _calculate_n_gram_precisions(self, reference_token_lists: List[List[str]], candidate_tokens: List[str]) -> List[float]:
        """
        Calculate precision for each n-gram order
        
        各n-gramオーダーの精度を計算
        """
        precisions = []
        
        for n in range(1, self.config.max_n + 1):
            # Get n-grams from candidate
            candidate_ngrams = self._get_ngrams(candidate_tokens, n)
            
            if not candidate_ngrams:
                precisions.append(0.0)
                continue
            
            # Get n-grams from all references
            reference_ngram_counts = defaultdict(int)
            for ref_tokens in reference_token_lists:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                ref_ngram_counter = Counter(ref_ngrams)
                
                # Take maximum count across all references for each n-gram
                for ngram, count in ref_ngram_counter.items():
                    reference_ngram_counts[ngram] = max(reference_ngram_counts[ngram], count)
            
            # Count matches
            candidate_ngram_counter = Counter(candidate_ngrams)
            matches = 0
            
            for ngram, count in candidate_ngram_counter.items():
                matches += min(count, reference_ngram_counts[ngram])
            
            # Calculate precision
            precision = matches / len(candidate_ngrams) if candidate_ngrams else 0.0
            precisions.append(precision)
        
        return precisions
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """
        Extract n-grams from token list
        
        トークンリストからn-gramを抽出
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return ngrams
    
    def _calculate_brevity_penalty(self, reference_token_lists: List[List[str]], candidate_tokens: List[str]) -> float:
        """
        Calculate brevity penalty to penalize short candidates
        
        短い候補文を罰する短文ペナルティを計算
        """
        candidate_length = len(candidate_tokens)
        
        if candidate_length == 0:
            return 0.0
        
        # Find the reference length closest to candidate length
        reference_lengths = [len(ref_tokens) for ref_tokens in reference_token_lists]
        closest_ref_length = min(reference_lengths, key=lambda x: abs(x - candidate_length))
        
        if candidate_length >= closest_ref_length:
            return 1.0
        else:
            return math.exp(1 - closest_ref_length / candidate_length)
    
    def _apply_smoothing(self, precisions: List[float], candidate_tokens: List[str]) -> List[float]:
        """
        Apply smoothing to handle zero precisions
        
        ゼロ精度を処理するためのスムージングを適用
        """
        smoothed = []
        candidate_length = len(candidate_tokens)
        
        for i, precision in enumerate(precisions):
            if precision > 0:
                smoothed.append(precision)
            else:
                # Apply smoothing based on configuration
                if self.config.smoothing_function == "epsilon":
                    smoothed_precision = self.config.epsilon
                elif self.config.smoothing_function == "add_one":
                    # Add-one smoothing
                    n_gram_order = i + 1
                    total_ngrams = max(1, candidate_length - n_gram_order + 1)
                    smoothed_precision = 1.0 / (total_ngrams + 1)
                elif self.config.smoothing_function == "exponential":
                    # Exponential decay smoothing
                    smoothed_precision = self.config.epsilon * (0.1 ** i)
                else:
                    smoothed_precision = self.config.epsilon
                
                smoothed.append(smoothed_precision)
        
        return smoothed
    
    def calculate_bleu_variants(self,
                              reference: Union[str, List[str]],
                              candidate: str) -> Dict[str, float]:
        """
        Calculate different BLEU variants (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
        
        異なるBLEUバリアント（BLEU-1、BLEU-2、BLEU-3、BLEU-4）を計算
        """
        # Preprocess inputs
        candidate = self.preprocess_text(candidate)
        if isinstance(reference, str):
            references = [self.preprocess_text(reference)]
        else:
            references = [self.preprocess_text(ref) for ref in reference]
        
        # Tokenize texts
        candidate_tokens = self._tokenize(candidate)
        reference_token_lists = [self._tokenize(ref) for ref in references]
        
        bleu_variants = {}
        
        for n in range(1, 5):  # BLEU-1 to BLEU-4
            # Create temporary config for this variant
            temp_config = BLEUConfig(
                name=f"BLEU-{n}",
                max_n=n,
                weights=[1.0/n] * n  # Uniform weights for this n
            )
            
            # Calculate n-gram precisions up to n
            n_gram_precisions = self._calculate_n_gram_precisions(reference_token_lists, candidate_tokens)[:n]
            
            # Apply smoothing
            smoothed_precisions = self._apply_smoothing(n_gram_precisions, candidate_tokens)
            
            # Calculate BLEU for this variant
            if any(p <= 0 for p in smoothed_precisions):
                bleu_score = 0.0
            else:
                log_precisions = [math.log(p) for p in smoothed_precisions]
                weighted_log_precision = sum(temp_config.weights[i] * log_p for i, log_p in enumerate(log_precisions))
                bp = self._calculate_brevity_penalty(reference_token_lists, candidate_tokens)
                bleu_score = bp * math.exp(weighted_log_precision)
            
            bleu_variants[f"BLEU-{n}"] = min(1.0, bleu_score)
        
        return bleu_variants