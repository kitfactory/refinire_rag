"""
ROUGE Evaluator Implementation

ROUGE評価器の実装
ROUGEは文書要約の評価で広く使用される指標で、n-gramやLCSの重複に基づいて評価を行います。
"""

from typing import List, Dict, Any, Optional, Union, Set, Tuple
from collections import Counter
import re

from .base_evaluator import ReferenceBasedEvaluator, BaseEvaluatorConfig, EvaluationScore


class ROUGEConfig(BaseEvaluatorConfig):
    """
    Configuration for ROUGE evaluator
    
    ROUGE評価器の設定
    """
    name: str = "ROUGE"
    rouge_types: List[str] = None  # Types of ROUGE to compute
    max_n: int = 2  # Maximum n-gram for ROUGE-N
    use_stemming: bool = False  # Enable stemming (simplified)
    case_sensitive: bool = False  # Case sensitivity
    remove_stopwords: bool = False  # Remove stopwords
    
    def __post_init__(self):
        super().__post_init__()
        if self.rouge_types is None:
            self.rouge_types = ["rouge-1", "rouge-2", "rouge-l"]


class ROUGEEvaluator(ReferenceBasedEvaluator):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) evaluator
    
    ROUGE（要約評価のためのリコール指向代替手法）評価器
    
    ROUGEは以下のバリアントを提供します：
    1. ROUGE-N: N-gramの重複に基づく評価
    2. ROUGE-L: 最長共通部分列（LCS）に基づく評価
    3. ROUGE-W: 重み付きLCSに基づく評価
    4. ROUGE-S: Skip-bigramに基づく評価
    """
    
    def __init__(self, config: ROUGEConfig):
        super().__init__(config)
        self.config: ROUGEConfig = config
        self.stopwords = self._get_stopwords() if config.remove_stopwords else set()
    
    def evaluate(self,
                reference: Union[str, List[str]],
                candidate: str,
                context: Optional[Dict[str, Any]] = None) -> EvaluationScore:
        """
        Evaluate candidate answer using ROUGE metrics
        
        ROUGEメトリクスを使用して候補回答を評価
        
        Args:
            reference: Reference answer(s)
                      参照回答
            candidate: Candidate answer to evaluate
                      評価する候補回答
            context: Additional context (not used in ROUGE)
                    追加コンテキスト（ROUGEでは未使用）
                    
        Returns:
            EvaluationScore: ROUGE evaluation result
                           ROUGE評価結果
        """
        self.validate_inputs(reference, candidate)
        
        # Preprocess inputs
        candidate = self.preprocess_text(candidate)
        if isinstance(reference, str):
            references = [self.preprocess_text(reference)]
        else:
            references = [self.preprocess_text(ref) for ref in reference]
        
        # Calculate ROUGE scores for all requested types
        rouge_scores = {}
        detailed_scores = {}
        
        for rouge_type in self.config.rouge_types:
            if rouge_type.startswith("rouge-n") or rouge_type.startswith("rouge-"):
                try:
                    if rouge_type == "rouge-l":
                        scores = self._calculate_rouge_l(references, candidate)
                    elif rouge_type == "rouge-w":
                        scores = self._calculate_rouge_w(references, candidate)
                    elif rouge_type == "rouge-s":
                        scores = self._calculate_rouge_s(references, candidate)
                    elif rouge_type.startswith("rouge-"):
                        # Extract n from rouge-n
                        n = int(rouge_type.split("-")[1])
                        scores = self._calculate_rouge_n(references, candidate, n)
                    else:
                        continue
                    
                    rouge_scores[rouge_type] = scores["f_score"]
                    detailed_scores[rouge_type] = scores
                    
                except (ValueError, IndexError):
                    # Invalid rouge type, skip
                    continue
        
        # Calculate overall ROUGE score (average of F-scores)
        overall_score = sum(rouge_scores.values()) / len(rouge_scores) if rouge_scores else 0.0
        
        # Detailed breakdown
        details = {
            "rouge_scores": rouge_scores,
            "detailed_scores": detailed_scores,
            "rouge_types_computed": list(rouge_scores.keys()),
            "preprocessing": {
                "case_sensitive": self.config.case_sensitive,
                "remove_stopwords": self.config.remove_stopwords,
                "use_stemming": self.config.use_stemming
            },
            "overall_computation": f"Average F-score of {len(rouge_scores)} ROUGE variants"
        }
        
        return EvaluationScore(
            metric_name=self.config.name,
            score=overall_score,
            details=details,
            confidence=0.85  # ROUGE is reliable for content overlap assessment
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words with optional preprocessing
        
        オプションの前処理付きでテキストを単語にトークン化
        """
        if not self.config.case_sensitive:
            text = text.lower()
        
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords if enabled
        if self.config.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming if enabled (simplified)
        if self.config.use_stemming:
            tokens = [self._stem(token) for token in tokens]
        
        return tokens
    
    def _calculate_rouge_n(self, references: List[str], candidate: str, n: int) -> Dict[str, float]:
        """
        Calculate ROUGE-N scores
        
        ROUGE-Nスコアを計算
        """
        candidate_tokens = self._tokenize(candidate)
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        
        if not candidate_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
        
        # Calculate best scores across all references
        best_precision = 0.0
        best_recall = 0.0
        best_f_score = 0.0
        
        for reference in references:
            ref_tokens = self._tokenize(reference)
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            
            if not ref_ngrams:
                continue
            
            # Count matching n-grams
            candidate_ngram_counts = Counter(candidate_ngrams)
            ref_ngram_counts = Counter(ref_ngrams)
            
            matches = 0
            for ngram, count in candidate_ngram_counts.items():
                matches += min(count, ref_ngram_counts[ngram])
            
            # Calculate precision, recall, and F-score
            precision = matches / len(candidate_ngrams) if candidate_ngrams else 0.0
            recall = matches / len(ref_ngrams) if ref_ngrams else 0.0
            f_score = self._calculate_f_score(precision, recall)
            
            # Keep best scores
            if f_score > best_f_score:
                best_precision = precision
                best_recall = recall
                best_f_score = f_score
        
        return {
            "precision": best_precision,
            "recall": best_recall,
            "f_score": best_f_score
        }
    
    def _calculate_rouge_l(self, references: List[str], candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE-L scores based on Longest Common Subsequence
        
        最長共通部分列に基づくROUGE-Lスコアを計算
        """
        candidate_tokens = self._tokenize(candidate)
        
        if not candidate_tokens:
            return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
        
        # Calculate best scores across all references
        best_precision = 0.0
        best_recall = 0.0
        best_f_score = 0.0
        
        for reference in references:
            ref_tokens = self._tokenize(reference)
            
            if not ref_tokens:
                continue
            
            # Calculate LCS length
            lcs_length = self._lcs_length(ref_tokens, candidate_tokens)
            
            # Calculate precision, recall, and F-score
            precision = lcs_length / len(candidate_tokens) if candidate_tokens else 0.0
            recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
            f_score = self._calculate_f_score(precision, recall)
            
            # Keep best scores
            if f_score > best_f_score:
                best_precision = precision
                best_recall = recall
                best_f_score = f_score
        
        return {
            "precision": best_precision,
            "recall": best_recall,
            "f_score": best_f_score
        }
    
    def _calculate_rouge_w(self, references: List[str], candidate: str, weight: float = 1.2) -> Dict[str, float]:
        """
        Calculate ROUGE-W scores based on Weighted Longest Common Subsequence
        
        重み付き最長共通部分列に基づくROUGE-Wスコアを計算
        """
        candidate_tokens = self._tokenize(candidate)
        
        if not candidate_tokens:
            return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
        
        # Calculate best scores across all references
        best_precision = 0.0
        best_recall = 0.0
        best_f_score = 0.0
        
        for reference in references:
            ref_tokens = self._tokenize(reference)
            
            if not ref_tokens:
                continue
            
            # Calculate weighted LCS
            wlcs_score = self._weighted_lcs(ref_tokens, candidate_tokens, weight)
            
            # Calculate precision, recall, and F-score
            # For ROUGE-W, we use the weighted score normalized by sequence lengths
            candidate_weight_sum = len(candidate_tokens) ** weight if candidate_tokens else 0.0
            ref_weight_sum = len(ref_tokens) ** weight if ref_tokens else 0.0
            
            precision = wlcs_score / candidate_weight_sum if candidate_weight_sum else 0.0
            recall = wlcs_score / ref_weight_sum if ref_weight_sum else 0.0
            f_score = self._calculate_f_score(precision, recall)
            
            # Keep best scores
            if f_score > best_f_score:
                best_precision = precision
                best_recall = recall
                best_f_score = f_score
        
        return {
            "precision": best_precision,
            "recall": best_recall,
            "f_score": best_f_score
        }
    
    def _calculate_rouge_s(self, references: List[str], candidate: str, max_skip: int = 4) -> Dict[str, float]:
        """
        Calculate ROUGE-S scores based on skip-bigrams
        
        Skip-bigramに基づくROUGE-Sスコアを計算
        """
        candidate_tokens = self._tokenize(candidate)
        candidate_skip_bigrams = self._get_skip_bigrams(candidate_tokens, max_skip)
        
        if not candidate_skip_bigrams:
            return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}
        
        # Calculate best scores across all references
        best_precision = 0.0
        best_recall = 0.0
        best_f_score = 0.0
        
        for reference in references:
            ref_tokens = self._tokenize(reference)
            ref_skip_bigrams = self._get_skip_bigrams(ref_tokens, max_skip)
            
            if not ref_skip_bigrams:
                continue
            
            # Count matching skip-bigrams
            candidate_bigram_counts = Counter(candidate_skip_bigrams)
            ref_bigram_counts = Counter(ref_skip_bigrams)
            
            matches = 0
            for bigram, count in candidate_bigram_counts.items():
                matches += min(count, ref_bigram_counts[bigram])
            
            # Calculate precision, recall, and F-score
            precision = matches / len(candidate_skip_bigrams) if candidate_skip_bigrams else 0.0
            recall = matches / len(ref_skip_bigrams) if ref_skip_bigrams else 0.0
            f_score = self._calculate_f_score(precision, recall)
            
            # Keep best scores
            if f_score > best_f_score:
                best_precision = precision
                best_recall = recall
                best_f_score = f_score
        
        return {
            "precision": best_precision,
            "recall": best_recall,
            "f_score": best_f_score
        }
    
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
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """
        Calculate the length of the Longest Common Subsequence
        
        最長共通部分列の長さを計算
        """
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _weighted_lcs(self, seq1: List[str], seq2: List[str], weight: float) -> float:
        """
        Calculate weighted LCS score
        
        重み付きLCSスコアを計算
        """
        m, n = len(seq1), len(seq2)
        
        # Dynamic programming for weighted LCS
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    # Find the length of the matching subsequence
                    match_length = 1
                    k = 1
                    while (i - k >= 1 and j - k >= 1 and 
                           seq1[i - k - 1] == seq2[j - k - 1]):
                        match_length += 1
                        k += 1
                    
                    # Apply weight to consecutive matches
                    weighted_score = match_length ** weight
                    dp[i][j] = dp[i - 1][j - 1] + weighted_score
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    def _get_skip_bigrams(self, tokens: List[str], max_skip: int) -> List[tuple]:
        """
        Extract skip-bigrams from token list
        
        トークンリストからskip-bigramを抽出
        """
        skip_bigrams = []
        
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + max_skip + 2, len(tokens))):
                skip_bigram = (tokens[i], tokens[j])
                skip_bigrams.append(skip_bigram)
        
        return skip_bigrams
    
    def _calculate_f_score(self, precision: float, recall: float, beta: float = 1.0) -> float:
        """
        Calculate F-score (F1 by default)
        
        F-スコアを計算（デフォルトはF1）
        """
        if precision + recall == 0:
            return 0.0
        
        beta_squared = beta ** 2
        f_score = (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        return f_score
    
    def _stem(self, word: str) -> str:
        """
        Simple stemming (remove common suffixes)
        
        簡易ステミング（一般的な接尾語を除去）
        """
        # Very simplified stemming - just remove common English suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment']
        
        word_lower = word.lower()
        for suffix in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return word_lower[:-len(suffix)]
        
        return word_lower
    
    def _get_stopwords(self) -> Set[str]:
        """
        Get a basic set of English stopwords
        
        英語の基本的なストップワードセットを取得
        """
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'the', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'am', 'is', 'are', 'was', 'were', 'been', 'being'
        }
    
    def calculate_rouge_variants(self,
                                reference: Union[str, List[str]],
                                candidate: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate all ROUGE variants with detailed metrics
        
        すべてのROUGEバリアントを詳細メトリクスで計算
        """
        # Preprocess inputs
        candidate = self.preprocess_text(candidate)
        if isinstance(reference, str):
            references = [self.preprocess_text(reference)]
        else:
            references = [self.preprocess_text(ref) for ref in reference]
        
        rouge_variants = {}
        
        # ROUGE-1 and ROUGE-2
        for n in [1, 2]:
            rouge_variants[f"ROUGE-{n}"] = self._calculate_rouge_n(references, candidate, n)
        
        # ROUGE-L
        rouge_variants["ROUGE-L"] = self._calculate_rouge_l(references, candidate)
        
        # ROUGE-W (optional)
        rouge_variants["ROUGE-W"] = self._calculate_rouge_w(references, candidate)
        
        # ROUGE-S (optional)
        rouge_variants["ROUGE-S"] = self._calculate_rouge_s(references, candidate)
        
        return rouge_variants