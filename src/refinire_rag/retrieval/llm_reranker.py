"""
LLM-based document reranker using RefinireAgent

Uses Large Language Models to evaluate relevance between queries and documents,
providing high-quality semantic reranking capabilities.
"""

import logging
import os
import time
import json
from typing import List, Optional, Dict, Any, Type

from .base import Reranker, RerankerConfig, SearchResult
from ..config import RefinireRAGConfig
from ..utils.model_config import get_default_llm_model

logger = logging.getLogger(__name__)


class LLMRerankerConfig(RerankerConfig):
    """Configuration for LLM Reranker
    
    LLM（大規模言語モデル）リランカーの設定
    """
    
    def __init__(self,
                 top_k: int = 5,
                 score_threshold: float = 0.0,
                 llm_model: str = None,
                 temperature: float = 0.1,
                 max_tokens: int = 100,
                 batch_size: int = 5,
                 use_chain_of_thought: bool = True,
                 scoring_method: str = "numerical",  # "numerical" or "ranking"
                 fallback_on_error: bool = True,
                 **kwargs):
        """Initialize LLM reranker configuration
        
        Args:
            top_k: Maximum number of results to return
                   返す結果の最大数
            score_threshold: Minimum score threshold for results
                           結果の最小スコア閾値
            llm_model: LLM model to use (defaults to environment setting)
                      使用するLLMモデル（環境設定のデフォルト）
            temperature: Temperature for LLM generation
                        LLM生成の温度パラメータ
            max_tokens: Maximum tokens for LLM response
                       LLM応答の最大トークン数
            batch_size: Number of documents to process in one LLM call
                       1回のLLM呼び出しで処理する文書数
            use_chain_of_thought: Use reasoning in prompts
                                思考の連鎖をプロンプトで使用
            scoring_method: Method for scoring ("numerical" or "ranking")
                           スコアリング方法（"numerical"または"ranking"）
            fallback_on_error: Return original results on LLM error
                              LLMエラー時に元の結果を返す
        """
        super().__init__(top_k=top_k, 
                        rerank_model="llm_semantic",
                        score_threshold=score_threshold)
        self.llm_model = llm_model or get_default_llm_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_chain_of_thought = use_chain_of_thought
        self.scoring_method = scoring_method
        self.fallback_on_error = fallback_on_error
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_env(cls) -> "LLMRerankerConfig":
        """Create configuration from environment variables
        
        環境変数からLLMRerankerConfigインスタンスを作成します。
        
        Returns:
            LLMRerankerConfig instance with values from environment
        """
        config = RefinireRAGConfig()
        
        # Get configuration values from environment
        top_k = config.reranker_top_k  # Uses REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K
        score_threshold = float(os.getenv("REFINIRE_RAG_LLM_RERANKER_SCORE_THRESHOLD", "0.0"))
        llm_model = os.getenv("REFINIRE_RAG_LLM_RERANKER_MODEL") or get_default_llm_model()
        temperature = float(os.getenv("REFINIRE_RAG_LLM_RERANKER_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("REFINIRE_RAG_LLM_RERANKER_MAX_TOKENS", "100"))
        batch_size = int(os.getenv("REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", "5"))
        use_chain_of_thought = os.getenv("REFINIRE_RAG_LLM_RERANKER_USE_COT", "true").lower() == "true"
        scoring_method = os.getenv("REFINIRE_RAG_LLM_RERANKER_SCORING_METHOD", "numerical")
        fallback_on_error = os.getenv("REFINIRE_RAG_LLM_RERANKER_FALLBACK_ON_ERROR", "true").lower() == "true"
        
        return cls(
            top_k=top_k,
            score_threshold=score_threshold,
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            batch_size=batch_size,
            use_chain_of_thought=use_chain_of_thought,
            scoring_method=scoring_method,
            fallback_on_error=fallback_on_error
        )


class LLMReranker(Reranker):
    """LLM-based document reranker
    
    LLM（大規模言語モデル）ベースの文書リランカー
    
    Uses Large Language Models to evaluate semantic relevance between
    queries and documents, providing high-quality reranking.
    
    大規模言語モデルを使用してクエリと文書間の意味的関連性を評価し、
    高品質な再ランキングを提供します。
    """
    
    def __init__(self, 
                 config: Optional[LLMRerankerConfig] = None,
                 top_k: Optional[int] = None,
                 score_threshold: Optional[float] = None,
                 llm_model: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 use_chain_of_thought: Optional[bool] = None,
                 scoring_method: Optional[str] = None,
                 fallback_on_error: Optional[bool] = None,
                 **kwargs):
        """Initialize LLM Reranker
        
        LLMリランカーを初期化
        
        Args:
            config: Reranker configuration (optional, can be created from other args)
            top_k: Maximum number of results to return (default from env or 5)
            score_threshold: Minimum score threshold (default from env or 0.0)
            llm_model: LLM model to use (default from env or auto-detect)
            temperature: Temperature for LLM generation (default from env or 0.1)
            max_tokens: Maximum tokens for LLM response (default from env or 100)
            batch_size: Documents per LLM call (default from env or 5)
            use_chain_of_thought: Use reasoning in prompts (default from env or True)
            scoring_method: Scoring method (default from env or "numerical")
            fallback_on_error: Fallback to original on error (default from env or True)
            **kwargs: Additional configuration parameters
        """
        # If config is provided, use it directly
        if config is not None:
            super().__init__(config)
        else:
            # Create config using keyword arguments with environment variable fallback
            actual_top_k = self._get_setting(top_k, "REFINIRE_RAG_LLM_RERANKER_TOP_K", 5, int)
            actual_score_threshold = self._get_setting(score_threshold, "REFINIRE_RAG_LLM_RERANKER_SCORE_THRESHOLD", 0.0, float)
            actual_llm_model = self._get_setting(llm_model, "REFINIRE_RAG_LLM_RERANKER_MODEL", get_default_llm_model(), str)
            actual_temperature = self._get_setting(temperature, "REFINIRE_RAG_LLM_RERANKER_TEMPERATURE", 0.1, float)
            actual_max_tokens = self._get_setting(max_tokens, "REFINIRE_RAG_LLM_RERANKER_MAX_TOKENS", 100, int)
            actual_batch_size = self._get_setting(batch_size, "REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", 5, int)
            actual_use_chain_of_thought = self._get_setting(use_chain_of_thought, "REFINIRE_RAG_LLM_RERANKER_USE_COT", True, bool)
            actual_scoring_method = self._get_setting(scoring_method, "REFINIRE_RAG_LLM_RERANKER_SCORING_METHOD", "numerical", str)
            actual_fallback_on_error = self._get_setting(fallback_on_error, "REFINIRE_RAG_LLM_RERANKER_FALLBACK_ON_ERROR", True, bool)
            
            # Create config with resolved values
            config = LLMRerankerConfig(
                top_k=actual_top_k,
                score_threshold=actual_score_threshold,
                llm_model=actual_llm_model,
                temperature=actual_temperature,
                max_tokens=actual_max_tokens,
                batch_size=actual_batch_size,
                use_chain_of_thought=actual_use_chain_of_thought,
                scoring_method=actual_scoring_method,
                fallback_on_error=actual_fallback_on_error,
                **kwargs
            )
            super().__init__(config)
        
        # Initialize LLM client (using refinire's get_llm)
        self._llm_client = None
        self._initialize_llm()
        
        logger.info(f"Initialized LLMReranker with model: {self.config.llm_model}")
    
    def _get_setting(self, value, env_var, default, value_type=str):
        """Get configuration setting from argument, environment variable, or default
        
        設定値を引数、環境変数、またはデフォルト値から取得
        
        Args:
            value: Direct argument value
            env_var: Environment variable name
            default: Default value if neither argument nor env var is set
            value_type: Type to convert to (str, int, bool, float)
            
        Returns:
            Configuration value with proper type
        """
        if value is not None:
            return value
        
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if value_type == bool:
                return env_value.lower() in ('true', '1', 'yes', 'on')
            elif value_type == int:
                try:
                    return int(env_value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_var}: {env_value}, using default: {default}")
                    return default
            elif value_type == float:
                try:
                    return float(env_value)
                except ValueError:
                    logger.warning(f"Invalid float value for {env_var}: {env_value}, using default: {default}")
                    return default
            else:
                return env_value
        
        return default
    
    def _initialize_llm(self):
        """Initialize LLM client using RefinireAgent
        
        RefinireAgentを使用してLLMクライアントを初期化
        """
        try:
            from refinire import RefinireAgent
            self._refinire_agent = RefinireAgent(
                name="llm_reranker",
                generation_instructions="You are an expert information retrieval system that evaluates document relevance.",
                model=self.config.llm_model,
                session_history=None,  # Disable session history for independent evaluations
                history_size=0  # No history retention
            )
            self._use_refinire = True
            logger.debug(f"Initialized LLM reranker with RefinireAgent, model: {self.config.llm_model}")
        except ImportError:
            logger.warning("Refinire library not available, LLM reranking will be disabled")
            self._refinire_agent = None
            self._use_refinire = False
        except Exception as e:
            logger.error(f"Failed to initialize RefinireAgent: {e}")
            self._refinire_agent = None
            self._use_refinire = False
    
    @classmethod
    def get_config_class(cls) -> Type[LLMRerankerConfig]:
        """Get configuration class for this reranker
        
        このリランカーの設定クラスを取得
        """
        return LLMRerankerConfig
    
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results using LLM evaluation
        
        LLM評価を使用して検索結果を再ランク
        
        Args:
            query: Original search query
                  元の検索クエリ
            results: Initial search results to rerank
                    再ランクする初期検索結果
            
        Returns:
            Reranked search results using LLM scores
            LLMスコアを使用した再ランク済み検索結果
        """
        start_time = time.time()
        
        try:
            logger.debug(f"LLM reranking {len(results)} results for query: '{query}'")
            
            if not results:
                return []
            
            # Check if LLM is available
            if not self._use_refinire or self._refinire_agent is None:
                if self.config.fallback_on_error:
                    logger.warning("LLM not available, returning original results")
                    return results[:self.config.top_k]
                else:
                    raise RuntimeError("LLM client not initialized")
            
            # Process results in batches
            llm_scores = self._evaluate_relevance_batch(query, results)
            
            # Create reranked results
            reranked_results = self._create_reranked_results(results, llm_scores)
            
            # Sort by LLM score (descending)
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            # Apply top_k limit and score threshold
            final_results = []
            for result in reranked_results:
                if len(final_results) >= self.config.top_k:
                    break
                if result.score >= self.config.score_threshold:
                    final_results.append(result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"LLM reranked {len(results)} → {len(final_results)} results in {processing_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"LLM reranking failed: {e}")
            
            if self.config.fallback_on_error:
                return results[:self.config.top_k]  # Fallback to original order
            else:
                raise
    
    def _evaluate_relevance_batch(self, query: str, results: List[SearchResult]) -> Dict[str, float]:
        """Evaluate relevance scores for documents in batches
        
        文書の関連性スコアをバッチで評価
        """
        llm_scores = {}
        
        # Process results in batches
        for i in range(0, len(results), self.config.batch_size):
            batch = results[i:i + self.config.batch_size]
            batch_scores = self._evaluate_batch(query, batch)
            llm_scores.update(batch_scores)
        
        return llm_scores
    
    def _evaluate_batch(self, query: str, batch: List[SearchResult]) -> Dict[str, float]:
        """Evaluate a single batch of documents
        
        単一バッチの文書を評価
        """
        if self.config.scoring_method == "numerical":
            return self._evaluate_numerical_batch(query, batch)
        elif self.config.scoring_method == "ranking":
            return self._evaluate_ranking_batch(query, batch)
        else:
            raise ValueError(f"Unknown scoring method: {self.config.scoring_method}")
    
    def _evaluate_numerical_batch(self, query: str, batch: List[SearchResult]) -> Dict[str, float]:
        """Evaluate batch using numerical scoring (0-10 scale)
        
        数値スコアリング（0-10スケール）を使用してバッチを評価
        """
        # Prepare documents for evaluation
        docs_text = []
        doc_ids = []
        
        for result in batch:
            # Truncate content if too long
            content = result.document.content
            if len(content) > 2000:  # Increased limit for better context
                content = content[:2000] + "..."
            
            docs_text.append(content)
            doc_ids.append(result.document_id)
        
        # Create evaluation prompt
        prompt = self._create_numerical_prompt(query, docs_text, doc_ids)
        
        try:
            logger.info(f"[DEBUG] LLM Reranker - Processing {len(batch)} documents for query: {query}")
            logger.info(f"[DEBUG] Document IDs: {doc_ids}")
            
            # Call LLM using RefinireAgent
            if self._use_refinire and self._refinire_agent:
                logger.info("[DEBUG] Using RefinireAgent run method")
                result = self._refinire_agent.run(prompt)
                response = result.content if hasattr(result, 'content') else str(result)
            else:
                raise RuntimeError("No LLM client available for reranking")
            
            logger.info(f"[DEBUG] LLM Response length: {len(response) if response else 0}")
            logger.info(f"[DEBUG] LLM Response preview: {response[:200] if response else 'None'}...")
            
            # Parse scores from response
            scores = self._parse_numerical_response(response, doc_ids)
            logger.info(f"[DEBUG] Parsed scores: {scores}")
            
            # Normalize scores to [0, 1] range
            normalized_scores = {}
            for doc_id, score in scores.items():
                normalized_scores[doc_id] = min(max(score / 10.0, 0.0), 1.0)
            
            logger.info(f"[DEBUG] Normalized scores: {normalized_scores}")
            return normalized_scores
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for batch: {e}")
            logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
            logger.error(f"[DEBUG] Exception details: {str(e)}")
            # Return original scores as fallback
            fallback_scores = {result.document_id: result.score for result in batch}
            logger.warning(f"[DEBUG] Using fallback scores: {fallback_scores}")
            return fallback_scores
    
    def _evaluate_ranking_batch(self, query: str, batch: List[SearchResult]) -> Dict[str, float]:
        """Evaluate batch using ranking method
        
        ランキング方法を使用してバッチを評価
        """
        # For ranking method, we ask LLM to rank documents
        docs_text = []
        doc_ids = []
        
        for result in batch:
            content = result.document.content
            if len(content) > 1000:
                content = content[:1000] + "..."
            docs_text.append(content)
            doc_ids.append(result.document_id)
        
        prompt = self._create_ranking_prompt(query, docs_text, doc_ids)
        
        try:
            # Call LLM using refinire interface
            if hasattr(self._llm_client, 'generate'):
                response = self._llm_client.generate(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            elif hasattr(self._llm_client, '__call__'):
                # Use refinire LLM interface with direct call
                try:
                    result = self._llm_client(prompt)
                    response = result.content if hasattr(result, 'content') else str(result)
                except Exception as call_error:
                    logger.warning(f"Direct call failed: {call_error}, trying invoke method")
                    if hasattr(self._llm_client, 'invoke'):
                        result = self._llm_client.invoke({'input': prompt})
                        response = result.content if hasattr(result, 'content') else str(result)
                    else:
                        raise call_error
            else:
                raise AttributeError("LLM client has no generate or __call__ method")
            
            # Parse ranking from response
            ranking = self._parse_ranking_response(response, doc_ids)
            
            # Convert ranking to scores (higher rank = higher score)
            scores = {}
            total_docs = len(doc_ids)
            for doc_id, rank in ranking.items():
                # Rank 1 gets highest score, rank N gets lowest score
                normalized_score = (total_docs - rank + 1) / total_docs
                scores[doc_id] = normalized_score
            
            return scores
            
        except Exception as e:
            logger.error(f"LLM ranking failed for batch: {e}")
            return {result.document_id: result.score for result in batch}
    
    def _create_numerical_prompt(self, query: str, docs_text: List[str], doc_ids: List[str]) -> str:
        """Create prompt for numerical scoring
        
        数値スコアリング用のプロンプトを作成
        """
        thinking_prompt = f"""
Think step by step about how relevant each document is to the query: "{query}"

Evaluation criteria (in order of importance):
1. DIRECT TOPIC RELEVANCE: Does the document directly discuss "{query}" as its main or significant topic?
2. EXPLICIT MENTION: Does the document explicitly mention the key terms from "{query}"?
3. CONTENT FOCUS: How much of the document content is specifically about "{query}" vs. related but different topics?
4. ANSWER COMPLETENESS: How well does the document answer the specific question "{query}"?

IMPORTANT: Prioritize documents that directly address the query topic over documents that only mention related concepts. A document specifically about "{query}" should always score higher than a document about a related but different topic.

""" if self.config.use_chain_of_thought else ""
        
        prompt = f"""You are an expert information retrieval system. Your task is to evaluate how relevant each document is to the given query.

Query: "{query}"

{thinking_prompt}Rate each document on a scale of 0-10 where:
- 10: Perfectly relevant - document directly discusses "{query}" as main topic and provides comprehensive information
- 8-9: Highly relevant - document specifically addresses "{query}" with detailed information
- 6-7: Moderately relevant - document mentions "{query}" and provides some specific information
- 3-5: Slightly relevant - document mentions concepts related to "{query}" but doesn't focus on it
- 1-2: Minimally relevant - document only tangentially mentions related concepts
- 0: Not relevant - document doesn't address "{query}" or related concepts

Focus on DIRECT TOPICAL RELEVANCE. A document that specifically discusses "{query}" should score much higher than a document that only mentions related business concepts.

Documents to evaluate:
"""
        
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, docs_text)):
            prompt += f"\nDocument {doc_id}:\n{doc_text}\n"
        
        prompt += f"""
Please provide scores in the following JSON format:
{{
    "scores": {{
"""
        
        for i, doc_id in enumerate(doc_ids):
            comma = "," if i < len(doc_ids) - 1 else ""
            prompt += f'        "{doc_id}": <score>{comma}\n'
        
        prompt += """    }
}"""
        
        return prompt
    
    def _create_ranking_prompt(self, query: str, docs_text: List[str], doc_ids: List[str]) -> str:
        """Create prompt for ranking method
        
        ランキング方法用のプロンプトを作成
        """
        thinking_prompt = """
Think step by step about the relevance of each document to the query.
Consider the same factors as in numerical scoring, then rank them from most to least relevant.

""" if self.config.use_chain_of_thought else ""
        
        prompt = f"""You are an expert information retrieval system. Your task is to rank documents by relevance to the given query.

Query: "{query}"

{thinking_prompt}Documents to rank:
"""
        
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, docs_text)):
            prompt += f"\nDocument {doc_id}:\n{doc_text}\n"
        
        prompt += f"""
Please rank the documents from most relevant (rank 1) to least relevant.
Provide rankings in the following JSON format:
{{
    "rankings": {{
"""
        
        for i, doc_id in enumerate(doc_ids):
            comma = "," if i < len(doc_ids) - 1 else ""
            prompt += f'        "{doc_id}": <rank>{comma}\n'
        
        prompt += """    }
}"""
        
        return prompt
    
    def _parse_numerical_response(self, response: str, doc_ids: List[str]) -> Dict[str, float]:
        """Parse numerical scores from LLM response
        
        LLM応答から数値スコアを解析
        """
        try:
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                if "scores" in data:
                    scores = {}
                    for doc_id in doc_ids:
                        if doc_id in data["scores"]:
                            score = float(data["scores"][doc_id])
                            scores[doc_id] = score
                        else:
                            scores[doc_id] = 5.0  # Default middle score
                    return scores
            
            # Fallback: try to extract numbers from response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            scores = {}
            
            for i, doc_id in enumerate(doc_ids):
                if i < len(numbers):
                    scores[doc_id] = float(numbers[i])
                else:
                    scores[doc_id] = 5.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to parse numerical response: {e}")
            # Return default scores
            return {doc_id: 5.0 for doc_id in doc_ids}
    
    def _parse_ranking_response(self, response: str, doc_ids: List[str]) -> Dict[str, int]:
        """Parse ranking from LLM response
        
        LLM応答からランキングを解析
        """
        try:
            # Try to parse JSON response
            if "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                if "rankings" in data:
                    rankings = {}
                    for doc_id in doc_ids:
                        if doc_id in data["rankings"]:
                            rank = int(data["rankings"][doc_id])
                            rankings[doc_id] = rank
                        else:
                            rankings[doc_id] = len(doc_ids)  # Default last rank
                    return rankings
            
            # Fallback: assign default rankings
            rankings = {}
            for i, doc_id in enumerate(doc_ids):
                rankings[doc_id] = i + 1
            
            return rankings
            
        except Exception as e:
            logger.error(f"Failed to parse ranking response: {e}")
            # Return default rankings
            return {doc_id: i + 1 for i, doc_id in enumerate(doc_ids)}
    
    def _create_reranked_results(self, results: List[SearchResult], 
                               llm_scores: Dict[str, float]) -> List[SearchResult]:
        """Create new SearchResult objects with LLM scores
        
        LLMスコアを持つ新しいSearchResultオブジェクトを作成
        """
        reranked_results = []
        
        for result in results:
            document_id = result.document_id
            llm_score = llm_scores.get(document_id, result.score)
            
            reranked_result = SearchResult(
                document_id=result.document_id,
                document=result.document,
                score=llm_score,
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "llm_score": llm_score,
                    "reranked_by": "LLMReranker",
                    "llm_model": self.config.llm_model,
                    "scoring_method": self.config.scoring_method
                }
            )
            reranked_results.append(reranked_result)
        
        return reranked_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with LLM-specific metrics
        
        LLM固有のメトリクスを含む処理統計を取得
        """
        stats = super().get_processing_stats()
        
        # Add LLM-specific stats
        stats.update({
            "reranker_type": "LLMReranker",
            "rerank_model": self.config.rerank_model,
            "score_threshold": self.config.score_threshold,
            "top_k": self.config.top_k,
            "llm_model": self.config.llm_model,
            "temperature": self.config.temperature,
            "batch_size": self.config.batch_size,
            "scoring_method": self.config.scoring_method,
            "llm_available": self._llm_client is not None
        })
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary
        
        現在の設定を辞書として取得
        """
        config_dict = {
            'top_k': self.config.top_k,
            'rerank_model': self.config.rerank_model,
            'score_threshold': self.config.score_threshold,
            'llm_model': self.config.llm_model,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'batch_size': self.config.batch_size,
            'use_chain_of_thought': self.config.use_chain_of_thought,
            'scoring_method': self.config.scoring_method,
            'fallback_on_error': self.config.fallback_on_error
        }
        
        # Add any additional attributes from the config
        for attr_name, attr_value in self.config.__dict__.items():
            if attr_name not in config_dict:
                config_dict[attr_name] = attr_value
                
        return config_dict