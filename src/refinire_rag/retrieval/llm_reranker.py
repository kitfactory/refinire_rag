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

from pydantic import BaseModel, Field
from .base import Reranker, RerankerConfig, SearchResult
from ..config import RefinireRAGConfig
from ..utils.model_config import get_default_llm_model

logger = logging.getLogger(__name__)


class DocumentScore(BaseModel):
    """Simplified Pydantic model for single document score
    
    単一文書スコア用の簡素化されたPydanticモデル
    """
    document_id: str = Field(description="Document identifier")
    score: float = Field(description="Relevance score between 0.0 and 10.0")

class DocumentScores(BaseModel):
    """Pydantic model for structured LLM reranker output
    
    LLMリランカーの構造化出力用Pydanticモデル
    """
    scores: List[DocumentScore] = Field(
        description="List of document scores"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "scores": [
                    {"document_id": "doc_123", "score": 8.5},
                    {"document_id": "doc_456", "score": 6.2}
                ]
            }
        }


class LLMRerankerConfig(RerankerConfig):
    """Configuration for LLM Reranker
    
    LLM（大規模言語モデル）リランカーの設定
    """
    
    def __init__(self,
                 top_k: int = 5,
                 score_threshold: float = 0.0,
                 llm_model: str = None,
                 temperature: float = 0.1,
                 batch_size: int = 5,
                 use_chain_of_thought: bool = False,
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
            batch_size: Number of documents to process in one LLM call
                       1回のLLM呼び出しで処理する文書数
            use_chain_of_thought: Use reasoning in prompts (disabled by default for performance)
                                思考の連鎖をプロンプトで使用（パフォーマンスのためデフォルト無効）
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
        batch_size = int(os.getenv("REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", "5"))
        use_chain_of_thought = os.getenv("REFINIRE_RAG_LLM_RERANKER_USE_COT", "false").lower() == "true"
        scoring_method = os.getenv("REFINIRE_RAG_LLM_RERANKER_SCORING_METHOD", "numerical")
        fallback_on_error = os.getenv("REFINIRE_RAG_LLM_RERANKER_FALLBACK_ON_ERROR", "true").lower() == "true"
        
        return cls(
            top_k=top_k,
            score_threshold=score_threshold,
            llm_model=llm_model,
            temperature=temperature,
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
            batch_size: Documents per LLM call (default from env or 5)
            use_chain_of_thought: Use reasoning in prompts (default from env or False)
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
            actual_batch_size = self._get_setting(batch_size, "REFINIRE_RAG_LLM_RERANKER_BATCH_SIZE", 5, int)
            actual_use_chain_of_thought = self._get_setting(use_chain_of_thought, "REFINIRE_RAG_LLM_RERANKER_USE_COT", False, bool)
            actual_scoring_method = self._get_setting(scoring_method, "REFINIRE_RAG_LLM_RERANKER_SCORING_METHOD", "numerical", str)
            actual_fallback_on_error = self._get_setting(fallback_on_error, "REFINIRE_RAG_LLM_RERANKER_FALLBACK_ON_ERROR", True, bool)
            
            # Create config with resolved values
            config = LLMRerankerConfig(
                top_k=actual_top_k,
                score_threshold=actual_score_threshold,
                llm_model=actual_llm_model,
                temperature=actual_temperature,
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
        """Initialize LLM client using RefinireAgent with structured output
        
        構造化出力対応のRefinireAgentを使用してLLMクライアントを初期化
        """
        try:
            from refinire import RefinireAgent
            # Try simplified structured output with list format
            self._refinire_agent = RefinireAgent(
                name="llm_reranker",
                generation_instructions="You are an expert information retrieval system that evaluates document relevance. Rate each document on a scale of 0.0-10.0 based on how well it answers the given query.",
                model=self.config.llm_model,
                output_model=DocumentScores,  # Simplified structured output
                session_history=None,  # Disable session history for independent evaluations
                history_size=0  # No history retention
            )
            self._use_refinire = True
            self._use_structured_output = True  # Re-enabled with simplified model
            logger.info(f"Initialized LLM reranker with RefinireAgent (simplified structured output), model: {self.config.llm_model}")
        except ImportError:
            logger.warning("Refinire library not available, LLM reranking will be disabled")
            self._refinire_agent = None
            self._use_refinire = False
            self._use_structured_output = False
        except Exception as e:
            logger.error(f"Failed to initialize RefinireAgent with structured output: {e}")
            # Fallback to non-structured mode
            try:
                from refinire import RefinireAgent
                self._refinire_agent = RefinireAgent(
                    name="llm_reranker",
                    generation_instructions="You are an expert information retrieval system that evaluates document relevance.",
                    model=self.config.llm_model,
                    session_history=None,
                    history_size=0
                )
                self._use_refinire = True
                self._use_structured_output = False
                logger.warning(f"Fallback to non-structured mode: {e}")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize RefinireAgent fallback: {fallback_error}")
                self._refinire_agent = None
                self._use_refinire = False
                self._use_structured_output = False
    
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
        """Evaluate batch using numerical scoring with structured output
        
        構造化出力を使用した数値スコアリング（0-10スケール）でバッチを評価
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
        
        try:
            logger.debug(f"LLM Reranker - Processing {len(batch)} documents for query: {query}")
            logger.debug(f"Document IDs: {doc_ids}")
            
            # Call LLM using RefinireAgent
            if not self._use_refinire or not self._refinire_agent:
                raise RuntimeError("No LLM client available for reranking")
            
            if self._use_structured_output:
                # Use structured output mode
                prompt = self._create_structured_prompt(query, docs_text, doc_ids)
                logger.debug("Using RefinireAgent structured output mode")
                result = self._refinire_agent.run(prompt)
                logger.debug(f"RefinireAgent result type: {type(result)}")
                logger.debug(f"RefinireAgent result: {str(result)[:200]}...")
                
                # Handle RefinireAgent API - result is Context object with result.result attribute
                content = None
                
                # Try to get result from different possible locations
                if hasattr(result, 'result') and result.result is not None:
                    content = result.result
                    logger.debug(f"Got result.result: {type(content)}")
                elif hasattr(result, 'shared_state') and f'{self._refinire_agent.name}_result' in result.shared_state:
                    content = result.shared_state[f'{self._refinire_agent.name}_result']
                    logger.debug(f"Got from shared_state: {type(content)}")
                else:
                    logger.warning(f"No result found in Context: {type(result)}")
                    logger.debug(f"Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
                    if hasattr(result, 'shared_state'):
                        logger.debug(f"Shared state keys: {list(result.shared_state.keys())}")
                
                if content is not None:
                    # Check if we got structured output (DocumentScores object)
                    if isinstance(content, DocumentScores):
                        # Convert list format to dict format
                        scores = {item.document_id: item.score for item in content.scores}
                        logger.debug(f"Structured output scores (list format): {scores}")
                    elif isinstance(content, dict) and 'scores' in content:
                        if isinstance(content['scores'], list):
                            # Handle list format: [{"document_id": "id", "score": 8.5}, ...]
                            scores = {item['document_id']: item['score'] for item in content['scores']}
                            logger.debug(f"Structured output scores (dict with list): {scores}")
                        else:
                            # Handle dict format: {"doc_id": 8.5, ...}
                            scores = content['scores']
                            logger.debug(f"Structured output scores (dict format): {scores}")
                    else:
                        # content is a string, parse it
                        logger.debug(f"Content is string, parsing: {str(content)[:200]}...")
                        scores = self._parse_numerical_response(str(content), doc_ids)
                else:
                    # Fallback to parsing the string representation of the whole result
                    logger.warning(f"No content found, parsing string representation")
                    response = str(result)
                    scores = self._parse_numerical_response(response, doc_ids)
            else:
                # Use prompt-based mode
                prompt = self._create_numerical_prompt(query, docs_text, doc_ids)
                logger.debug("Using RefinireAgent prompt-based mode")
                result = self._refinire_agent.run(prompt)
                logger.debug(f"Non-structured result type: {type(result)}")
                
                # Handle RefinireAgent response - extract from Context object
                if hasattr(result, 'result') and result.result is not None:
                    response = str(result.result)
                    logger.debug(f"Got result.result for non-structured: {response[:100]}...")
                elif hasattr(result, 'shared_state') and f'{self._refinire_agent.name}_result' in result.shared_state:
                    response = str(result.shared_state[f'{self._refinire_agent.name}_result'])
                    logger.debug(f"Got from shared_state for non-structured: {response[:100]}...")
                else:
                    logger.warning(f"No result found in non-structured Context: {type(result)}")
                    response = str(result)
                
                scores = self._parse_numerical_response(response, doc_ids)
            
            # Validate and normalize scores
            normalized_scores = {}
            for doc_id in doc_ids:
                if doc_id in scores:
                    score = float(scores[doc_id])
                    # Clamp to valid range and normalize to [0, 1]
                    score = min(max(score, 0.0), 10.0) / 10.0
                    normalized_scores[doc_id] = score
                else:
                    logger.warning(f"Document ID {doc_id} not found in scores, using default")
                    normalized_scores[doc_id] = 0.5  # Default middle score
            
            logger.debug(f"Final normalized scores: {normalized_scores}")
            return normalized_scores
            
        except Exception as e:
            logger.error(f"LLM evaluation failed for batch: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            # Return original scores as fallback
            fallback_scores = {result.document_id: result.score for result in batch}
            logger.warning(f"Using fallback scores: {fallback_scores}")
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
        """Create prompt for numerical scoring with structured output
        
        構造化出力対応の数値スコアリング用プロンプトを作成
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
        
        # Create concrete example with actual document IDs
        example_scores = {}
        for i, doc_id in enumerate(doc_ids):
            example_scores[doc_id] = f"{5.0 + i * 0.5:.1f}"  # Generate realistic example scores
        
        example_json = "{\n    \"scores\": {\n"
        for i, (doc_id, score) in enumerate(example_scores.items()):
            comma = "," if i < len(doc_ids) - 1 else ""
            example_json += f'        "{doc_id}": {score}{comma}\n'
        example_json += "    }\n}"
        
        prompt = f"""You are an expert information retrieval system. Your task is to evaluate how relevant each document is to the given query.

Query: "{query}"

{thinking_prompt}Rate each document on a scale of 0.0-10.0 where:
- 10.0: Perfectly relevant - document directly discusses "{query}" as main topic and provides comprehensive information
- 8.0-9.0: Highly relevant - document specifically addresses "{query}" with detailed information
- 6.0-7.0: Moderately relevant - document mentions "{query}" and provides some specific information
- 3.0-5.0: Slightly relevant - document mentions concepts related to "{query}" but doesn't focus on it
- 1.0-2.0: Minimally relevant - document only tangentially mentions related concepts
- 0.0: Not relevant - document doesn't address "{query}" or related concepts

Focus on DIRECT TOPICAL RELEVANCE. A document that specifically discusses "{query}" should score much higher than a document that only mentions related business concepts.

Documents to evaluate:
"""
        
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, docs_text)):
            prompt += f"\nDocument {doc_id}:\n{doc_text}\n"
        
        prompt += f"""
CRITICAL: You must respond with ONLY valid JSON in the exact format below. Do not include any text before or after the JSON.

Required JSON format (use this exact structure):
{example_json}

Rules:
- Use decimal numbers with one decimal place (e.g., 7.5, not 7 or 7.567)
- Scores must be between 0.0 and 10.0
- Include all document IDs exactly as shown
- Do not add any explanation or text outside the JSON
- Do not use markdown code blocks (```json)

Respond with JSON only:"""
        
        return prompt
    
    def _create_structured_prompt(self, query: str, docs_text: List[str], doc_ids: List[str]) -> str:
        """Create prompt for structured output mode
        
        構造化出力モード用のプロンプトを作成
        """
        thinking_prompt = f"""
Think step by step about how relevant each document is to the query: "{query}"

Evaluation criteria (in order of importance):
1. DIRECT TOPIC RELEVANCE: Does the document directly discuss "{query}" as its main or significant topic?
2. EXPLICIT MENTION: Does the document explicitly mention the key terms from "{query}"?
3. CONTENT FOCUS: How much of the document content is specifically about "{query}" vs. related but different topics?
4. ANSWER COMPLETENESS: How well does the document answer the specific question "{query}"?

IMPORTANT: Prioritize documents that directly address the query topic over documents that only mention related concepts.

""" if self.config.use_chain_of_thought else ""
        
        prompt = f"""You are an expert information retrieval system. Your task is to evaluate how relevant each document is to the given query.

Query: "{query}"

{thinking_prompt}Rate each document on a scale of 0.0-10.0 where:
- 10.0: Perfectly relevant - document directly discusses "{query}" as main topic and provides comprehensive information
- 8.0-9.0: Highly relevant - document specifically addresses "{query}" with detailed information
- 6.0-7.0: Moderately relevant - document mentions "{query}" and provides some specific information
- 3.0-5.0: Slightly relevant - document mentions concepts related to "{query}" but doesn't focus on it
- 1.0-2.0: Minimally relevant - document only tangentially mentions related concepts
- 0.0: Not relevant - document doesn't address "{query}" or related concepts

Focus on DIRECT TOPICAL RELEVANCE. A document that specifically discusses "{query}" should score much higher than a document that only mentions related business concepts.

Documents to evaluate:
"""
        
        for i, (doc_id, doc_text) in enumerate(zip(doc_ids, docs_text)):
            prompt += f"\nDocument {doc_id}:\n{doc_text}\n"
        
        prompt += f"""
Provide relevance scores for each document in the following format:
{{
  "scores": [
    {{"document_id": "{doc_ids[0]}", "score": 8.5}},
    {{"document_id": "{doc_ids[1] if len(doc_ids) > 1 else doc_ids[0]}", "score": 6.0}}
  ]
}}

Document IDs to score: {', '.join(doc_ids)}"""
        
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
        """Parse numerical scores from LLM response using RefinireAgent pattern
        
        RefinireAgentパターンを使用してLLM応答から数値スコアを解析
        """
        try:
            # Clean response - remove markdown code block markers if present (RefinireAgent pattern)
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            logger.debug(f"Cleaned response for parsing: {clean_response[:200]}...")
            
            # Extract JSON from response if it contains other text
            if "{" in clean_response and "}" in clean_response:
                json_start = clean_response.find("{")
                json_end = clean_response.rfind("}") + 1
                json_str = clean_response[json_start:json_end]
                
                # Parse JSON
                data = json.loads(json_str)
                logger.debug(f"Parsed JSON data: {data}")
                
                if "scores" in data and isinstance(data["scores"], dict):
                    scores = {}
                    for doc_id in doc_ids:
                        if doc_id in data["scores"]:
                            score = float(data["scores"][doc_id])
                            # Clamp score to valid range
                            score = min(max(score, 0.0), 10.0)
                            scores[doc_id] = score
                        else:
                            logger.warning(f"Document ID {doc_id} not found in LLM response scores")
                            scores[doc_id] = 5.0  # Default middle score
                    
                    logger.debug(f"Extracted scores: {scores}")
                    return scores
                else:
                    logger.warning("No 'scores' field found in JSON response")
            
            # Fallback: try to extract numbers in order from response
            logger.warning("Falling back to number extraction from response")
            import re
            numbers = re.findall(r'\d+\.?\d*', clean_response)
            scores = {}
            
            for i, doc_id in enumerate(doc_ids):
                if i < len(numbers):
                    score = float(numbers[i])
                    score = min(max(score, 0.0), 10.0)  # Clamp to valid range
                    scores[doc_id] = score
                else:
                    scores[doc_id] = 5.0
            
            logger.warning(f"Fallback scores extracted: {scores}")
            return scores
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response that failed to parse: {response[:500]}...")
            # Return safe fallback scores
            return {doc_id: 5.0 for doc_id in doc_ids}
        except Exception as e:
            logger.error(f"Failed to parse numerical response: {e}")
            logger.error(f"Response type: {type(response)}, content: {response[:200]}...")
            # Return safe fallback scores
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
            "use_chain_of_thought": self.config.use_chain_of_thought,
            "fallback_on_error": self.config.fallback_on_error,
            "llm_available": self._refinire_agent is not None,
            "structured_output_enabled": getattr(self, '_use_structured_output', False)
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