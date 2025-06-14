"""Simple LLM-based answer synthesizer

A basic implementation of the AnswerSynthesizer interface that generates
answers using OpenAI's GPT models.
"""

import time
from typing import List, Optional, Dict, Any, Type
import logging
from openai import OpenAI

from refinire_rag.use_cases.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.retrieval import SimpleRetrieverConfig, SimpleRerankerConfig, SimpleReaderConfig
from refinire_rag.models.document import Document
from refinire_rag.embedding import TFIDFEmbedder
from refinire_rag.storage.vector_store import VectorEntry

from refinire_rag.embedding import TFIDFEmbeddingConfig

logger = logging.getLogger(__name__)

class SimpleAnswerSynthesizerConfig(AnswerSynthesizerConfig):
    """Configuration for SimpleAnswerSynthesizer
    
    SimpleAnswerSynthesizerの設定
    
    Args:
        max_context_length: Maximum length of context to use
                          使用するコンテキストの最大長
        llm_model: LLM model to use for answer generation
                  回答生成に使用するLLMモデル
        temperature: Temperature for answer generation
                    回答生成の温度パラメータ
        max_tokens: Maximum tokens to generate
                   生成する最大トークン数
        openai_api_key: OpenAI API key
                       OpenAI APIキー
        openai_organization: OpenAI organization ID
                            OpenAI組織ID
    """
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None

class SimpleAnswerSynthesizer(AnswerSynthesizer):
    """Simple LLM-based answer synthesizer
    
    OpenAIのGPTモデルを使用して回答を生成する
    基本的なAnswerSynthesizerの実装。
    """
    
    def __init__(self, config: Optional[SimpleAnswerSynthesizerConfig] = None):
        """Initialize SimpleAnswerSynthesizer
        
        Args:
            config: Synthesizer configuration
                   合成器の設定
        """
        super().__init__(config or SimpleAnswerSynthesizerConfig())
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.config.openai_api_key,
            organization=self.config.openai_organization
        )
        
        logger.info(f"Initialized SimpleAnswerSynthesizer with model: {self.config.llm_model}")
    
    @classmethod
    def get_config_class(cls) -> Type[SimpleAnswerSynthesizerConfig]:
        """Get configuration class for this synthesizer"""
        return SimpleAnswerSynthesizerConfig
    
    def synthesize(self, query: str, contexts: List[SearchResult]) -> str:
        """Synthesize answer from query and context documents
        
        Args:
            query: User query
                  ユーザークエリ
            contexts: Relevant context documents
                     関連文書
                     
        Returns:
            str: Synthesized answer
                 合成された回答
        """
        start_time = time.time()
        
        try:
            # Prepare context
            context_text = self._prepare_context(contexts)
            
            # Prepare prompt
            prompt = self._prepare_prompt(query, context_text)
            
            # Generate answer
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Update statistics
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += time.time() - start_time
            
            return answer
            
        except Exception as e:
            logger.error(f"Error synthesizing answer: {str(e)}")
            self.processing_stats["errors_encountered"] += 1
            raise
    
    def _prepare_context(self, contexts: List[SearchResult]) -> str:
        """Prepare context text from search results
        
        Args:
            contexts: Search results to use as context
                     コンテキストとして使用する検索結果
                     
        Returns:
            str: Formatted context text
                 フォーマットされたコンテキストテキスト
        """
        context_texts = []
        total_length = 0
        
        for result in contexts:
            doc_text = result.document.text
            if total_length + len(doc_text) > self.config.max_context_length:
                break
            context_texts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n".join(context_texts)
    
    def _prepare_prompt(self, query: str, context: str) -> str:
        """Prepare prompt for LLM
        
        Args:
            query: User query
                  ユーザークエリ
            context: Context text
                    コンテキストテキスト
                    
        Returns:
            str: Formatted prompt
                 フォーマットされたプロンプト
        """
        return f"""Based on the following context, please answer the question.
If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {query}

Answer:"""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with synthesizer-specific metrics
        
        Returns:
            Dict[str, Any]: Processing statistics
                           処理統計
        """
        stats = super().get_processing_stats()
        
        # Add synthesizer-specific stats
        stats.update({
            "synthesizer_type": "SimpleAnswerSynthesizer",
            "model": self.config.llm_model
        })
        
        return stats