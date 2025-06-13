"""
Simple LLM-based answer reader

A basic implementation of the Reader interface that generates
answers using an LLM with context from retrieved documents.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Type

from .base import Reader, ReaderConfig, SearchResult
try:
    from refinire import get_llm
except ImportError:
    # Mock implementation for testing
    class MockLLMClient:
        def __init__(self, model: str):
            self.model = model
        
        def complete(self, prompt: str) -> str:
            # Simple mock response
            if "RAG" in prompt:
                return "RAG（Retrieval-Augmented Generation）は、検索拡張生成技術です。外部知識ベースを活用して、より正確で根拠のある回答を生成します。"
            elif "ベクトル検索" in prompt:
                return "ベクトル検索は、文書を高次元ベクトル空間に埋め込み、意味的類似性に基づいて検索を行う技術です。"
            elif "LLM" in prompt:
                return "LLM（大規模言語モデル）は、自然言語処理において重要な役割を果たすAI技術です。"
            else:
                return "ご質問の内容について、参考文書に基づいた回答を提供いたします。"
    
    def get_llm(model: str):
        return MockLLMClient(model)

logger = logging.getLogger(__name__)


class SimpleReaderConfig(ReaderConfig):
    """Configuration for SimpleReader"""
    
    def __init__(self,
                 max_context_length: int = 2000,
                 llm_model: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 include_sources: bool = True,
                 context_separator: str = "\n\n---\n\n",
                 **kwargs):
        super().__init__(max_context_length=max_context_length,
                        llm_model=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens)
        self.include_sources = include_sources
        self.context_separator = context_separator
        
        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class SimpleReader(Reader):
    """Simple LLM-based answer reader
    
    Generates answers by combining retrieved document context
    with an LLM prompt to produce coherent, grounded responses.
    """
    
    def __init__(self, config: Optional[SimpleReaderConfig] = None):
        """Initialize SimpleReader
        
        Args:
            config: Reader configuration
        """
        super().__init__(config or SimpleReaderConfig())
        
        # Initialize LLM client
        self._llm_client = get_llm(self.config.llm_model)
        
        logger.info(f"Initialized SimpleReader with model: {self.config.llm_model}")
    
    @classmethod
    def get_config_class(cls) -> Type[SimpleReaderConfig]:
        """Get configuration class for this reader"""
        return SimpleReaderConfig
    
    def read(self, query: str, contexts: List[SearchResult]) -> str:
        """Generate answer from query and context documents
        
        Args:
            query: User query
            contexts: Relevant context documents
            
        Returns:
            Generated answer string
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Generating answer for query: '{query}' with {len(contexts)} contexts")
            
            if not contexts:
                return "申し訳ございませんが、関連する情報が見つかりませんでした。"
            
            # Build context from search results
            context_text = self._build_context(contexts)
            
            # Generate prompt
            prompt = self._build_prompt(query, context_text)
            
            # Get LLM response
            response = self._llm_client.complete(prompt)
            
            # Extract answer from response
            answer = self._extract_answer(response, contexts)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["queries_processed"] += 1
            self.processing_stats["processing_time"] += processing_time
            
            logger.debug(f"Generated answer ({len(answer)} chars) in {processing_time:.3f}s")
            return answer
            
        except Exception as e:
            self.processing_stats["errors_encountered"] += 1
            logger.error(f"Answer generation failed: {e}")
            return f"申し訳ございませんが、回答の生成中にエラーが発生しました: {str(e)}"
    
    def _build_context(self, contexts: List[SearchResult]) -> str:
        """Build context text from search results"""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(contexts):
            doc = result.document
            content = doc.content
            
            # Add source identification if enabled
            if self.config.include_sources:
                source_info = f"[ソース {i+1}]"
                if 'path' in doc.metadata:
                    source_info += f" ({doc.metadata['path']})"
                content = f"{source_info}\n{content}"
            
            # Check length limits
            content_length = len(content)
            if current_length + content_length > self.config.max_context_length:
                # Truncate if needed
                remaining_length = self.config.max_context_length - current_length
                if remaining_length > 100:  # Only include if substantial
                    content = content[:remaining_length] + "..."
                    context_parts.append(content)
                break
            
            context_parts.append(content)
            current_length += content_length + len(self.config.context_separator)
        
        return self.config.context_separator.join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""以下の文書を参考に、ユーザーの質問に正確かつ有用な回答を生成してください。

質問: {query}

参考文書:
{context}

回答要件:
- 参考文書の情報のみに基づいて回答してください
- 文書に記載されていない情報は推測しないでください
- 回答は簡潔で分かりやすくしてください
- 可能な限り具体的な例や詳細を含めてください
- 情報が不十分な場合は、その旨を明記してください

回答:"""
        
        return prompt
    
    def _extract_answer(self, llm_response: str, contexts: List[SearchResult]) -> str:
        """Extract and post-process answer from LLM response"""
        # Simple extraction - just use the response as-is
        answer = llm_response.strip()
        
        # Basic post-processing
        if not answer:
            return "申し訳ございませんが、適切な回答を生成できませんでした。"
        
        # Add source references if configured and not already present
        if self.config.include_sources and len(contexts) > 0:
            if "[ソース" not in answer and "参考" not in answer:
                source_count = len(contexts)
                answer += f"\n\n（{source_count}件の文書を参考にしました）"
        
        return answer
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics with reader-specific metrics"""
        stats = super().get_processing_stats()
        
        # Add reader-specific stats
        stats.update({
            "reader_type": "SimpleReader",
            "llm_model": self.config.llm_model,
            "max_context_length": self.config.max_context_length,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "include_sources": self.config.include_sources
        })
        
        return stats