"""Simple LLM-based answer synthesizer and reader

A basic implementation of the AnswerSynthesizer interface that generates
answers using OpenAI's GPT models.
"""

import os
import time
from typing import List, Optional, Dict, Any, Type
import logging

from .base import AnswerSynthesizer, AnswerSynthesizerConfig, SearchResult
from ..utils.model_config import get_default_llm_model
from ..config import RefinireRAGConfig

try:
    from refinire import RefinireAgent
except ImportError:
    RefinireAgent = None

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
        generation_instructions: Instructions for the LLM on how to generate answers
                               LLMに対する回答生成方法の指示
        system_prompt: System prompt for OpenAI chat completions (when not using Refinire)
                      OpenAIチャット補完用のシステムプロンプト（Refinireを使用しない場合）
        openai_api_key: OpenAI API key
                       OpenAI APIキー
        openai_organization: OpenAI organization ID
                            OpenAI組織ID
    """
    
    def __init__(self, 
                 max_context_length: int = 2000,
                 llm_model: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 generation_instructions: str = "You are a helpful assistant that answers questions based on the provided context.",
                 system_prompt: str = "You are a helpful assistant that answers questions based on the provided context.",
                 openai_api_key: Optional[str] = None,
                 openai_organization: Optional[str] = None,
                 **kwargs):
        
        # Initialize parent class
        super().__init__(max_context_length=max_context_length,
                        llm_model=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs)
        
        # Set additional attributes
        self.generation_instructions = generation_instructions
        self.system_prompt = system_prompt
        self.openai_api_key = openai_api_key
        self.openai_organization = openai_organization
        # Additional initialization can be added here if needed
    
    @classmethod
    def from_env(cls) -> "SimpleAnswerSynthesizerConfig":
        """Create configuration from environment variables
        
        Creates a SimpleAnswerSynthesizerConfig instance from environment variables.
        環境変数からSimpleAnswerSynthesizerConfigインスタンスを作成します。
        
        Returns:
            SimpleAnswerSynthesizerConfig instance with values from environment
        """
        config = RefinireRAGConfig()
        
        # Get configuration values from environment
        max_context_length = int(os.getenv("REFINIRE_RAG_SYNTHESIZER_MAX_CONTEXT_LENGTH", "2000"))
        llm_model = get_default_llm_model()  # Uses env vars hierarchy
        temperature = float(os.getenv("REFINIRE_RAG_SYNTHESIZER_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("REFINIRE_RAG_SYNTHESIZER_MAX_TOKENS", "500"))
        
        generation_instructions = os.getenv(
            "REFINIRE_RAG_SYNTHESIZER_GENERATION_INSTRUCTIONS",
            "You are a helpful assistant that answers questions based on the provided context."
        )
        system_prompt = os.getenv(
            "REFINIRE_RAG_SYNTHESIZER_SYSTEM_PROMPT", 
            "You are a helpful assistant that answers questions based on the provided context."
        )
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_organization = os.getenv("OPENAI_ORGANIZATION")
        
        return cls(
            max_context_length=max_context_length,
            llm_model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            generation_instructions=generation_instructions,
            system_prompt=system_prompt,
            openai_api_key=openai_api_key,
            openai_organization=openai_organization
        )


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
        # Create config from environment if not provided
        if config is None:
            config = SimpleAnswerSynthesizerConfig.from_env()
        else:
            config = config or SimpleAnswerSynthesizerConfig()
            
        super().__init__(config)
        
        # Initialize Refinire Agent (preferred) or fallback to OpenAI
        if RefinireAgent is not None:
            try:
                self._refinire_agent = RefinireAgent(
                    name="answer_synthesizer",
                    generation_instructions=self.config.generation_instructions,
                    model=self.config.llm_model,
                    session_history=None,  # Disable session history for independent answers
                    history_size=0  # No history retention
                )
                self._use_refinire = True
                logger.info(f"Initialized SimpleAnswerSynthesizer with RefinireAgent, model: {self.config.llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize RefinireAgent: {e}. Falling back to OpenAI.")
                self._init_openai_client()
        else:
            logger.warning("Refinire not available. Using OpenAI client.")
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client as fallback"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config.openai_api_key,
                organization=self.config.openai_organization
            )
            self._use_refinire = False
            logger.info(f"Initialized SimpleAnswerSynthesizer with OpenAI client, model: {self.config.llm_model}")
        except ImportError:
            logger.error("Neither Refinire nor OpenAI is available. SimpleAnswerSynthesizer will not work.")
            self._use_refinire = None
    
    
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
            
            # Generate answer using Refinire or OpenAI
            if self._use_refinire and hasattr(self, '_refinire_agent'):
                result = self._refinire_agent.run(prompt)
                # Extract answer from Context object (refinire 0.2.8 API)
                if hasattr(result, 'result') and result.result is not None:
                    answer = str(result.result)
                elif hasattr(result, 'shared_state') and f'{self._refinire_agent.name}_result' in result.shared_state:
                    answer = str(result.shared_state[f'{self._refinire_agent.name}_result'])
                elif hasattr(result, 'content'):
                    answer = str(result.content)  # Fallback for old API
                else:
                    answer = str(result)
            elif self._use_refinire is False and hasattr(self, 'client'):
                response = self.client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                answer = response.choices[0].message.content
            else:
                raise RuntimeError("No LLM client available (neither Refinire nor OpenAI)")
            
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
            doc_text = result.document.content
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
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary"""
        config_dict = {
            'llm_model': self.config.llm_model,
            'max_context_length': self.config.max_context_length,
            'max_answer_length': self.config.max_answer_length,
            'include_source_info': self.config.include_source_info,
            'temperature': self.config.temperature
        }
        
        # Add any additional attributes from the config
        for attr_name, attr_value in self.config.__dict__.items():
            if attr_name not in config_dict:
                config_dict[attr_name] = attr_value
                
        return config_dict


# Aliases for backward compatibility and simpler usage
SimpleReaderConfig = SimpleAnswerSynthesizerConfig
SimpleReader = SimpleAnswerSynthesizer