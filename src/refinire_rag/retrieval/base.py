"""
Base classes for retrieval components

Defines the core interfaces for QueryComponent and its implementations:
Retriever, Reranker, and Reader components used in QueryEngine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Type, Iterable, Iterator
import numpy as np

from ..models.document import Document
from ..document_processor import DocumentProcessor
from ..utils.model_config import get_default_llm_model


@dataclass
class SearchResult:
    """Search result from retrieval"""
    document_id: str
    document: Document
    score: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResult:
    """Final query result with answer"""
    query: str
    normalized_query: Optional[str] = None
    answer: str = ""
    sources: List[SearchResult] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}


class QueryComponentConfig:
    """Base configuration for query components"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class RetrieverConfig(QueryComponentConfig):
    """Configuration for retrievers"""
    top_k: int = 10
    similarity_threshold: float = 0.0
    enable_filtering: bool = True


@dataclass  
class RerankerConfig(QueryComponentConfig):
    """Configuration for rerankers"""
    top_k: int = 5
    rerank_model: str = "cross-encoder"
    score_threshold: float = 0.0


@dataclass
class AnswerSynthesizerConfig(QueryComponentConfig):
    """Configuration for answer synthesizers
    
    LLMを使用した回答合成の設定
    
    Args:
        max_context_length: Maximum length of context to use
                          使用するコンテキストの最大長
        llm_model: LLM model to use for answer generation
                  回答生成に使用するLLMモデル
        temperature: Temperature for answer generation
                    回答生成の温度パラメータ
        max_tokens: Maximum tokens to generate
                   生成する最大トークン数
    """
    max_context_length: int = 2000
    llm_model: str = None  # Will be set to default in __post_init__
    temperature: float = 0.1
    max_tokens: int = 500
    
    def __post_init__(self):
        """Initialize default values"""
        # Set default LLM model from environment variables if not specified
        if self.llm_model is None:
            self.llm_model = get_default_llm_model()


class QueryComponent(ABC):
    """Base class for query processing components
    
    Similar to DocumentProcessor but for query processing workflow.
    Provides unified interface for Retriever, Reranker, and Reader.
    """
    
    def __init__(self, config: Optional[QueryComponentConfig] = None):
        self.config = config or QueryComponentConfig()
        
        # Processing statistics
        self.processing_stats = {
            "queries_processed": 0,
            "processing_time": 0.0,
            "errors_encountered": 0
        }
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[QueryComponentConfig]:
        """Get the configuration class for this component"""
        pass
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()


class Retriever(QueryComponent):
    """Base class for document retrievers
    
    Unified interface for all document retrieval implementations including
    vector search, keyword search, and hybrid approaches.
    
    すべての文書検索実装（ベクトル検索、キーワード検索、ハイブリッドアプローチ）
    の統一インターフェース。
    """
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[RetrieverConfig]:
        """Get the configuration class for this retriever"""
        pass
    
    @abstractmethod
    def retrieve(self, 
                 query: str, 
                 limit: Optional[int] = None,
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Retrieve relevant documents for query
        
        Args:
            query: Search query text
                  検索クエリテキスト
            limit: Maximum number of results to return (uses config.top_k if None)
                  返す結果の最大数（Noneの場合はconfig.top_kを使用）
            metadata_filter: Optional metadata filters for constraining search
                           検索を制約するオプションのメタデータフィルタ
                           Example: {"department": "AI", "year": 2024, "status": "active"}
            
        Returns:
            List[SearchResult]: List of search results with scores, sorted by relevance
                               関連度でソートされたスコア付き検索結果のリスト
        """
        pass


class Indexer:
    """Base class for document indexing capabilities
    
    Provides document indexing and management functionality that can be
    used by Retriever implementations to create searchable stores.
    
    検索可能なストアを作成するためにRetriever実装で使用できる
    文書インデックスと管理機能を提供します。
    """
    
    @abstractmethod
    def index_document(self, document: Document) -> None:
        """Index a single document for search
        
        Args:
            document: Document to index
                     インデックスする文書
        """
        pass
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """Index multiple documents efficiently
        
        Args:
            documents: List of documents to index
                      インデックスする文書のリスト
        """
        pass
    
    @abstractmethod
    def remove_document(self, document_id: str) -> bool:
        """Remove document from index
        
        Args:
            document_id: ID of document to remove
                        削除する文書のID
                        
        Returns:
            bool: True if document was found and removed, False otherwise
                 文書が見つかって削除された場合True、そうでなければFalse
        """
        pass
    
    @abstractmethod
    def update_document(self, document: Document) -> bool:
        """Update an existing document in the index
        
        Args:
            document: Updated document (must have existing ID)
                     更新する文書（既存のIDを持つ必要がある）
                     
        Returns:
            bool: True if document was found and updated, False otherwise
                 文書が見つかって更新された場合True、そうでなければFalse
        """
        pass
    
    @abstractmethod
    def clear_index(self) -> None:
        """Remove all documents from the index
        
        インデックスからすべての文書を削除
        """
        pass
    
    @abstractmethod
    def get_document_count(self) -> int:
        """Get the number of documents in the index
        
        Returns:
            int: Number of indexed documents
                インデックスされた文書数
        """
        pass
    
    def index_document_batch(self, documents: List[Document], batch_size: int = 100) -> None:
        """Index documents in batches for better performance
        
        Args:
            documents: Documents to index
                      インデックスする文書
            batch_size: Number of documents per batch
                       バッチあたりの文書数
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.index_documents(batch)


class Reranker(QueryComponent):
    """Base class for result rerankers"""
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[RerankerConfig]:
        """Get the configuration class for this reranker"""
        pass
    
    @abstractmethod
    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on relevance
        
        Args:
            query: Original query
            results: Initial search results
            
        Returns:
            Reranked search results
        """
        pass


class AnswerSynthesizer(QueryComponent):
    """Base class for answer synthesizers
    
    Synthesizes answers from user queries and relevant context documents
    using LLM-based generation.
    
    ユーザークエリと関連文書からLLMベースの生成を使用して
    回答を合成する基底クラス。
    """
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[AnswerSynthesizerConfig]:
        """Get the configuration class for this synthesizer"""
        pass
    
    @abstractmethod
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
        pass


class KeywordSearch(DocumentProcessor, Indexer, Retriever):
    """
    Base class for keyword-based document search with DocumentProcessor integration.
    DocumentProcessor統合を備えたキーワードベースの文書検索の基底クラス
    
    This class combines:
    - Indexer: Document indexing capabilities
    - Retriever: Document search capabilities 
    - DocumentProcessor: Pipeline integration
    
    このクラスは以下を組み合わせます：
    - Indexer: 文書インデックス機能
    - Retriever: 文書検索機能
    - DocumentProcessor: パイプライン統合
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize KeywordSearch with DocumentProcessor integration"""
        DocumentProcessor.__init__(self, config=config or {})
        
        # Add keyword search specific stats
        self.processing_stats.update({
            "documents_indexed": 0,
            "searches_performed": 0,
            "index_size": 0
        })
    
    @classmethod
    @abstractmethod  
    def get_config_class(cls) -> Type[Dict]:
        """Get the configuration class for this keyword search"""
        pass
    
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Process documents by indexing them and yielding them unchanged
        
        Args:
            documents: Input documents to index
            config: Optional configuration override
            
        Yields:
            Documents (unchanged, after indexing)
        """
        for document in documents:
            try:
                # Index the document
                self.index_document(document)
                self.processing_stats["documents_indexed"] += 1
                
                # Yield document unchanged (DocumentProcessor pattern)
                yield document
                
            except Exception as e:
                self.processing_stats["processing_errors"] += 1
                # Log error but continue processing
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error indexing document {document.id}: {e}")
                
                # Still yield the document
                yield document
    
    @abstractmethod
    def add_document(self, document: Document) -> None:
        """
        Add a document to the store.
        ストアに文書を追加する
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for documents using keyword matching.
        キーワードマッチングを使用して文書を検索する
        """
        pass