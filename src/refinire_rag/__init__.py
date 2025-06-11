"""
refinire-rag: RAG (Retrieval-Augmented Generation) functionality for Refinire

This package provides modular RAG capabilities with use cases implemented as
Refinire Step subclasses and single-responsibility backend modules.
"""

__version__ = "0.1.0"

# Core models
from .models.document import Document
from .models.config import LoadingConfig, LoadingResult

# Loaders
from .loaders.base import Loader, MetadataGenerator, PathBasedMetadataGenerator
from .loaders.universal import UniversalLoader
from .loaders.specialized import (
    SpecializedLoader,
    TextLoader,
    MarkdownLoader,
    PDFLoader,
    HTMLLoader,
    CSVLoader,
    JSONLoader
)
from .loaders.document_store_loader import DocumentStoreLoader

# Storage
from .storage import (
    DocumentStore, SearchResult, StorageStats, SQLiteDocumentStore,
    VectorStore, VectorSearchResult, VectorEntry, VectorStoreStats,
    InMemoryVectorStore, PickleVectorStore
)

# Processing
from .processing import DocumentProcessor, DocumentPipeline, DocumentProcessorConfig

# Chunking
from .chunking import Chunker, ChunkingConfig, TokenBasedChunker, SentenceAwareChunker

# Embedding
from .embedding import (
    Embedder, 
    EmbeddingConfig, 
    EmbeddingResult,
    OpenAIEmbedder,
    OpenAIEmbeddingConfig,
    TFIDFEmbedder,
    TFIDFEmbeddingConfig
)

# Use Cases
from .use_cases import CorpusManager, CorpusManagerConfig

# Exceptions
from .exceptions import (
    RefinireRAGError,
    LoaderError,
    EmbeddingError,
    StorageError
)

__all__ = [
    # Version
    "__version__",
    
    # Core models
    "Document",
    "LoadingConfig", 
    "LoadingResult",
    
    # Loaders
    "Loader",
    "MetadataGenerator",
    "PathBasedMetadataGenerator",
    "UniversalLoader",
    "SpecializedLoader",
    "TextLoader",
    "MarkdownLoader", 
    "PDFLoader",
    "HTMLLoader",
    "CSVLoader",
    "JSONLoader",
    "DocumentStoreLoader",
    
    # Storage
    "DocumentStore",
    "SearchResult",
    "StorageStats", 
    "SQLiteDocumentStore",
    "VectorStore",
    "VectorSearchResult",
    "VectorEntry",
    "VectorStoreStats",
    "InMemoryVectorStore",
    "PickleVectorStore",
    
    # Processing
    "DocumentProcessor",
    "DocumentPipeline", 
    "DocumentProcessorConfig",
    
    # Chunking
    "Chunker",
    "ChunkingConfig",
    "TokenBasedChunker",
    "SentenceAwareChunker",
    
    # Embedding
    "Embedder",
    "EmbeddingConfig", 
    "EmbeddingResult",
    "OpenAIEmbedder",
    "OpenAIEmbeddingConfig",
    "TFIDFEmbedder",
    "TFIDFEmbeddingConfig",
    
    # Use Cases
    "CorpusManager",
    "CorpusManagerConfig",
    
    # Exceptions
    "RefinireRAGError",
    "LoaderError",
    "EmbeddingError",
    "StorageError",
]