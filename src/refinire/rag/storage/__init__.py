"""
Storage Components for refinire-rag

Provides document storage and vector storage interfaces and implementations.
DocumentStore handles raw documents and processing stages.
VectorStore handles embeddings and similarity search.
"""

from .document_store import DocumentStore, SearchResult, StorageStats
from .sqlite_store import SQLiteDocumentStore
from .vector_store import VectorStore, VectorSearchResult, VectorEntry, VectorStoreStats
from .in_memory_vector_store import InMemoryVectorStore
from .pickle_vector_store import PickleVectorStore

__all__ = [
    # Document Storage
    "DocumentStore", 
    "SearchResult", 
    "StorageStats",
    "SQLiteDocumentStore",
    
    # Vector Storage
    "VectorStore",
    "VectorSearchResult", 
    "VectorEntry",
    "VectorStoreStats",
    "InMemoryVectorStore",
    "PickleVectorStore"
]