"""
Refinire RAG package.
/ Refinire RAGパッケージ
"""

import importlib
import logging

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

# Get the version from pyproject.toml
# pyproject.tomlからバージョンを取得
try:
    __version__ = version("refinire-rag")
except PackageNotFoundError:
    __version__ = "unknown"

# Import core models
# コアモデルをインポート
from .models import Document, QAPair, EvaluationResult

# Import core interfaces
# コアインターフェースをインポート
from .corpusstore import CorpusStore

# Import implementations
# 実装をインポート
from .corpus_store.sqlite_corpus_store import SQLiteCorpusStore

# Import application components
# アプリケーションコンポーネントをインポート
from .application.corpus_manager_new import CorpusManager
from .application.query_engine_new import QueryEngine
from .application.quality_lab import QualityLab

# Import embedding components
# 埋め込みコンポーネントをインポート
from .embedding.tfidf_embedder import TFIDFEmbedder
from .embedding.openai_embedder import OpenAIEmbedder

# Import document processing
# ドキュメント処理をインポート
from .document_processor import DocumentProcessor

# Import storage components
# ストレージコンポーネントをインポート
from .storage.document_store import DocumentStore
from .storage.sqlite_store import SQLiteDocumentStore

# Import loader components
# ローダーコンポーネントをインポート
try:
    from .loader.document_store_loader import DocumentStoreLoader
except ImportError:
    DocumentStoreLoader = None

# Import chunking components
# チャンク処理コンポーネントをインポート
try:
    from .chunking.chunker import TokenBasedChunker
except ImportError:
    TokenBasedChunker = None

__all__ = [
    "Document",
    "QAPair",
    "EvaluationResult",
    "CorpusStore",
    "SQLiteCorpusStore",
    "CorpusManager",
    "QueryEngine",
    "QualityLab",
    "TFIDFEmbedder",
    "OpenAIEmbedder",
    "DocumentProcessor",
    "DocumentStore",
    "SQLiteDocumentStore",
]