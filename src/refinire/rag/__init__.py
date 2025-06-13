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

__all__ = [
    "Document",
    "QAPair",
    "EvaluationResult",
    "CorpusStore",
    "SQLiteCorpusStore",
]