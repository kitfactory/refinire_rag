"""
Exception classes for refinire-rag
refinire-rag用の例外クラス
"""


class RefinireRAGError(Exception):
    """
    Base exception for refinire-rag
    refinire-ragの基底例外
    
    All refinire-rag specific exceptions should inherit from this class
    すべてのrefinire-rag固有の例外はこのクラスを継承する必要があります
    """
    pass


# === Core Processing Errors ===
# コア処理エラー

class ProcessingError(RefinireRAGError):
    """
    Error in document processing pipeline
    文書処理パイプラインでのエラー
    """
    pass


class LoaderError(RefinireRAGError):
    """
    Error in document loading operations
    文書ロード操作でのエラー
    """
    pass


class SplitterError(RefinireRAGError):
    """
    Error in document splitting/chunking operations
    文書分割・チャンク化操作でのエラー
    """
    pass


class EmbeddingError(RefinireRAGError):
    """
    Error in embedding generation
    埋め込み生成でのエラー
    """
    pass


class MetadataError(RefinireRAGError):
    """
    Error in metadata generation or processing
    メタデータ生成・処理でのエラー
    """
    pass


# === Storage and Retrieval Errors ===
# ストレージ・検索エラー

class StorageError(RefinireRAGError):
    """
    Error in storage operations (vector stores, document stores)
    ストレージ操作でのエラー（ベクトルストア、文書ストア）
    """
    pass


class DocumentStoreError(StorageError):
    """
    Error in document store operations
    文書ストア操作でのエラー
    """
    pass


class VectorStoreError(StorageError):
    """
    Error in vector store operations
    ベクトルストア操作でのエラー
    """
    pass


class RetrievalError(RefinireRAGError):
    """
    Error in document retrieval operations
    文書検索操作でのエラー
    """
    pass


# === Configuration and Validation Errors ===
# 設定・検証エラー

class ConfigurationError(RefinireRAGError):
    """
    Error in configuration or setup
    設定・セットアップでのエラー
    """
    pass


class ValidationError(RefinireRAGError):
    """
    Error in data validation
    データ検証でのエラー
    """
    pass


class FilterError(RefinireRAGError):
    """
    Error in filtering operations
    フィルタリング操作でのエラー
    """
    pass


# === Use Case and Integration Errors ===
# ユースケース・統合エラー

class CorpusManagerError(RefinireRAGError):
    """
    Error in corpus management operations
    コーパス管理操作でのエラー
    """
    pass


class QueryEngineError(RefinireRAGError):
    """
    Error in query engine operations
    クエリエンジン操作でのエラー
    """
    pass


class EvaluationError(RefinireRAGError):
    """
    Error in evaluation operations
    評価操作でのエラー
    """
    pass


# === External Integration Errors ===
# 外部統合エラー

class LLMError(RefinireRAGError):
    """
    Error in LLM integration or communication
    LLM統合・通信でのエラー
    """
    pass


class PluginError(RefinireRAGError):
    """
    Error in plugin operations
    プラグイン操作でのエラー
    """
    pass


# === Specific Operation Errors ===
# 特定操作エラー

class FileError(LoaderError):
    """
    Error in file operations (reading, parsing, etc.)
    ファイル操作でのエラー（読み込み、パースなど）
    """
    pass


class NetworkError(RefinireRAGError):
    """
    Error in network operations
    ネットワーク操作でのエラー
    """
    pass


class SerializationError(RefinireRAGError):
    """
    Error in serialization/deserialization operations
    シリアライゼーション・デシリアライゼーション操作でのエラー
    """
    pass


class PermissionError(RefinireRAGError):
    """
    Error due to insufficient permissions
    権限不足によるエラー
    """
    pass


# === Utility Functions ===
# ユーティリティ関数

def wrap_exception(exception: Exception, message: str = None) -> RefinireRAGError:
    """
    Wrap a generic exception in a RefinireRAGError
    汎用例外をRefinireRAGErrorでラップ
    
    Args:
        exception: Original exception to wrap
        message: Optional custom message
        exception: ラップする元の例外
        message: オプションのカスタムメッセージ
        
    Returns:
        RefinireRAGError wrapping the original exception
        元の例外をラップするRefinireRAGError
    """
    if isinstance(exception, RefinireRAGError):
        return exception
    
    if message:
        wrapped_message = f"{message}: {str(exception)}"
    else:
        wrapped_message = str(exception)
    
    # Map common exceptions to specific RefinireRAG exceptions
    # 一般的な例外を特定のRefinireRAG例外にマッピング
    if isinstance(exception, (FileNotFoundError, IOError, OSError)):
        return FileError(wrapped_message)
    elif isinstance(exception, (ValueError, TypeError)):
        return ValidationError(wrapped_message)
    elif isinstance(exception, PermissionError):
        return PermissionError(wrapped_message)
    else:
        return ProcessingError(wrapped_message)