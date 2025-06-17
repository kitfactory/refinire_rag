"""
Configuration management for refinire-rag using environment variables.
Provides centralized access to all configuration settings with defaults.
"""
"""
refinire-ragの環境変数を使った設定管理。
デフォルト値付きのすべての設定への一元アクセスを提供します。
"""

import os
from typing import Optional


class RefinireRAGConfig:
    """
    Central configuration class for refinire-rag.
    Manages all environment variables with appropriate defaults.
    """
    """
    refinire-ragの中央設定クラス。
    適切なデフォルト値ですべての環境変数を管理します。
    """
    
    # Critical Variables (必須変数)
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API authentication key"""
        """OpenAI API認証キー"""
        return os.getenv("OPENAI_API_KEY")
    
    # Important Variables (重要変数)
    @property
    def llm_model(self) -> str:
        """Primary LLM model for RAG operations"""
        """RAG操作用のメインLLMモデル"""
        return os.getenv("REFINIRE_RAG_LLM_MODEL", "gpt-4o-mini")
    
    @property
    def data_dir(self) -> str:
        """Base data directory for all storage"""
        """すべてのストレージ用ベースデータディレクトリ"""
        return os.getenv("REFINIRE_RAG_DATA_DIR", "./data")
    
    @property
    def corpus_store(self) -> str:
        """Default corpus store type"""
        """デフォルトコーパスストアタイプ"""
        return os.getenv("REFINIRE_RAG_CORPUS_STORE", "sqlite")
    
    @property
    def retriever_top_k(self) -> int:
        """Top-K results for retriever"""
        """リトリーバーのTop-K結果数"""
        return int(os.getenv("REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K", "10"))
    
    @property
    def log_level(self) -> str:
        """Logging level"""
        """ログレベル"""
        return os.getenv("REFINIRE_RAG_LOG_LEVEL", "INFO")
    
    # Optional Variables (オプション変数)
    @property
    def fallback_llm_model(self) -> str:
        """Fallback LLM model"""
        """フォールバックLLMモデル"""
        return os.getenv("REFINIRE_DEFAULT_LLM_MODEL", "gpt-4o-mini")
    
    @property
    def refinire_dir(self) -> str:
        """Base directory for Refinire files"""
        """Refinireファイル用ベースディレクトリ"""
        return os.getenv("REFINIRE_DIR", "./refinire")
    
    @property
    def enable_telemetry(self) -> bool:
        """Enable OpenTelemetry tracing"""
        """OpenTelemetryトレーシングを有効化"""
        return os.getenv("REFINIRE_RAG_ENABLE_TELEMETRY", "true").lower() in ("true", "1", "yes")
    
    # Embedding Configuration (埋め込み設定)
    @property
    def openai_embedding_model(self) -> str:
        """OpenAI embedding model name"""
        """OpenAI埋め込みモデル名"""
        return os.getenv("REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    
    @property
    def openai_embedding_api_key(self) -> Optional[str]:
        """OpenAI API key for embeddings (falls back to main API key)"""
        """埋め込み用OpenAI APIキー（メインAPIキーにフォールバック）"""
        return os.getenv("REFINIRE_RAG_OPENAI_EMBEDDING_API_KEY") or self.openai_api_key
    
    @property
    def embedding_dimension(self) -> int:
        """Embedding dimension"""
        """埋め込み次元数"""
        return int(os.getenv("REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION", "1536"))
    
    @property
    def embedding_batch_size(self) -> int:
        """Batch size for embedding requests"""
        """埋め込みリクエストのバッチサイズ"""
        return int(os.getenv("REFINIRE_RAG_OPENAI_EMBEDDING_BATCH_SIZE", "100"))
    
    # Query Engine Configuration (クエリエンジン設定)
    @property
    def enable_query_normalization(self) -> bool:
        """Enable query normalization"""
        """クエリ正規化を有効化"""
        return os.getenv("REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION", "true").lower() in ("true", "1", "yes")
    
    @property
    def total_top_k(self) -> int:
        """Total top-K results"""
        """総Top-K結果数"""
        return int(os.getenv("REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K", "20"))
    
    @property
    def reranker_top_k(self) -> int:
        """Top-K results for reranker"""
        """リランカーのTop-K結果数"""
        return int(os.getenv("REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K", "5"))
    
    @property
    def enable_caching(self) -> bool:
        """Enable result caching"""
        """結果キャッシュを有効化"""
        return os.getenv("REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING", "true").lower() in ("true", "1", "yes")
    
    # Processing Configuration (処理設定)
    @property
    def corpus_manager_batch_size(self) -> int:
        """Processing batch size"""
        """処理バッチサイズ"""
        return int(os.getenv("REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE", "100"))
    
    @property
    def enable_parallel_processing(self) -> bool:
        """Enable parallel processing"""
        """並列処理を有効化"""
        return os.getenv("REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING", "false").lower() in ("true", "1", "yes")
    
    @property
    def fail_on_error(self) -> bool:
        """Fail on processing error"""
        """処理エラー時に失敗"""
        return os.getenv("REFINIRE_RAG_CORPUS_MANAGER_FAIL_ON_ERROR", "false").lower() in ("true", "1", "yes")
    
    # Evaluation Configuration (評価設定)
    @property
    def qa_generation_model(self) -> str:
        """Model for QA generation"""
        """QA生成用モデル"""
        return os.getenv("REFINIRE_RAG_QUALITY_LAB_QA_GENERATION_MODEL", "gpt-4o-mini")
    
    @property
    def evaluation_timeout(self) -> float:
        """Evaluation timeout in seconds"""
        """評価タイムアウト（秒）"""
        return float(os.getenv("REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT", "30.0"))
    
    @property
    def similarity_threshold(self) -> float:
        """Similarity threshold for evaluation"""
        """評価用類似性閾値"""
        return float(os.getenv("REFINIRE_RAG_QUALITY_LAB_SIMILARITY_THRESHOLD", "0.7"))
    
    # File Path Configuration (ファイルパス設定)
    @property
    def dictionary_file_path(self) -> str:
        """Dictionary file path"""
        """辞書ファイルパス"""
        return os.getenv("REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH", "./data/domain_dictionary.md")
    
    @property
    def graph_file_path(self) -> str:
        """Knowledge graph file path"""
        """知識グラフファイルパス"""
        return os.getenv("REFINIRE_RAG_GRAPH_BUILDER_GRAPH_FILE_PATH", "./data/domain_knowledge_graph.md")
    
    @property
    def test_cases_file_path(self) -> str:
        """Test cases file path"""
        """テストケースファイルパス"""
        return os.getenv("REFINIRE_RAG_TEST_SUITE_TEST_CASES_FILE", "./data/test_cases.json")
    
    # Validation Methods (検証メソッド)
    def validate_critical_config(self) -> bool:
        """
        Validates that all critical configuration is present.
        Returns True if valid, False otherwise.
        """
        """
        すべての重要な設定が存在することを検証します。
        有効な場合はTrue、そうでなければFalseを返します。
        """
        if not self.openai_api_key:
            return False
        return True
    
    def get_missing_critical_vars(self) -> list[str]:
        """
        Returns a list of missing critical environment variables.
        """
        """
        不足している重要な環境変数のリストを返します。
        """
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        return missing
    
    def get_config_summary(self) -> dict:
        """
        Returns a summary of current configuration settings.
        Excludes sensitive information like API keys.
        """
        """
        現在の設定のサマリーを返します。
        APIキーなどの機密情報は除外します。
        """
        return {
            "llm_model": self.llm_model,
            "data_dir": self.data_dir,
            "corpus_store": self.corpus_store,
            "retriever_top_k": self.retriever_top_k,
            "log_level": self.log_level,
            "enable_telemetry": self.enable_telemetry,
            "embedding_model": self.openai_embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "has_openai_api_key": bool(self.openai_api_key),
        }


# Global configuration instance (グローバル設定インスタンス)
config = RefinireRAGConfig()