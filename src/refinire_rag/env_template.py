"""
Environment variable templates for refinire-rag using oneenv.
Provides structured configuration with importance levels and grouping.
"""
"""
oneenvを使ったrefinire-ragの環境変数テンプレート。
重要度レベルとグルーピングによる構造化された設定を提供します。
"""

from oneenv.models import EnvVarConfig, EnvTemplate


def refinire_rag_env_template() -> EnvTemplate:
    """
    Defines the environment variable template for refinire-rag.
    Groups variables by importance and functionality.
    """
    """
    refinire-ragの環境変数テンプレートを定義します。
    重要度と機能ごとに変数をグループ化します。
    """
    
    variables = {
        # Critical Variables (必須変数)
        "OPENAI_API_KEY": EnvVarConfig(
            description="OpenAI API authentication key",
            required=True,
            var_type="str",
            group="Authentication",
            importance="critical",
            example="sk-proj-...",
        ),
        
        # Important Variables (重要変数)
        "REFINIRE_RAG_LLM_MODEL": EnvVarConfig(
            description="Primary LLM model for RAG operations",
            required=False,
            var_type="str",
            default="gpt-4o-mini",
            group="Core Configuration",
            importance="important",
            example="gpt-4o-mini",
        ),
        
        "REFINIRE_RAG_DATA_DIR": EnvVarConfig(
            description="Base data directory for all storage",
            required=False,
            var_type="str",
            default="./data",
            group="Core Configuration",
            importance="important",
            example="./data",
        ),
        
        "REFINIRE_RAG_CORPUS_STORE": EnvVarConfig(
            description="Default corpus store type",
            required=False,
            var_type="str",
            default="sqlite",
            group="Core Configuration",
            importance="important",
            example="sqlite",
            choices=["sqlite", "memory", "chroma", "faiss"],
        ),
        
        "REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K": EnvVarConfig(
            description="Top-K results for retriever",
            required=False,
            var_type="int",
            default="10",
            group="Query Configuration",
            importance="important",
            example="10",
        ),
        
        "REFINIRE_RAG_LOG_LEVEL": EnvVarConfig(
            description="Logging level",
            required=False,
            var_type="str",
            default="INFO",
            group="System Configuration",
            importance="important",
            example="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        ),
        
        # Plugin System Configuration (プラグインシステム設定)
        "REFINIRE_RAG_DOCUMENT_STORES": EnvVarConfig(
            description="Comma-separated list of document store plugins",
            required=False,
            var_type="str",
            default="sqlite",
            group="Plugin Configuration",
            importance="important",
            example="sqlite,memory",
        ),
        
        "REFINIRE_RAG_VECTOR_STORES": EnvVarConfig(
            description="Comma-separated list of vector store plugins",
            required=False,
            var_type="str",
            default="inmemory_vector",
            group="Plugin Configuration",
            importance="important",
            example="inmemory_vector,pickle_vector",
        ),
        
        "REFINIRE_RAG_KEYWORD_STORES": EnvVarConfig(
            description="Comma-separated list of keyword store plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="tfidf_keyword",
        ),
        
        "REFINIRE_RAG_RETRIEVERS": EnvVarConfig(
            description="Comma-separated list of retriever plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="simple,hybrid",
        ),
        
        "REFINIRE_RAG_RERANKERS": EnvVarConfig(
            description="Comma-separated list of reranker plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="simple",
        ),
        
        "REFINIRE_RAG_SYNTHESIZERS": EnvVarConfig(
            description="Comma-separated list of synthesizer plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="answer",
        ),
        
        "REFINIRE_RAG_EVALUATORS": EnvVarConfig(
            description="Comma-separated list of evaluator plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="bleu,rouge,llm_judge",
        ),
        
        # Optional Variables (オプション変数)
        "REFINIRE_DEFAULT_LLM_MODEL": EnvVarConfig(
            description="Fallback LLM model",
            required=False,
            var_type="str",
            default="gpt-4o-mini",
            group="Core Configuration",
            importance="optional",
            example="gpt-4o-mini",
        ),
        
        "REFINIRE_DIR": EnvVarConfig(
            description="Base directory for Refinire files",
            required=False,
            var_type="str",
            default="./refinire",
            group="Core Configuration",
            importance="optional",
            example="./refinire",
        ),
        
        "REFINIRE_RAG_ENABLE_TELEMETRY": EnvVarConfig(
            description="Enable OpenTelemetry tracing (planned feature)",
            required=False,
            var_type="bool",
            default="true",
            group="System Configuration",
            importance="optional",
            example="true",
        ),
        
        # Embedding Configuration (埋め込み設定)
        "REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME": EnvVarConfig(
            description="OpenAI embedding model name",
            required=False,
            var_type="str",
            default="text-embedding-3-small",
            group="Embedding Configuration",
            importance="optional",
            example="text-embedding-3-small",
        ),
        
        "REFINIRE_RAG_OPENAI_EMBEDDING_API_KEY": EnvVarConfig(
            description="OpenAI API key for embeddings (falls back to OPENAI_API_KEY)",
            required=False,
            var_type="str",
            default="",
            group="Embedding Configuration",
            importance="optional",
            example="sk-proj-...",
        ),
        
        "REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION": EnvVarConfig(
            description="Embedding dimension",
            required=False,
            var_type="int",
            default="1536",
            group="Embedding Configuration",
            importance="optional",
            example="1536",
        ),
        
        "REFINIRE_RAG_OPENAI_EMBEDDING_BATCH_SIZE": EnvVarConfig(
            description="Batch size for embedding requests",
            required=False,
            var_type="int",
            default="100",
            group="Embedding Configuration",
            importance="optional",
            example="100",
        ),
        
        # Component Configuration (コンポーネント設定)
        # Retriever Configuration
        "REFINIRE_RAG_RETRIEVER_TOP_K": EnvVarConfig(
            description="Maximum number of results to retrieve",
            required=False,
            var_type="int",
            default="10",
            group="Component Configuration",
            importance="optional",
            example="10",
        ),
        
        "REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD": EnvVarConfig(
            description="Minimum similarity threshold for retrieval",
            required=False,
            var_type="float",
            default="0.0",
            group="Component Configuration",
            importance="optional",
            example="0.0",
        ),
        
        "REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING": EnvVarConfig(
            description="Enable metadata filtering for retrieval",
            required=False,
            var_type="bool",
            default="true",
            group="Component Configuration",
            importance="optional",
            example="true",
        ),
        
        "REFINIRE_RAG_RETRIEVER_VECTOR_STORE": EnvVarConfig(
            description="Vector store plugin name for SimpleRetriever",
            required=False,
            var_type="str",
            default="inmemory_vector",
            group="Component Configuration",
            importance="optional",
            example="inmemory_vector",
        ),
        
        "REFINIRE_RAG_RETRIEVER_EMBEDDER": EnvVarConfig(
            description="Embedder plugin name for SimpleRetriever",
            required=False,
            var_type="str",
            default="openai",
            group="Component Configuration",
            importance="optional",
            example="openai",
        ),
        
        # Hybrid Retriever Configuration
        "REFINIRE_RAG_HYBRID_FUSION_METHOD": EnvVarConfig(
            description="Fusion method for hybrid retrieval (rrf, weighted, max)",
            required=False,
            var_type="str",
            default="rrf",
            group="Component Configuration",
            importance="optional",
            example="rrf",
        ),
        
        "REFINIRE_RAG_HYBRID_RRF_K": EnvVarConfig(
            description="RRF parameter k for reciprocal rank fusion",
            required=False,
            var_type="int",
            default="60",
            group="Component Configuration",
            importance="optional",
            example="60",
        ),
        
        "REFINIRE_RAG_HYBRID_RETRIEVERS": EnvVarConfig(
            description="Comma-separated list of retriever names for hybrid retrieval",
            required=False,
            var_type="str",
            default="simple,tfidf_keyword",
            group="Component Configuration",
            importance="optional",
            example="simple,tfidf_keyword",
        ),
        
        "REFINIRE_RAG_HYBRID_RETRIEVER_WEIGHTS": EnvVarConfig(
            description="Comma-separated list of weights for hybrid retrievers",
            required=False,
            var_type="str",
            default="",
            group="Component Configuration",
            importance="optional",
            example="1.0,0.5",
        ),
        
        # Reranker Configuration
        "REFINIRE_RAG_RERANKER_SCORE_THRESHOLD": EnvVarConfig(
            description="Minimum score threshold for reranker",
            required=False,
            var_type="float",
            default="0.0",
            group="Component Configuration",
            importance="optional",
            example="0.0",
        ),
        
        "REFINIRE_RAG_RERANKER_BOOST_EXACT_MATCHES": EnvVarConfig(
            description="Boost exact term matches in reranking",
            required=False,
            var_type="bool",
            default="true",
            group="Component Configuration",
            importance="optional",
            example="true",
        ),
        
        "REFINIRE_RAG_RERANKER_BOOST_RECENT_DOCS": EnvVarConfig(
            description="Boost recent documents in reranking",
            required=False,
            var_type="bool",
            default="false",
            group="Component Configuration",
            importance="optional",
            example="false",
        ),
        
        "REFINIRE_RAG_RERANKER_LENGTH_PENALTY_FACTOR": EnvVarConfig(
            description="Length penalty factor for reranking",
            required=False,
            var_type="float",
            default="0.1",
            group="Component Configuration",
            importance="optional",
            example="0.1",
        ),
        
        # Synthesizer Configuration
        "REFINIRE_RAG_SYNTHESIZER_MAX_CONTEXT_LENGTH": EnvVarConfig(
            description="Maximum context length for answer synthesis",
            required=False,
            var_type="int",
            default="2000",
            group="Component Configuration",
            importance="optional",
            example="2000",
        ),
        
        "REFINIRE_RAG_SYNTHESIZER_TEMPERATURE": EnvVarConfig(
            description="Temperature for answer generation",
            required=False,
            var_type="float",
            default="0.1",
            group="Component Configuration",
            importance="optional",
            example="0.1",
        ),
        
        "REFINIRE_RAG_SYNTHESIZER_MAX_TOKENS": EnvVarConfig(
            description="Maximum tokens for answer generation",
            required=False,
            var_type="int",
            default="500",
            group="Component Configuration",
            importance="optional",
            example="500",
        ),
        
        "REFINIRE_RAG_SYNTHESIZER_GENERATION_INSTRUCTIONS": EnvVarConfig(
            description="Generation instructions for Refinire LLM",
            required=False,
            var_type="str",
            default="You are a helpful assistant that answers questions based on the provided context.",
            group="Component Configuration",
            importance="optional",
            example="You are a helpful assistant that answers questions based on the provided context.",
        ),
        
        "REFINIRE_RAG_SYNTHESIZER_SYSTEM_PROMPT": EnvVarConfig(
            description="System prompt for OpenAI completions",
            required=False,
            var_type="str",
            default="You are a helpful assistant that answers questions based on the provided context.",
            group="Component Configuration",
            importance="optional",
            example="You are a helpful assistant that answers questions based on the provided context.",
        ),
        
        # Query Engine Configuration (クエリエンジン設定)
        "REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION": EnvVarConfig(
            description="Enable query normalization",
            required=False,
            var_type="bool",
            default="true",
            group="Query Configuration",
            importance="optional",
            example="true",
        ),
        
        "REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K": EnvVarConfig(
            description="Total top-K results",
            required=False,
            var_type="int",
            default="20",
            group="Query Configuration",
            importance="optional",
            example="20",
        ),
        
        "REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K": EnvVarConfig(
            description="Top-K results for reranker",
            required=False,
            var_type="int",
            default="5",
            group="Query Configuration",
            importance="optional",
            example="5",
        ),
        
        "REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING": EnvVarConfig(
            description="Enable result caching",
            required=False,
            var_type="bool",
            default="true",
            group="Query Configuration",
            importance="optional",
            example="true",
        ),
        
        # Corpus Manager Configuration (コーパスマネージャー設定)
        "REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE": EnvVarConfig(
            description="Processing batch size",
            required=False,
            var_type="int",
            default="100",
            group="Processing Configuration",
            importance="optional",
            example="100",
        ),
        
        "REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING": EnvVarConfig(
            description="Enable parallel processing",
            required=False,
            var_type="bool",
            default="false",
            group="Processing Configuration",
            importance="optional",
            example="false",
        ),
        
        "REFINIRE_RAG_CORPUS_MANAGER_FAIL_ON_ERROR": EnvVarConfig(
            description="Fail on processing error",
            required=False,
            var_type="bool",
            default="false",
            group="Processing Configuration",
            importance="optional",
            example="false",
        ),
        
        # Evaluation Configuration (評価設定)  
        "REFINIRE_RAG_QUALITY_LAB_QA_GENERATION_MODEL": EnvVarConfig(
            description="Model for QA generation",
            required=False,
            var_type="str",
            default="gpt-4o-mini",
            group="Evaluation Configuration",
            importance="optional",
            example="gpt-4o-mini",
        ),
        
        "REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT": EnvVarConfig(
            description="Evaluation timeout in seconds",
            required=False,
            var_type="float",
            default="30.0",
            group="Evaluation Configuration",
            importance="optional",
            example="30.0",
        ),
        
        "REFINIRE_RAG_QUALITY_LAB_SIMILARITY_THRESHOLD": EnvVarConfig(
            description="Similarity threshold for evaluation",
            required=False,
            var_type="float",
            default="0.7",
            group="Evaluation Configuration",
            importance="optional",
            example="0.7",
        ),
        
        # QualityLab Configuration (QualityLab設定)
        "REFINIRE_RAG_QA_GENERATION_MODEL": EnvVarConfig(
            description="LLM model for QA pair generation",
            required=False,
            var_type="str",
            default="gpt-4o-mini",
            group="QualityLab Configuration",
            importance="optional",
            example="gpt-4o-mini",
        ),
        
        "REFINIRE_RAG_QA_PAIRS_PER_DOCUMENT": EnvVarConfig(
            description="Number of QA pairs to generate per document",
            required=False,
            var_type="int",
            default="3",
            group="QualityLab Configuration",
            importance="optional",
            example="3",
        ),
        
        "REFINIRE_RAG_QUESTION_TYPES": EnvVarConfig(
            description="Comma-separated list of question types to generate",
            required=False,
            var_type="str",
            default="factual,conceptual,analytical,comparative",
            group="QualityLab Configuration",
            importance="optional",
            example="factual,conceptual,analytical,comparative",
        ),
        
        "REFINIRE_RAG_OUTPUT_FORMAT": EnvVarConfig(
            description="Output format for evaluation reports",
            required=False,
            var_type="str",
            default="markdown",
            group="QualityLab Configuration",
            importance="optional",
            example="markdown",
        ),
        
        "REFINIRE_RAG_INCLUDE_DETAILED_ANALYSIS": EnvVarConfig(
            description="Include detailed analysis in evaluation reports",
            required=False,
            var_type="bool",
            default="true",
            group="QualityLab Configuration",
            importance="optional",
            example="true",
        ),
        
        "REFINIRE_RAG_INCLUDE_CONTRADICTION_DETECTION": EnvVarConfig(
            description="Enable contradiction detection in evaluations",
            required=False,
            var_type="bool",
            default="true",
            group="QualityLab Configuration",
            importance="optional",
            example="true",
        ),
        
        "REFINIRE_RAG_EVALUATION_DB_PATH": EnvVarConfig(
            description="Path to SQLite database for evaluation results",
            required=False,
            var_type="str",
            default="./data/evaluation.db",
            group="QualityLab Configuration",
            importance="optional",
            example="./data/evaluation.db",
        ),
        
        # Loader Configuration (ローダー設定)
        "REFINIRE_RAG_LOADERS": EnvVarConfig(
            description="Comma-separated list of loader plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="text,csv,json,html,directory",
        ),
        
        "REFINIRE_RAG_PROCESSORS": EnvVarConfig(
            description="Comma-separated list of processor plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="normalizer,chunker,dictionary_maker",
        ),
        
        "REFINIRE_RAG_SPLITTERS": EnvVarConfig(
            description="Comma-separated list of splitter plugins",
            required=False,
            var_type="str",
            default="",
            group="Plugin Configuration",
            importance="optional",
            example="character,recursive_character,token",
        ),
        
        # Advanced Configuration (高度設定)
        "REFINIRE_RAG_ENABLE_ASYNC_PROCESSING": EnvVarConfig(
            description="Enable asynchronous processing",
            required=False,
            var_type="bool",
            default="false",
            group="Performance Configuration",
            importance="optional",
            example="false",
        ),
        
        "REFINIRE_RAG_MAX_WORKERS": EnvVarConfig(
            description="Maximum number of worker threads",
            required=False,
            var_type="int",
            default="4",
            group="Performance Configuration",
            importance="optional",
            example="4",
        ),
        
        "REFINIRE_RAG_CACHE_ENABLED": EnvVarConfig(
            description="Enable component caching",
            required=False,
            var_type="bool",
            default="true",
            group="Performance Configuration",
            importance="optional",
            example="true",
        ),
        
        "REFINIRE_RAG_CACHE_SIZE": EnvVarConfig(
            description="Maximum cache size (MB)",
            required=False,
            var_type="int",
            default="100",
            group="Performance Configuration",
            importance="optional",
            example="100",
        ),
        
        # File Path Configuration (ファイルパス設定)
        "REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH": EnvVarConfig(
            description="Dictionary file path",
            required=False,
            var_type="str",
            default="./data/domain_dictionary.md",
            group="File Path Configuration",
            importance="optional",
            example="./data/domain_dictionary.md",
        ),
        
        "REFINIRE_RAG_GRAPH_BUILDER_GRAPH_FILE_PATH": EnvVarConfig(
            description="Knowledge graph file path",
            required=False,
            var_type="str",
            default="./data/domain_knowledge_graph.md",
            group="File Path Configuration",
            importance="optional",
            example="./data/domain_knowledge_graph.md",
        ),
        
        "REFINIRE_RAG_TEST_SUITE_TEST_CASES_FILE": EnvVarConfig(
            description="Test cases file path",
            required=False,
            var_type="str",
            default="./data/test_cases.json",
            group="File Path Configuration",
            importance="optional",
            example="./data/test_cases.json",
        ),
    }
    
    return EnvTemplate(
        variables=variables,
        source="refinire-rag"
    )


# Legacy support function for backward compatibility
def get_env_template() -> EnvTemplate:
    """
    Returns the environment variable template for refinire-rag.
    This is a legacy function for backward compatibility.
    Use refinire_rag_env_template() instead.
    """
    """
    refinire-ragの環境変数テンプレートを返します。
    これは後方互換性のためのレガシー関数です。
    代わりにrefinire_rag_env_template()を使用してください。
    """
    return refinire_rag_env_template()