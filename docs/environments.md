# Environment Variables Configuration

## Overview / 概要

This document lists all environment variables that can be used to configure the refinire-rag system. Each configuration class can be customized using environment variables following the pattern `REFINIRE_RAG_{CLASS_NAME}_{SETTING_NAME}`.

このドキュメントでは、refinire-ragシステムを設定するために使用できるすべての環境変数を一覧表示しています。各設定クラスは、`REFINIRE_RAG_{CLASS_NAME}_{SETTING_NAME}`というパターンに従った環境変数を使用してカスタマイズできます。

## Importance Level Guide / 重要度レベルガイド

| Level | Icon | Description | Usage |
|-------|------|-------------|-------|
| **Critical** | 🔴 | **Must be set** for the system to function | Required for basic operation |
| **Important** | 🟡 | **Significantly affects** functionality or performance | Recommended for production |
| **Optional** | 🟢 | **Fine-tuning** and optimization settings | Advanced customization |

### Quick Priority Summary / 優先度クイックサマリー

**🔴 Critical (1 variable):**
- `OPENAI_API_KEY` - Required for OpenAI services

**🟡 Important (23 variables):**
- Model configurations, file paths, core feature toggles
- Performance settings, timeout values, batch sizes
- Essential for production deployment

**🟢 Optional (113+ variables):**
- LLM parameters, advanced features, debug settings
- Fine-tuning parameters, optional toggles
- Can be left at default values for initial setup

## Global Environment Variables / グローバル環境変数

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `OPENAI_API_KEY` | OpenAI API authentication key | None (required) | str | 🔴 **Critical** |
| `REFINIRE_RAG_LLM_MODEL` | Primary LLM model for RAG operations | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_DEFAULT_LLM_MODEL` | Fallback LLM model | "gpt-4o-mini" | str | 🟢 **Optional** |
| `REFINIRE_DIR` | Base directory for Refinire files | "./refinire" | str | 🟢 **Optional** |
| `REFINIRE_RAG_CORPUS_STORE` | Default corpus store type | "sqlite" | str | 🟡 **Important** |
| `REFINIRE_RAG_DATA_DIR` | Base data directory for all storage | "./data" | str | 🟡 **Important** |
| `REFINIRE_RAG_LOG_LEVEL` | Logging level | "INFO" | str | 🟢 **Optional** |
| `REFINIRE_RAG_ENABLE_TELEMETRY` | Enable OpenTelemetry tracing | "true" | bool | 🟢 **Optional** |

## Evaluation Configuration / 評価設定

### Base Evaluator Configuration / 基本評価器設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_BASE_EVALUATOR_NAME` | Evaluator name | "BaseEvaluator" | str | 🟢 **Optional** |
| `REFINIRE_RAG_BASE_EVALUATOR_ENABLED` | Enable evaluator | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_BASE_EVALUATOR_WEIGHT` | Evaluator weight in composite evaluation | "1.0" | float | 🟢 **Optional** |
| `REFINIRE_RAG_BASE_EVALUATOR_THRESHOLD` | Score threshold for evaluation | "0.5" | float | 🟢 **Optional** |

### QuestEval Configuration / QuestEval設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------||
| `REFINIRE_RAG_QUESTEVAL_NAME` | QuestEval evaluator name | "QuestEval" | str | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_MODEL_NAME` | LLM model for semantic evaluation | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_RAG_QUESTEVAL_ENABLE_CONSISTENCY` | Enable consistency evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_ENABLE_ANSWERABILITY` | Enable answerability evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_ENABLE_SOURCE_SUPPORT` | Enable source support evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_ENABLE_FLUENCY` | Enable fluency evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_CONSISTENCY_WEIGHT` | Weight for consistency component | "0.3" | float | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_ANSWERABILITY_WEIGHT` | Weight for answerability component | "0.25" | float | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_SOURCE_SUPPORT_WEIGHT` | Weight for source support component | "0.25" | float | 🟢 **Optional** |
| `REFINIRE_RAG_QUESTEVAL_FLUENCY_WEIGHT` | Weight for fluency component | "0.2" | float | 🟢 **Optional** |

### BLEU Configuration / BLEU設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------||
| `REFINIRE_RAG_BLEU_NAME` | BLEU evaluator name | "BLEU" | str | 🟢 **Optional** |
| `REFINIRE_RAG_BLEU_MAX_N` | Maximum n-gram order | "4" | int | 🟢 **Optional** |
| `REFINIRE_RAG_BLEU_WEIGHTS` | Weights for n-gram orders (comma-separated) | "0.25,0.25,0.25,0.25" | str | 🟢 **Optional** |
| `REFINIRE_RAG_BLEU_SMOOTHING_FUNCTION` | Smoothing function type | "epsilon" | str | 🟢 **Optional** |
| `REFINIRE_RAG_BLEU_EPSILON` | Epsilon value for smoothing | "0.1" | float | 🟢 **Optional** |
| `REFINIRE_RAG_BLEU_CASE_SENSITIVE` | Case sensitivity for comparison | "false" | bool | 🟢 **Optional** |

### ROUGE Configuration / ROUGE設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------||
| `REFINIRE_RAG_ROUGE_NAME` | ROUGE evaluator name | "ROUGE" | str | 🟢 **Optional** |
| `REFINIRE_RAG_ROUGE_TYPES` | ROUGE types to compute (comma-separated) | "rouge-1,rouge-2,rouge-l" | str | 🟢 **Optional** |
| `REFINIRE_RAG_ROUGE_MAX_N` | Maximum n-gram for ROUGE-N | "2" | int | 🟢 **Optional** |
| `REFINIRE_RAG_ROUGE_USE_STEMMING` | Enable stemming | "false" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_ROUGE_CASE_SENSITIVE` | Case sensitivity | "false" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_ROUGE_REMOVE_STOPWORDS` | Remove stopwords | "false" | bool | 🟢 **Optional** |

### LLM Judge Configuration / LLM Judge設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------||
| `REFINIRE_RAG_LLM_JUDGE_NAME` | LLM Judge evaluator name | "LLM_Judge" | str | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_MODEL_NAME` | LLM model for judging | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_RAG_LLM_JUDGE_JUDGE_MODEL_NAME` | Alternative judge model name | "gpt-4o-mini" | str | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_EVALUATION_CRITERIA` | Evaluation criteria (comma-separated) | "relevance,accuracy,completeness,coherence,helpfulness" | str | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_SCORING_SCALE` | Scoring scale (1-N) | "10" | int | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_INCLUDE_REASONING` | Include reasoning in output | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_TEMPERATURE` | Temperature for LLM | "0.1" | float | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_MAX_TOKENS` | Maximum tokens for judge response | "1000" | int | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_ENABLE_RELEVANCE` | Enable relevance evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_ENABLE_ACCURACY` | Enable accuracy evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_ENABLE_COMPLETENESS` | Enable completeness evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_ENABLE_COHERENCE` | Enable coherence evaluation | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_LLM_JUDGE_ENABLE_HELPFULNESS` | Enable helpfulness evaluation | "true" | bool | 🟢 **Optional** |

## Processing Configuration / 処理設定

### Dictionary Maker Configuration / 辞書作成設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------||
| `REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH` | Dictionary file path | "./data/domain_dictionary.md" | str | 🟡 **Important** |
| `REFINIRE_RAG_DICTIONARY_MAKER_BACKUP_DICTIONARY` | Backup dictionary files | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_LLM_MODEL` | LLM model for term extraction | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_RAG_DICTIONARY_MAKER_LLM_TEMPERATURE` | LLM temperature | "0.3" | float | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_MAX_TOKENS` | Maximum tokens for LLM | "2000" | int | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_FOCUS_ON_TECHNICAL_TERMS` | Focus on technical terms | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_EXTRACT_ABBREVIATIONS` | Extract abbreviations | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_DETECT_EXPRESSION_VARIATIONS` | Detect expression variations | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_MIN_TERM_IMPORTANCE` | Minimum term importance | "medium" | str | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_SKIP_IF_NO_NEW_TERMS` | Skip processing if no new terms | "false" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_VALIDATE_EXTRACTED_TERMS` | Validate extracted terms | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_UPDATE_DOCUMENT_METADATA` | Update document metadata | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DICTIONARY_MAKER_PRESERVE_ORIGINAL_DOCUMENT` | Preserve original document | "true" | bool | 🟢 **Optional** |

### Graph Builder Configuration / グラフ構築設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_GRAPH_BUILDER_GRAPH_FILE_PATH` | Knowledge graph file path | "./data/domain_knowledge_graph.md" | str | 🟡 **Important** |
| `REFINIRE_RAG_GRAPH_BUILDER_DICTIONARY_FILE_PATH` | Dictionary file path | "./data/domain_dictionary.md" | str | 🟡 **Important** |
| `REFINIRE_RAG_GRAPH_BUILDER_BACKUP_GRAPH` | Backup graph files | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_LLM_MODEL` | LLM model for graph building | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_RAG_GRAPH_BUILDER_LLM_TEMPERATURE` | LLM temperature | "0.3" | float | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_MAX_TOKENS` | Maximum tokens for LLM | "3000" | int | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_FOCUS_ON_IMPORTANT_RELATIONSHIPS` | Focus on important relationships | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_EXTRACT_HIERARCHICAL_RELATIONSHIPS` | Extract hierarchical relationships | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_EXTRACT_CAUSAL_RELATIONSHIPS` | Extract causal relationships | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_EXTRACT_COMPOSITION_RELATIONSHIPS` | Extract composition relationships | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_MIN_RELATIONSHIP_IMPORTANCE` | Minimum relationship importance | "medium" | str | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_USE_DICTIONARY_TERMS` | Use dictionary terms | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_GRAPH_BUILDER_AUTO_DETECT_DICTIONARY_PATH` | Auto-detect dictionary path | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_SKIP_IF_NO_NEW_RELATIONSHIPS` | Skip if no new relationships | "false" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_VALIDATE_EXTRACTED_RELATIONSHIPS` | Validate extracted relationships | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_DEDUPLICATE_RELATIONSHIPS` | Deduplicate relationships | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_UPDATE_DOCUMENT_METADATA` | Update document metadata | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_GRAPH_BUILDER_PRESERVE_ORIGINAL_DOCUMENT` | Preserve original document | "true" | bool | 🟢 **Optional** |

### Test Suite Configuration / テストスイート設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_TEST_SUITE_TEST_CASES_FILE` | Test cases file path | "./data/test_cases.json" | str | 🟡 **Important** |
| `REFINIRE_RAG_TEST_SUITE_RESULTS_OUTPUT_FILE` | Results output file path | "./data/test_results.json" | str | 🟡 **Important** |
| `REFINIRE_RAG_TEST_SUITE_AUTO_GENERATE_CASES` | Auto-generate test cases | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_TEST_SUITE_MAX_CASES_PER_DOCUMENT` | Maximum cases per document | "3" | int | 🟢 **Optional** |
| `REFINIRE_RAG_TEST_SUITE_INCLUDE_NEGATIVE_CASES` | Include negative test cases | "true" | bool | 🟢 **Optional** |

### Evaluator Configuration / 評価器設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_EVALUATOR_INCLUDE_CATEGORY_ANALYSIS` | Include category analysis | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_EVALUATOR_INCLUDE_TEMPORAL_ANALYSIS` | Include temporal analysis | "false" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_EVALUATOR_INCLUDE_FAILURE_ANALYSIS` | Include failure analysis | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_EVALUATOR_CONFIDENCE_THRESHOLD` | Confidence threshold | "0.7" | float | 🟡 **Important** |
| `REFINIRE_RAG_EVALUATOR_RESPONSE_TIME_THRESHOLD` | Response time threshold (seconds) | "2.0" | float | 🟡 **Important** |
| `REFINIRE_RAG_EVALUATOR_ACCURACY_THRESHOLD` | Accuracy threshold | "0.8" | float | 🟡 **Important** |
| `REFINIRE_RAG_EVALUATOR_OUTPUT_FORMAT` | Output format | "markdown" | str | 🟢 **Optional** |

## Application Configuration / アプリケーション設定

### Quality Lab Configuration / QualityLab設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_QUALITY_LAB_QA_GENERATION_MODEL` | Model for QA generation | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_RAG_QUALITY_LAB_QA_PAIRS_PER_DOCUMENT` | QA pairs per document | "3" | int | 🟢 **Optional** |
| `REFINIRE_RAG_QUALITY_LAB_QUESTION_TYPES` | Question types (comma-separated) | "factual,conceptual,analytical,comparative" | str | 🟢 **Optional** |
| `REFINIRE_RAG_QUALITY_LAB_EVALUATION_TIMEOUT` | Evaluation timeout (seconds) | "30.0" | float | 🟡 **Important** |
| `REFINIRE_RAG_QUALITY_LAB_SIMILARITY_THRESHOLD` | Similarity threshold | "0.7" | float | 🟡 **Important** |
| `REFINIRE_RAG_QUALITY_LAB_OUTPUT_FORMAT` | Output format | "markdown" | str | 🟢 **Optional** |
| `REFINIRE_RAG_QUALITY_LAB_INCLUDE_DETAILED_ANALYSIS` | Include detailed analysis | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUALITY_LAB_INCLUDE_CONTRADICTION_DETECTION` | Include contradiction detection | "true" | bool | 🟡 **Important** |

### Query Engine Configuration / クエリエンジン設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_QUERY_ENGINE_ENABLE_QUERY_NORMALIZATION` | Enable query normalization | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K` | Top-K for retriever | "10" | int | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_TOTAL_TOP_K` | Total top-K | "20" | int | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_RERANKER_TOP_K` | Top-K for reranker | "5" | int | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_SYNTHESIZER_MAX_CONTEXT` | Max context for synthesizer | "2000" | int | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING` | Enable caching | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_CACHE_TTL` | Cache TTL (seconds) | "3600" | int | 🟢 **Optional** |
| `REFINIRE_RAG_QUERY_ENGINE_INCLUDE_SOURCES` | Include sources in response | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_QUERY_ENGINE_INCLUDE_CONFIDENCE` | Include confidence scores | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUERY_ENGINE_INCLUDE_PROCESSING_METADATA` | Include processing metadata | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUERY_ENGINE_DEDUPLICATE_RESULTS` | Deduplicate results | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_QUERY_ENGINE_COMBINE_SCORES` | Score combination method | "max" | str | 🟢 **Optional** |

### Corpus Manager Configuration / コーパスマネージャー設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_CORPUS_MANAGER_ENABLE_PROCESSING` | Enable document processing | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_ENABLE_CHUNKING` | Enable chunking | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_ENABLE_EMBEDDING` | Enable embedding | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_AUTO_FIT_EMBEDDER` | Auto-fit embedder | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_CORPUS_MANAGER_STORE_INTERMEDIATE_RESULTS` | Store intermediate results | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE` | Batch size | "100" | int | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING` | Enable parallel processing | "false" | bool | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_FAIL_ON_ERROR` | Fail on error | "false" | bool | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_MAX_ERRORS` | Maximum errors allowed | "10" | int | 🟡 **Important** |
| `REFINIRE_RAG_CORPUS_MANAGER_ENABLE_PROGRESS_REPORTING` | Enable progress reporting | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_CORPUS_MANAGER_PROGRESS_INTERVAL` | Progress reporting interval | "10" | int | 🟢 **Optional** |

## Embedding Configuration / 埋め込み設定

### OpenAI Embedding Configuration / OpenAI埋め込み設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME` | OpenAI embedding model name | "text-embedding-3-small" | str | 🟡 **Important** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_API_KEY` | OpenAI API key | "" | str | 🔴 **Critical** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_API_BASE` | OpenAI API base URL | "https://api.openai.com/v1" | str | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_ORGANIZATION` | OpenAI organization | "" | str | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_EMBEDDING_DIMENSION` | Embedding dimension | "1536" | int | 🟡 **Important** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_BATCH_SIZE` | Batch size | "100" | int | 🟡 **Important** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_MAX_TOKENS` | Maximum tokens | "8191" | int | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_REQUESTS_PER_MINUTE` | Requests per minute limit | "3000" | int | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_MAX_RETRIES` | Maximum retries | "3" | int | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_RETRY_DELAY_SECONDS` | Retry delay (seconds) | "1.0" | float | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_STRIP_NEWLINES` | Strip newlines from text | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_OPENAI_EMBEDDING_USER_IDENTIFIER` | User identifier | "" | str | 🟢 **Optional** |

### Base Embedding Configuration / 基本埋め込み設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_EMBEDDING_MODEL_NAME` | Default embedding model name | "text-embedding-3-small" | str | 🟡 **Important** |
| `REFINIRE_RAG_EMBEDDING_EMBEDDING_DIMENSION` | Embedding dimension | "768" | int | 🟡 **Important** |
| `REFINIRE_RAG_EMBEDDING_NORMALIZE_VECTORS` | Normalize vectors | "true" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_EMBEDDING_BATCH_SIZE` | Batch size | "100" | int | 🟡 **Important** |
| `REFINIRE_RAG_EMBEDDING_MAX_TOKENS` | Maximum tokens | "8192" | int | 🟢 **Optional** |
| `REFINIRE_RAG_EMBEDDING_ENABLE_CACHING` | Enable caching | "true" | bool | 🟡 **Important** |
| `REFINIRE_RAG_EMBEDDING_CACHE_TTL_SECONDS` | Cache TTL (seconds) | "3600" | int | 🟢 **Optional** |
| `REFINIRE_RAG_EMBEDDING_MAX_RETRIES` | Maximum retries | "3" | int | 🟢 **Optional** |
| `REFINIRE_RAG_EMBEDDING_RETRY_DELAY_SECONDS` | Retry delay (seconds) | "1.0" | float | 🟢 **Optional** |
| `REFINIRE_RAG_EMBEDDING_FAIL_ON_ERROR` | Fail on error | "true" | bool | 🟡 **Important** |

## Retrieval Configuration / 検索設定

### Retriever Configuration / リトリーバー設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_RETRIEVER_TOP_K` | Top-K results | "10" | int | 🟡 **Important** |
| `REFINIRE_RAG_RETRIEVER_SIMILARITY_THRESHOLD` | Similarity threshold | "0.0" | float | 🟡 **Important** |
| `REFINIRE_RAG_RETRIEVER_ENABLE_FILTERING` | Enable result filtering | "true" | bool | 🟡 **Important** |

### Reranker Configuration / リランカー設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_RERANKER_TOP_K` | Top-K results after reranking | "5" | int | 🟡 **Important** |
| `REFINIRE_RAG_RERANKER_RERANK_MODEL` | Reranking model | "cross-encoder" | str | 🟡 **Important** |
| `REFINIRE_RAG_RERANKER_SCORE_THRESHOLD` | Score threshold | "0.0" | float | 🟡 **Important** |

### Answer Synthesizer Configuration / 回答合成設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_ANSWER_SYNTHESIZER_MAX_CONTEXT_LENGTH` | Maximum context length | "2000" | int | 🟡 **Important** |
| `REFINIRE_RAG_ANSWER_SYNTHESIZER_LLM_MODEL` | LLM model for synthesis | "gpt-4o-mini" | str | 🟡 **Important** |
| `REFINIRE_RAG_ANSWER_SYNTHESIZER_TEMPERATURE` | LLM temperature | "0.1" | float | 🟢 **Optional** |
| `REFINIRE_RAG_ANSWER_SYNTHESIZER_MAX_TOKENS` | Maximum tokens in response | "500" | int | 🟢 **Optional** |

## Loader Configuration / ローダー設定

### Document Load Configuration / 文書読み込み設定

| Environment Variable | Description | Default Value | Type | Importance |
|---------------------|-------------|---------------|------|-----------|
| `REFINIRE_RAG_DOCUMENT_LOAD_STRATEGY` | Loading strategy | "FULL" | str | 🟡 **Important** |
| `REFINIRE_RAG_DOCUMENT_LOAD_BATCH_SIZE` | Batch size | "100" | int | 🟡 **Important** |
| `REFINIRE_RAG_DOCUMENT_LOAD_MAX_DOCUMENTS` | Maximum documents to load | "1000" | int | 🟡 **Important** |
| `REFINIRE_RAG_DOCUMENT_LOAD_SORT_BY` | Sort field | "created_at" | str | 🟢 **Optional** |
| `REFINIRE_RAG_DOCUMENT_LOAD_SORT_ORDER` | Sort order | "desc" | str | 🟢 **Optional** |
| `REFINIRE_RAG_DOCUMENT_LOAD_INCLUDE_DELETED` | Include deleted documents | "false" | bool | 🟢 **Optional** |
| `REFINIRE_RAG_DOCUMENT_LOAD_VALIDATE_DOCUMENTS` | Validate documents | "true" | bool | 🟡 **Important** |

## Usage Examples / 使用例

### Basic Configuration / 基本設定

```bash
# Set LLM model for all components
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"

# Configure evaluation
export REFINIRE_RAG_QUESTEVAL_MODEL_NAME="gpt-4o-mini"
export REFINIRE_RAG_LLM_JUDGE_SCORING_SCALE=5

# Configure OpenAI embedding
export OPENAI_API_KEY="your_api_key_here"
export REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME="text-embedding-3-large"
```

### Evaluation-focused Configuration / 評価重視設定

```bash
# Enable all evaluation metrics
export REFINIRE_RAG_QUESTEVAL_ENABLED=true
export REFINIRE_RAG_BLEU_ENABLED=true
export REFINIRE_RAG_ROUGE_ENABLED=true
export REFINIRE_RAG_LLM_JUDGE_ENABLED=true

# Configure detailed evaluation
export REFINIRE_RAG_QUALITY_LAB_INCLUDE_DETAILED_ANALYSIS=true
export REFINIRE_RAG_QUALITY_LAB_INCLUDE_CONTRADICTION_DETECTION=true
export REFINIRE_RAG_EVALUATOR_INCLUDE_FAILURE_ANALYSIS=true
```

### Production Configuration / 本番設定

```bash
# Configure for production workloads
export REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE=200
export REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING=true
export REFINIRE_RAG_QUERY_ENGINE_ENABLE_CACHING=true
export REFINIRE_RAG_QUERY_ENGINE_CACHE_TTL=7200

# Set performance thresholds
export REFINIRE_RAG_EVALUATOR_RESPONSE_TIME_THRESHOLD=1.0
export REFINIRE_RAG_EVALUATOR_ACCURACY_THRESHOLD=0.9
```

## Default Values Philosophy / デフォルト値の設計思想

### Production-Ready Defaults / 本番対応デフォルト値

All default values are designed to work in production environments with minimal configuration:

すべてのデフォルト値は、最小限の設定で本番環境で動作するように設計されています：

- **LLM Model**: "gpt-4o-mini" - Cost-effective model with good performance
- **File Paths**: "./data/" prefix - Organized data directory structure  
- **Timeouts**: Conservative values to prevent hanging
- **Batch Sizes**: Balanced for memory usage and performance
- **Caching**: Enabled by default with reasonable TTL values
- **Error Handling**: Fail-safe defaults with proper error reporting

### Environment-Specific Overrides / 環境固有のオーバーライド

| Environment | Recommended Overrides | Purpose |
|-------------|----------------------|---------|
| **Development** | `REFINIRE_RAG_LOG_LEVEL="DEBUG"` | Detailed logging |
| | `REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE="10"` | Faster iteration |
| | `REFINIRE_RAG_EVALUATOR_INCLUDE_FAILURE_ANALYSIS="true"` | Debug evaluation issues |
| **Testing** | `REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"` | Consistent test results |
| | `REFINIRE_RAG_QUALITY_LAB_QA_PAIRS_PER_DOCUMENT="1"` | Faster test execution |
| | `REFINIRE_RAG_ENABLE_TELEMETRY="false"` | No external calls in tests |
| **Production** | `REFINIRE_RAG_LOG_LEVEL="INFO"` | Reduced log volume |
| | `REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING="true"` | Better performance |
| | `REFINIRE_RAG_QUERY_ENGINE_CACHE_TTL="7200"` | Longer cache duration |
| | `REFINIRE_RAG_EVALUATOR_RESPONSE_TIME_THRESHOLD="1.0"` | Strict performance monitoring |

## Configuration Validation / 設定検証

### Critical Environment Variables (🔴) / 必須環境変数

These variables **must** be set for the system to function:

| Variable | Reason | Example |
|----------|--------|---------|
| `OPENAI_API_KEY` | OpenAI API access required for embeddings and LLM operations | `sk-proj-...` |

### Important Environment Variables (🟡) / 重要環境変数

These variables should be set for production deployment and optimal operation:

| Category | Variables | Reason |
|----------|-----------|--------|
| **Model Configuration** | `REFINIRE_RAG_LLM_MODEL` | Consistent LLM model across components |
| **Data Management** | `REFINIRE_RAG_DATA_DIR` | Centralized data storage location |
| | `REFINIRE_RAG_CORPUS_STORE` | Database storage type selection |
| **Core Features** | `REFINIRE_RAG_CORPUS_MANAGER_ENABLE_*` | Enable/disable core processing features |
| | `REFINIRE_RAG_QUERY_ENGINE_ENABLE_*` | Control query processing behavior |
| **Performance** | `REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE` | Processing batch size optimization |
| | `REFINIRE_RAG_QUERY_ENGINE_RETRIEVER_TOP_K` | Search result limits |
| | `REFINIRE_RAG_EVALUATOR_*_THRESHOLD` | Performance monitoring thresholds |
| **File Paths** | `REFINIRE_RAG_DICTIONARY_MAKER_DICTIONARY_FILE_PATH` | Domain dictionary location |
| | `REFINIRE_RAG_OPENAI_EMBEDDING_MODEL_NAME` | Embedding model selection |

### Optional Environment Variables (🟢) / オプション環境変数

These variables provide fine-tuning capabilities but can be left at default values:

| Category | Examples | Purpose |
|----------|----------|---------|
| **LLM Parameters** | `*_TEMPERATURE`, `*_MAX_TOKENS` | Model behavior fine-tuning |
| **Advanced Features** | `*_BACKUP_*`, `*_INCLUDE_*_ANALYSIS` | Enhanced functionality |
| **Debug/Monitoring** | `*_LOG_LEVEL`, `*_ENABLE_TELEMETRY` | Development and monitoring |
| **Evaluation Tuning** | `*_WEIGHT`, `*_SCORING_SCALE` | Evaluation metric customization |

## Notes / 注意事項

1. **Type Conversion / 型変換**: Boolean values should be "true" or "false" (case-insensitive), integers should be numeric strings, floats should include decimal points.

2. **List Values / リスト値**: For list-type configurations (like weights or criteria), use comma-separated values without spaces.

3. **File Paths / ファイルパス**: All file paths support both absolute and relative paths. Relative paths are resolved from the current working directory. Use `REFINIRE_RAG_DATA_DIR` as a base for organization.

4. **Model Names / モデル名**: LLM model names should match the format expected by the Refinire library (e.g., "gpt-4o-mini", "claude-3-sonnet").

5. **Priority / 優先順位**: Environment variables take precedence over default values in configuration classes.

6. **Validation / 検証**: Invalid environment variable values will fall back to default values with a warning message.

7. **Security / セキュリティ**: Never commit API keys or sensitive configuration to version control. Use environment variables or secure secret management.

8. **Performance / パフォーマンス**: Batch sizes and timeout values may need adjustment based on your hardware and network conditions.

## Quick Start Configuration / クイックスタート設定

### Minimal Setup / 最小設定

```bash
# Essential configuration for basic operation
export OPENAI_API_KEY="your_openai_api_key"
export REFINIRE_RAG_DATA_DIR="./data"
```

### Development Setup / 開発環境設定

```bash
# Development-optimized configuration
export OPENAI_API_KEY="your_openai_api_key"
export REFINIRE_RAG_DATA_DIR="./data"
export REFINIRE_RAG_LOG_LEVEL="DEBUG"
export REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE="10"
export REFINIRE_RAG_QUALITY_LAB_QA_PAIRS_PER_DOCUMENT="2"
```

### Production Setup / 本番環境設定

```bash
# Production-optimized configuration
export OPENAI_API_KEY="your_openai_api_key"
export REFINIRE_RAG_DATA_DIR="/app/data"
export REFINIRE_RAG_LOG_LEVEL="INFO"
export REFINIRE_RAG_CORPUS_MANAGER_PARALLEL_PROCESSING="true"
export REFINIRE_RAG_CORPUS_MANAGER_BATCH_SIZE="200"
export REFINIRE_RAG_QUERY_ENGINE_CACHE_TTL="7200"
export REFINIRE_RAG_EVALUATOR_RESPONSE_TIME_THRESHOLD="1.0"
```