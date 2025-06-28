# refinire-rag Concept Document

## Overview
refinire-ragは、Refinireライブラリのサブパッケージとして提供されるRAG（Retrieval-Augmented Generation）機能を実装するPythonライブラリです。モジュラーアーキテクチャを採用し、ユースケースをRefinire Stepサブクラスとして実装し、単一責務のバックエンドモジュールを提供します。

## Architecture

### Use Case Classes (Refinire Steps)
- **CorpusManager**: 文書の読み込み、正規化、チャンク分割、埋め込み生成、保存
- **QueryEngine**: 文書検索、再ランキング、回答生成
- **QualityLab**: 評価データ作成、自動RAG評価、矛盾検出、レポート生成

### Backend Modules (All implement DocumentProcessor)
- **Loader**: 外部ファイル → Document変換
- **DictionaryMaker**: LLMベースのドメイン固有用語抽出と累積MDディクショナリ
- **Normalizer**: MDディクショナリベースの表現バリエーション正規化
- **GraphBuilder**: LLMベースの関係抽出と累積MDナレッジグラフ
- **Chunker**: トークンベースのチャンク分割
- **VectorStoreProcessor**: チャンク → ベクター生成と保存 (Embedderを統合)
- **Retriever**: 文書検索
- **Reranker**: 候補再ランキング  
- **Reader**: LLMベースの回答生成
- **TestSuite**: 評価実行器
- **Evaluator**: メトリクス集約
- **ContradictionDetector**: 主張抽出 + NLI検出
- **InsightReporter**: 閾値ベースの解釈とレポート

## Current Implementation Status

### ✅ Completed Features
- Environment variable-based configuration system
- Plugin registry with built-in and external plugin support
- Hybrid search with vector and keyword retrieval
- Multiple reranker implementations (Heuristic, RRF, LLM)
- SQLite document storage
- OpenAI embeddings integration
- 3-step educational example (hybrid_rag_example.py)

### 🔧 Current Architecture Improvements
- Plugin pattern implementation across all components
- Automatic retriever creation from configured stores
- Graceful fallback mechanisms for missing components

## Future Extensions

### 🚀 Priority Enhancements

#### 1. RefinireAgent Context Integration
**Status**: Concept identified, not yet implemented
**Description**: QueryEngineの検索結果をRefinireAgentのContextProviderとして適切に提供する機能

**Current Issue**:
- SimpleAnswerSynthesizerは検索結果を単純なテキストとして結合している
- QueryEngineの検索結果（List[SearchResult]）がRefinireAgentのContextとして適切に渡されていない
- 現在のLLMは「I cannot find the answer in the provided context」と回答することが多い

**Proposed Solution**:
```python
# 将来の実装例
from refinire import Context, RefinireAgent

class RefinireContextProvider:
    def create_context_from_search_results(self, search_results: List[SearchResult]) -> Context:
        # SearchResultsをRefinire ContextオブジェクトとしてラップDictionary API
        # メタデータ、スコア、文書IDなどの情報を保持
        pass

class EnhancedAnswerSynthesizer(AnswerSynthesizer):
    def synthesize(self, query: str, contexts: List[SearchResult]) -> str:
        # RefinireAgentにContextProviderとして検索結果を渡す
        context = self.context_provider.create_context_from_search_results(contexts)
        agent = RefinireAgent(context_provider=context)
        return agent.generate_answer(query)
```

**Benefits**:
- より正確で文脈を理解した回答生成
- RefinireAgentの高度な推論能力の活用
- メタデータとスコア情報を保持した文脈提供
- Refinireエコシステムとの完全な統合

**Implementation Tasks**:
1. Refinire ContextとRefinireAgentのAPI調査
2. SearchResult → Contextの変換機能実装
3. EnhancedAnswerSynthesizerの作成
4. hybrid_rag_example.pyでのテスト
5. 既存のSimpleAnswerSynthesizerとの互換性維持

---

#### 2. Advanced Query Processing
- Query normalization using corpus dictionary
- Multi-step query decomposition
- Query expansion using domain knowledge graph

#### 3. Enhanced Retrieval Methods
- Dense passage retrieval integration
- Cross-encoder reranking
- Temporal and spatial search capabilities

#### 4. Production Features
- Caching layers for embedding and search results
- Batch processing capabilities
- API endpoint integration
- Monitoring and logging enhancements

#### 5. Quality Assessment
- Automated evaluation pipelines
- A/B testing framework for retrieval methods
- Bias detection and mitigation tools

## Technical Decisions

### Configuration Management
- Environment variable-based configuration with fallback defaults
- Plugin pattern for component discovery and instantiation
- Keyword argument → environment variable → default value hierarchy

### Integration Strategy
- Refinire Step subclasses for use case orchestration
- DocumentProcessor interface for processing pipeline uniformity
- Plugin registry for extensible component ecosystem

### Testing Strategy
- Comprehensive unit tests for all components
- Integration tests for end-to-end workflows
- Example-driven documentation and testing

## Dependencies

### Core Dependencies
- Python 3.10+
- Refinire library for LLM integration and agent framework
- OpenAI API for embeddings and language models

### Optional Plugin Dependencies
- refinire-rag-chroma: Chroma vector store integration
- refinire-rag-bm25s-j: BM25s keyword search integration

### Development Dependencies
- pytest for testing framework
- pytest-cov for coverage reporting
- uv for package management