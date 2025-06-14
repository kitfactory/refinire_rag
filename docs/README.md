# refinire-rag Documentation

Official documentation for the refined RAG framework that makes enterprise-grade document processing effortless.

## 🌟 Why refinire-rag?

Traditional RAG frameworks are powerful but complex. refinire-rag refines the development experience with radical simplicity and enterprise-grade productivity.

**[→ Why refinire-rag? The Complete Story](why_refinire_rag.md)** | **[→ なぜrefinire-rag？完全版](why_refinire_rag_ja.md)**

### ⚡ 10x Simpler Development
```python
# LangChain: 50+ lines of complex setup
# refinire-rag: 5 lines to production-ready RAG
manager = CorpusManager.create_simple_rag(doc_store, vector_store)
results = manager.process_corpus(["documents/"])
answer = query_engine.answer("How does this work?")
```

### 🏢 Enterprise-Ready Features Built-In
- **Incremental Processing**: Handle 10,000+ documents efficiently
- **Japanese Optimization**: Built-in linguistic processing
- **Access Control**: Department-level data isolation
- **Production Monitoring**: Comprehensive observability
- **Unified Architecture**: One pattern for everything

## 📚 Documentation Structure

### 🎯 Tutorials
Learn how to build RAG systems step by step - from simple prototypes to enterprise deployment.

- **English Version**
  - [Tutorial Overview](tutorials/tutorial_overview.md)
  - [Tutorial 1: Basic RAG Pipeline](tutorials/tutorial_01_basic_rag.md)
  - [Tutorial 6: Incremental Document Loading](tutorials/tutorial_06_incremental_loading.md)

- **Japanese Version** 🇯🇵
  - [チュートリアル概要](tutorials/tutorial_overview_ja.md)
  - [チュートリアル1: 基本的なRAGパイプライン](tutorials/tutorial_01_basic_rag_ja.md)
  - [チュートリアル2: コーパス管理とドキュメント処理](tutorials/tutorial_02_corpus_management_ja.md)
  - [チュートリアル3: クエリエンジンと回答生成](tutorials/tutorial_03_query_engine_ja.md)
  - [チュートリアル4: 高度な正規化とクエリ処理](tutorials/tutorial_04_normalization_ja.md)
  - [チュートリアル5: 企業での利用方法](tutorials/tutorial_05_enterprise_usage_ja.md)
  - [チュートリアル6: 増分文書ローディング](tutorials/tutorial_06_incremental_loading_ja.md)

### 📖 API Reference
Detailed API documentation for each module.

- **English Version**
  - [API Reference Top](api/index.md)
  - [models - Data Model Definitions](api/models.md)
  - [processing - Document Processing Pipeline](api/processing.md)
  - [CorpusManager - Corpus Management](api/corpus_manager.md)
  - [QueryEngine - Query Processing Engine](api/query_engine.md)

- **Japanese Version**
  - [APIリファレンス トップ](api/index_ja.md)
  - [models - データモデル定義](api/models_ja.md)
  - [processing - ドキュメント処理パイプライン](api/processing_ja.md)
  - [CorpusManager - コーパス管理](api/corpus_manager_ja.md)
  - [QueryEngine - クエリ処理エンジン](api/query_engine_ja.md)

### 🏗️ Architecture & Design
System design philosophy and implementation details.

- [Architecture Overview](architecture.md) - **Updated with Unified Architecture (Dec 2024)**
- [Migration Guide: Unified Architecture](migration_guide_unified_architecture.md) - **New: VectorStore Integration**
- [Concept](concept.md)
- [Requirements](requirements.md)
- [Function Specifications](function_spec.md)
- [Backend Interfaces](backend_interfaces.md)
- [Use Case Interfaces](usecase_interfaces.md)

## 🚀 Quick Start

### Installation
```bash
pip install refinire-rag
```

### 30-Second RAG System
```python
from refinire_rag import create_simple_rag

# One-liner enterprise RAG
rag = create_simple_rag("your_documents/")
answer = rag.query("How does this work?")
print(answer)
```

### Production-Ready Setup (Unified Architecture)
```python
from refinire_rag.use_cases import CorpusManager, QueryEngine
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore
from refinire_rag.embedding import OpenAIEmbedder

# Configure storage with unified architecture
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# Configure embedder directly to vector store (no wrapper needed)
embedder = OpenAIEmbedder()
vector_store.set_embedder(embedder)

# Build corpus - VectorStore used directly in pipeline
manager = CorpusManager(doc_store, vector_store)  # Simplified constructor
results = manager.build_corpus(["documents/"])

# Query with VectorStore as integrated retriever
query_engine = QueryEngine(
    document_store=doc_store,
    retriever=vector_store,  # VectorStore implements Retriever interface
    reader=reader
)
result = query_engine.answer("What is our company policy on remote work?")
```

### Enterprise Features
```python
# Incremental updates (90%+ time savings on large corpora)
incremental_loader = IncrementalLoader(document_store, cache_file=".cache.json")
results = incremental_loader.process_incremental(["documents/"])

# Department-level data isolation (Tutorial 5 pattern)
hr_rag = CorpusManager.create_simple_rag(hr_doc_store, hr_vector_store)
sales_rag = CorpusManager.create_simple_rag(sales_doc_store, sales_vector_store)

# Production monitoring
stats = corpus_manager.get_corpus_stats()
```

## 🏆 Framework Comparison

| Feature | LangChain/LlamaIndex | refinire-rag | Advantage |
|---------|---------------------|---------------|-----------|
| **Development Speed** | Complex setup | 5-line setup | **90% faster** |
| **Enterprise Features** | Custom development | Built-in | **Ready out-of-box** |
| **Japanese Processing** | Additional work | Optimized | **Native support** |
| **Incremental Updates** | Manual implementation | Automatic | **90% time savings** |
| **Code Consistency** | Component-specific APIs | Unified interface | **Easier maintenance** |
| **Team Productivity** | Steep learning curve | Single pattern | **Faster onboarding** |

## 📝 Documentation Languages

- 🇬🇧 **English**: Default file names (e.g., `tutorial_01_basic_rag.md`)
- 🇯🇵 **Japanese**: File names with `_ja` suffix (e.g., `tutorial_01_basic_rag_ja.md`)

## 🔗 Related Links

- [Refinire Library](https://github.com/kitfactory/refinire) - Parent workflow framework
- [GitHub Repository](https://github.com/your-org/refinire-rag)
- [Issue Tracker](https://github.com/your-org/refinire-rag/issues)
- [Discussions](https://github.com/your-org/refinire-rag/discussions)

## 📄 License

This documentation is published under the MIT License.

---

**refinire-rag: Where enterprise RAG development becomes effortless.**