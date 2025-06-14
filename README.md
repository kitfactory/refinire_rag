# refinire-rag

The refined RAG framework that makes enterprise-grade document processing effortless.

## 🌟 Why refinire-rag?

Traditional RAG frameworks are powerful but complex. refinire-rag refines the development experience with radical simplicity and enterprise-grade productivity.

**[→ Why refinire-rag? The Complete Story](docs/why_refinire_rag.md)** | **[→ なぜrefinire-rag？完全版](docs/why_refinire_rag_ja.md)**

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

## Overview

refinire-rag provides RAG (Retrieval-Augmented Generation) functionality as a sub-package of the Refinire library. The library follows a unified DocumentProcessor architecture with dependency injection for maximum flexibility and enterprise-grade capabilities.

## Architecture

### Application Classes (Refinire Steps)
- **CorpusManager**: Document loading, normalization, chunking, embedding generation, and storage
- **QueryEngine**: Document retrieval, re-ranking, and answer generation (inherits from Refinire Step)
- **QualityLab**: Evaluation data creation, automatic RAG evaluation, conflict detection, and report generation

### DocumentProcessor Unified Architecture
All document processing components inherit from a single base class with consistent interface:

#### Document Processing Pipeline
- **UniversalLoader**: Multi-format document loading with parallel processing
- **Normalizer**: Dictionary-based term normalization and linguistic optimization
- **Chunker**: Intelligent document chunking for optimal embedding
- **DictionaryMaker**: Term and abbreviation extraction with LLM integration
- **GraphBuilder**: Knowledge graph construction and relationship extraction
- **VectorStore**: Integrated embedding generation, vector storage, and retrieval (DocumentProcessor + Indexer + Retriever)

#### Quality & Evaluation
- **TestSuite**: Comprehensive evaluation pipeline execution
- **Evaluator**: Multi-metric aggregation and analysis
- **ContradictionDetector**: Automated conflict detection with NLI
- **InsightReporter**: Intelligent threshold-based reporting

### Query Processing Components
- **Retriever**: Semantic and hybrid document search
- **Reranker**: Context-aware result re-ranking
- **Reader**: LLM-powered answer generation

## Architecture Highlights

### DocumentProcessor Unified Architecture
All document processing components inherit from a single base class with consistent `process(document) -> List[Document]` interface:

```python
# Every processor follows the same pattern (統合アーキテクチャ)
normalizer = Normalizer(config)
chunker = Chunker(config)
vector_store = InMemoryVectorStore()  # VectorStore直接使用
vector_store.set_embedder(embedder)   # 埋め込み設定

# Chain them together - VectorStoreを直接パイプラインで使用
pipeline = DocumentPipeline([normalizer, chunker, vector_store])
results = pipeline.process_document(document)
```

### Incremental Processing
Efficient handling of large document collections with automatic change detection:

```python
# Only process new/changed files
incremental_loader = IncrementalLoader(document_store, cache_file=".cache.json")
results = incremental_loader.process_incremental(["documents/"])
# Skips unchanged files, processes only what's needed
```

### Enterprise-Ready Features
- **Multi-format document loading** with parallel processing ([detailed guide](docs/loader_implementation.md))
- **Japanese text optimization** with linguistic normalization
- **Department-level data isolation** patterns
- **Comprehensive monitoring** and error handling
- **Production deployment** ready configurations

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

### Production-Ready Setup
```python
from refinire_rag.application import CorpusManager, QueryEngine
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore

# Configure storage
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# Build corpus with incremental processing
manager = CorpusManager.create_simple_rag(doc_store, vector_store)
results = manager.process_corpus(["documents/"])

# Query with confidence
query_engine = QueryEngine(retriever, reranker, reader)
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

## 📚 Documentation

### 🎯 Tutorials
Learn how to build RAG systems step by step - from simple prototypes to enterprise deployment.

- [Tutorial Overview](docs/tutorials/tutorial_overview.md)
- [Tutorial 1: Basic RAG Pipeline](docs/tutorials/tutorial_01_basic_rag.md)
- [Tutorial 6: Incremental Document Loading](docs/tutorials/tutorial_06_incremental_loading.md)

### 📖 API Reference
Detailed API documentation for each module.

- [API Reference](docs/api/index.md)
- [Document Processing Pipeline](docs/api/processing.md)
- [Corpus Management](docs/api/corpus_manager.md)
- [Query Engine](docs/api/query_engine.md)

### 🏗️ Architecture & Design
System design philosophy and implementation details.

- [Architecture Overview](docs/architecture.md)
- [Design Philosophy](docs/concept.md)
- [Loader Implementation](docs/loader_implementation.md) - Detailed document loading guide
- [Requirements](docs/requirements.md)
- [Function Specifications](docs/function_spec.md)

## Key Features

### Flexible Document Model
- Minimal required metadata (4 fields)
- Completely flexible additional metadata
- Database-friendly design for search and lineage tracking

### Parallel Processing
- Concurrent document loading with ThreadPoolExecutor/ProcessPoolExecutor
- Async support for high-throughput scenarios
- Progress tracking and error recovery

### Extension-Based Architecture
- Universal loader delegates to specialized loaders by file extension
- Easy registration of custom loaders
- Subpackage support for advanced processing (Docling, Unstructured, etc.)

### Metadata Enrichment
- Path-based metadata generation with pattern matching
- Automatic file type detection and classification
- Custom metadata generators for domain-specific requirements

### Error Handling
- Comprehensive exception hierarchy
- Configurable error handling (fail-fast or skip-errors)
- Detailed error reporting and logging

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/        # Unit tests
python -m pytest tests/integration/ # Integration tests

# Run examples
python examples/simple_rag_test.py
```

### Project Structure
```
refinire-rag/
├── src/refinire_rag/          # Main package
│   ├── models/                # Data models
│   ├── loaders/              # Document loading system
│   ├── processing/           # Document processing pipeline
│   ├── storage/              # Storage systems
│   ├── application/            # Use case classes
│   └── retrieval/            # Search and answer generation
├── docs/                     # Architecture documentation
├── examples/                 # Usage examples
└── tests/                    # Test suite
    ├── unit/                 # Unit tests
    └── integration/          # Integration tests
```

## Contributing

This project follows the architecture defined in the documentation. When implementing new features:

1. Follow the DocumentProcessor interface patterns
2. Maintain dependency injection for testability
3. Add comprehensive error handling and logging
4. Include usage examples and tests
5. Update documentation for new features

## 📝 Documentation Languages

- 🇬🇧 **English**: Default file names (e.g., `tutorial_01_basic_rag.md`)
- 🇯🇵 **Japanese**: File names with `_ja` suffix (e.g., `tutorial_01_basic_rag_ja.md`)

## 🔗 Related Links

- [Refinire Library](https://github.com/kitfactory/refinire) - Parent workflow framework
- [GitHub Repository](https://github.com/your-org/refinire-rag)
- [Issue Tracker](https://github.com/your-org/refinire-rag/issues)
- [Discussions](https://github.com/your-org/refinire-rag/discussions)

## License

[License information to be added]

---

**refinire-rag: Where enterprise RAG development becomes effortless.**