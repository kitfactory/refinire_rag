# refinire-rag API Reference

API reference documentation for the refinire-rag library.

## Package Structure

### Core Modules

- [models](models_en.md) - Data model definitions
- [processing](processing_en.md) - Document processing pipeline
- [storage](storage_en.md) - Storage interfaces
- [embedding](embedding_en.md) - Embedding generation
- [retrieval](retrieval_en.md) - Search and answer generation
- [loaders](loaders_en.md) - File loaders

### Use Case Classes

- [CorpusManager](corpus_manager_en.md) - Corpus management
- [QueryEngine](query_engine_en.md) - Query processing engine
- [QualityLab](quality_lab_en.md) - Quality evaluation (planned)

## Quick Reference

### Basic Usage

```python
from refinire_rag.use_cases.corpus_manager_new import CorpusManager
from refinire_rag.use_cases.query_engine import QueryEngine
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore

# Initialize storage
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# Create Simple RAG
corpus_manager = CorpusManager.create_simple_rag(doc_store, vector_store)

# Build corpus
stats = corpus_manager.build_corpus(["document1.txt", "document2.txt"])

# Create query engine
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader
)

# Question answering
result = query_engine.answer("What is RAG?")
```

### Advanced Usage

```python
# Build custom pipeline
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.processing.normalizer import Normalizer
from refinire_rag.processing.chunker import Chunker

pipeline = DocumentPipeline([
    Normalizer(config),
    Chunker(config)
])

# Build corpus with stage selection
stats = corpus_manager.build_corpus(
    file_paths=files,
    stages=["load", "dictionary", "normalize", "chunk", "vector"],
    stage_configs={
        "normalizer_config": NormalizerConfig(...),
        "chunker_config": ChunkingConfig(...)
    }
)
```

## Design Principles

1. **DocumentProcessor Pattern**: All processing implements the DocumentProcessor interface
2. **Pipeline Composition**: Processing can be combined as pipelines
3. **Externalized Configuration**: All settings managed by Config classes
4. **Type Safety**: Type definitions using Pydantic

## Version Information

- Current Version: 0.1.0
- Python Requirements: 3.10 or higher