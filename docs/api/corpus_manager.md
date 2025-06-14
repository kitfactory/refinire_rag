# CorpusManager - Document Corpus Management

CorpusManager orchestrates the complete document processing pipeline from loading to embedding storage, built on the DocumentPipeline architecture.

## Overview

CorpusManager provides a unified interface for managing document corpora with the following capabilities:

- **DocumentPipeline Integration** - Uses configurable DocumentProcessor chains
- **Incremental Processing** - Efficient updates with IncrementalLoader
- **Flexible Configuration** - Modular component composition
- **Comprehensive Workflow** - Loading → Processing → Chunking → Embedding → Storage

```python
from refinire_rag.application.corpus_manager import CorpusManager
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore
from refinire_rag.embedding import OpenAIEmbedder, OpenAIEmbeddingConfig
from refinire_rag.processing import Normalizer, Chunker, ChunkingConfig

# Initialize storage
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# Configure embedder and set to vector store (統合アーキテクチャ)
embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(model="text-embedding-ada-002"))
vector_store.set_embedder(embedder)  # VectorStoreに直接設定

# Create CorpusManager with integrated architecture
corpus_manager = CorpusManager(
    document_store=doc_store,
    vector_store=vector_store  # VectorStoreを直接使用、ラッパー不要
)
```

## Core Workflow Methods

### process_corpus

Complete corpus processing pipeline.

```python
# Process documents from sources
results = corpus_manager.process_corpus([
    "documents/", 
    "data/manual.pdf",
    "specs/technical_docs/"
])

print(f"Documents loaded: {results['documents_loaded']}")
print(f"Documents processed: {results['documents_processed']}")
print(f"Documents embedded: {results['documents_embedded']}")
print(f"Total time: {results['total_processing_time']:.2f}s")
```

### load_documents

Load documents from various sources.

```python
# Load from mixed sources
documents = corpus_manager.load_documents([
    "/path/to/single_file.txt",
    "/path/to/directory/",
    Path("another/directory")
])

print(f"Loaded {len(documents)} documents")
```

### process_documents

Process documents through the configured pipeline.

```python
# Process with configured processors
processed_docs = corpus_manager.process_documents(documents)

# Includes normalization, chunking, etc. based on configuration
```

### embed_documents

Generate embeddings and store in vector store.

```python
# Generate embeddings
embedded_docs = corpus_manager.embed_documents(processed_docs)

# Documents are automatically stored in vector store
for doc, result in embedded_docs:
    if result and result.success:
        print(f"Embedded {doc.id}: {result.dimension}D vector")
```

## Incremental Processing

CorpusManager integrates with IncrementalLoader for efficient large-scale document management.

```python
from refinire_rag.loaders.incremental_loader import IncrementalLoader

# Setup incremental processing
incremental_loader = IncrementalLoader(
    document_store=corpus_manager._document_store,
    base_loader=corpus_manager._loader,
    cache_file=".corpus_cache.json"
)

# Process only changed files
results = incremental_loader.process_incremental(
    sources=["documents/"],
    force_reload={"important_doc.txt"}
)

# Process through corpus manager pipeline
if results['new'] or results['updated']:
    # Process new/updated documents
    processed = corpus_manager.process_documents(
        results['new'] + results['updated']
    )
    
    # Generate embeddings
    corpus_manager.embed_documents(processed)

print(f"New: {len(results['new'])}")
print(f"Updated: {len(results['updated'])}")
print(f"Skipped: {len(results['skipped'])}")
```

## Configuration Options

### CorpusManagerConfig

Complete configuration for CorpusManager behavior.

```python
from refinire_rag.application.corpus_manager import CorpusManagerConfig
from refinire_rag.models.config import LoadingConfig

config = CorpusManagerConfig(
    # Document loading
    loading_config=LoadingConfig(
        skip_errors=True,
        parallel=True,
        max_workers=4
    ),
    
    # Document processing pipeline
    enable_processing=True,
    processors=[
        DictionaryMaker(dict_config),
        Normalizer(norm_config),
        TokenBasedChunker(chunk_config)
    ],
    
    # Chunking
    enable_chunking=True,
    chunking_config=ChunkingConfig(
        chunk_size=512,
        overlap=50,
        split_by_sentence=True
    ),
    
    # Embedding
    enable_embedding=True,
    embedder=TFIDFEmbedder(embedding_config),
    auto_fit_embedder=True,
    
    # Storage
    document_store=SQLiteDocumentStore("corpus.db"),
    vector_store=InMemoryVectorStore(),
    store_intermediate_results=True,
    
    # Processing options
    batch_size=100,
    parallel_processing=False,
    
    # Error handling
    fail_on_error=False,
    max_errors=10,
    
    # Progress reporting
    enable_progress_reporting=True,
    progress_interval=10
)
```

## Utility Methods

### search_documents

Search documents in the corpus.

```python
# Semantic search using embeddings
results = corpus_manager.search_documents(
    query="RAG implementation best practices",
    limit=10,
    use_semantic=True
)

# Text search fallback
results = corpus_manager.search_documents(
    query="machine learning",
    limit=5,
    use_semantic=False
)
```

### get_corpus_stats

Get comprehensive corpus statistics.

```python
stats = corpus_manager.get_corpus_stats()

print(f"Documents loaded: {stats['documents_loaded']}")
print(f"Documents processed: {stats['documents_processed']}")
print(f"Documents embedded: {stats['documents_embedded']}")
print(f"Total processing time: {stats['total_processing_time']:.2f}s")
print(f"Storage stats: {stats['storage_stats']}")
print(f"Vector stats: {stats['vector_stats']}")
```

### get_document_lineage

Track document processing lineage.

```python
# Get all versions/transformations of a document
lineage = corpus_manager.get_document_lineage("document_123")

for doc in lineage:
    stage = doc.metadata.get("processing_stage", "original")
    print(f"{doc.id}: {stage}")
```

### cleanup

Clean up resources.

```python
# Close connections and clean up
corpus_manager.cleanup()
```

## Integration Examples

### Simple RAG Setup

```python
# Minimal configuration for basic RAG
config = CorpusManagerConfig(
    document_store=SQLiteDocumentStore("simple.db"),
    vector_store=InMemoryVectorStore(),
    embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(min_df=1)),
    processors=[TokenBasedChunker()],
    enable_progress_reporting=True
)

corpus_manager = CorpusManager(config)

# Process documents
results = corpus_manager.process_corpus(["documents/"])
```

### Advanced RAG with Processing

```python
# Full processing pipeline
config = CorpusManagerConfig(
    document_store=SQLiteDocumentStore("advanced.db"),
    vector_store=InMemoryVectorStore(),
    embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
        min_df=2, 
        max_features=10000,
        ngram_range=(1, 2)
    )),
    processors=[
        DictionaryMaker(DictionaryMakerConfig(
            dictionary_file_path="dictionary.md",
            focus_on_technical_terms=True
        )),
        Normalizer(NormalizerConfig(
            dictionary_file_path="dictionary.md",
            whole_word_only=False
        )),
        TokenBasedChunker(ChunkingConfig(
            chunk_size=512,
            overlap=50,
            split_by_sentence=True
        ))
    ],
    batch_size=50,
    fail_on_error=False
)

corpus_manager = CorpusManager(config)
```

## Error Handling

```python
from refinire_rag.exceptions import RefinireRAGError

try:
    results = corpus_manager.process_corpus(file_paths)
    print(f"✅ Success: {results['documents_loaded']} documents processed")
except RefinireRAGError as e:
    print(f"❌ Processing error: {e}")
    # Graceful degradation or retry logic
except Exception as e:
    print(f"💥 Unexpected error: {e}")
```

## Best Practices

1. **Start Simple**: Begin with basic configuration and add complexity incrementally
2. **Monitor Performance**: Use progress reporting and statistics for optimization
3. **Handle Errors Gracefully**: Configure appropriate error handling for production
4. **Use Incremental Loading**: For large or frequently updated document collections
5. **Optimize Embeddings**: Choose appropriate embedder and configuration for your use case

```python
# Production-ready example with error handling
def build_production_corpus(file_paths: List[str]) -> Dict[str, Any]:
    config = CorpusManagerConfig(
        fail_on_error=False,
        max_errors=10,
        enable_progress_reporting=True,
        batch_size=100
    )
    
    corpus_manager = CorpusManager(config)
    
    try:
        results = corpus_manager.process_corpus(file_paths)
        
        # Log successful processing
        logger.info(f"Corpus built successfully: {results['documents_loaded']} documents")
        
        return results
        
    except Exception as e:
        logger.error(f"Corpus building failed: {e}")
        raise
    
    finally:
        corpus_manager.cleanup()
```