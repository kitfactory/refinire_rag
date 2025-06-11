# processing - Document Processing Pipeline

Unified document processing architecture where all document processing components inherit from the `DocumentProcessor` base class and implement a single `process(document) -> List[Document]` interface.

## DocumentProcessor Base Class

All document processing functions inherit from this unified base class.

```python
from refinire_rag.processing.document_processor import DocumentProcessor

class DocumentProcessor(ABC):
    """Base interface for all document processing components"""
    
    @abstractmethod
    def process(self, document: Document, config: Optional[DocumentProcessorConfig] = None) -> List[Document]:
        """Process a document and return processed documents"""
        pass
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[DocumentProcessorConfig]:
        """Get the configuration class for this processor"""
        pass
```

### Unified Architecture Benefits

- **Consistent Interface**: All processors use the same `process()` method
- **Pipeline Composition**: Easy chaining of multiple processors
- **Configuration Management**: Standardized configuration handling with dataclasses
- **Error Handling**: Unified error handling and logging
- **Testing**: Consistent testing patterns across all processors

### Core Implementation Classes

#### Document Loading & Transformation
- `UniversalLoader` - File loading with format detection
- `Normalizer` - Dictionary-based normalization  
- `TokenBasedChunker` - Document chunking
- `DictionaryMaker` - Term extraction and dictionary updates
- `GraphBuilder` - Knowledge graph construction
- `VectorStoreProcessor` - Embedding generation and storage

#### Quality & Evaluation
- `TestSuite` - Evaluation pipeline execution
- `Evaluator` - Metrics aggregation
- `ContradictionDetector` - Conflict detection
- `InsightReporter` - Analysis reporting

### Incremental Processing

The `IncrementalLoader` provides efficient handling of large document collections:

```python
from refinire_rag.loaders.incremental_loader import IncrementalLoader

# Setup incremental loader
incremental_loader = IncrementalLoader(
    document_store=document_store,
    cache_file=".cache.json"
)

# Process only changed/new files
results = incremental_loader.process_incremental(
    sources=["documents/"],
    force_reload={"specific_file.txt"}
)
```

## Normalizer

Performs dictionary-based expression normalization.

```python
from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig

# Configuration
config = NormalizerConfig(
    dictionary_file_path="dictionary.md",
    normalize_variations=True,
    expand_abbreviations=True,
    whole_word_only=False,  # For Japanese support
    case_sensitive=False
)

# Create normalizer
normalizer = Normalizer(config)

# Example usage
doc = Document(id="doc1", content="About retrieval-enhanced generation")
normalized_docs = normalizer.process(doc)
# → "About retrieval-augmented generation"
```

### NormalizerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dictionary_file_path | str | Required | Dictionary file path |
| normalize_variations | bool | True | Normalize expression variations |
| expand_abbreviations | bool | True | Expand abbreviations |
| whole_word_only | bool | False | Match only on word boundaries |
| case_sensitive | bool | False | Case-sensitive matching |

## Chunker

Splits documents into searchable units.

```python
from refinire_rag.processing.chunker import Chunker, ChunkingConfig

# Configuration
config = ChunkingConfig(
    chunk_size=500,
    overlap=50,
    split_by_sentence=True,
    min_chunk_size=100
)

# Create chunker
chunker = Chunker(config)

# Example usage
doc = Document(id="doc1", content="Long document text...")
chunks = chunker.process(doc)
# → [Chunk1, Chunk2, ...]
```

### ChunkingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| chunk_size | int | 500 | Chunk size (in tokens) |
| overlap | int | 50 | Overlap between chunks |
| split_by_sentence | bool | True | Split on sentence boundaries |
| min_chunk_size | int | 100 | Minimum chunk size |

## DictionaryMaker

Creates and updates term dictionaries from documents.

```python
from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig

# Configuration
config = DictionaryMakerConfig(
    dictionary_file_path="dictionary.md",
    focus_on_technical_terms=True,
    extract_abbreviations=True,
    min_term_frequency=2
)

# Create dictionary maker
dict_maker = DictionaryMaker(config)

# Example usage
doc = Document(id="doc1", content="RAG (Retrieval-Augmented Generation) is...")
dict_maker.process(doc)
# → Dictionary file is updated
```

### DictionaryMakerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| dictionary_file_path | str | Required | Dictionary file path |
| focus_on_technical_terms | bool | True | Focus on technical terms |
| extract_abbreviations | bool | True | Extract abbreviations |
| min_term_frequency | int | 1 | Minimum term frequency |

## DocumentPipeline

Chains multiple processors to build a pipeline.

```python
from refinire_rag.processing.document_pipeline import DocumentPipeline

# Build pipeline
pipeline = DocumentPipeline([
    DictionaryMaker(dict_config),
    Normalizer(norm_config),
    Chunker(chunk_config)
])

# Example usage
doc = Document(id="doc1", content="About retrieval-enhanced generation...")
results = pipeline.process(doc)
# → Dictionary creation → Normalization → Chunking
```

### Methods

- `process(document: Document) -> List[Document]` - Process document
- `add_processor(processor: DocumentProcessor)` - Add processor
- `remove_processor(index: int)` - Remove processor

## VectorStoreProcessor

Vectorizes and stores documents.

```python
from refinire_rag.processing.vector_store_processor import VectorStoreProcessor

# Configuration
config = VectorStoreProcessorConfig(
    embedder_type="tfidf",
    embedder_config={"min_df": 1, "max_df": 1.0},
    batch_size=100
)

# Create processor
processor = VectorStoreProcessor(vector_store, config)

# Example usage
doc = Document(id="doc1", content="RAG is a retrieval-augmented generation technology")
processor.process(doc)
# → Vectorized and stored
```

## GraphBuilder

Builds knowledge graphs from documents.

```python
from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig

# Configuration
config = GraphBuilderConfig(
    graph_file_path="knowledge_graph.md",
    extract_entities=True,
    extract_relations=True,
    min_confidence=0.7
)

# Create graph builder
graph_builder = GraphBuilder(config)

# Example usage
doc = Document(id="doc1", content="RAG combines LLMs with external knowledge")
graph_builder.process(doc)
# → Graph file is updated
```

## Usage Patterns

### 1. Simple RAG Pipeline

```python
pipeline = DocumentPipeline([
    Chunker(chunk_config),
    VectorStoreProcessor(vector_store, vector_config)
])
```

### 2. Semantic RAG Pipeline

```python
pipeline = DocumentPipeline([
    DictionaryMaker(dict_config),
    Normalizer(norm_config),
    Chunker(chunk_config),
    VectorStoreProcessor(vector_store, vector_config)
])
```

### 3. Knowledge RAG Pipeline

```python
pipeline = DocumentPipeline([
    DictionaryMaker(dict_config),
    GraphBuilder(graph_config),
    Normalizer(norm_config),
    Chunker(chunk_config),
    VectorStoreProcessor(vector_store, vector_config)
])
```