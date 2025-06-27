# CorpusManager - Document Corpus Management

CorpusManager provides comprehensive document corpus construction and management with plugin-based multi-retriever support.

## Overview

CorpusManager provides core operations for document corpus management with plugin integration:

- **Document Import** - Import original documents from directories with incremental loading
- **Corpus Rebuild** - Rebuild corpus from existing original documents using knowledge artifacts
- **Multi-Retriever Support** - Support for multiple retrievers (VectorStore, KeywordSearch, etc.)
- **Plugin Integration** - Automatic plugin discovery and environment-based configuration
- **Corpus Management** - Clear and manage corpus state across all retrievers

```python
from refinire_rag.application import CorpusManager

# Method 1: Create with environment variables (recommended)
# Set environment: REFINIRE_RAG_RETRIEVERS=chroma,bm25s
corpus_manager = CorpusManager()

# Method 2: Manual initialization with specific retrievers
from refinire_rag.storage import SQLiteStore

document_store = SQLiteStore("corpus.db")
retrievers = [vector_store, keyword_store]  # Multiple retrievers

corpus_manager = CorpusManager(
    document_store=document_store,
    retrievers=retrievers
)
```

## Public API Methods

### import_original_documents

Import original documents from a directory with incremental loading.

```python
# Basic document import
stats = corpus_manager.import_original_documents(
    corpus_name="product_docs",
    directory="/path/to/documents"
)

print(f"Files processed: {stats.total_files_processed}")
print(f"Documents created: {stats.total_documents_created}")
```

```python
# Advanced import with options
stats = corpus_manager.import_original_documents(
    corpus_name="knowledge_base",
    directory="/docs",
    glob="**/*.{md,txt,pdf}",  # Filter file types
    use_multithreading=True,
    force_reload=False,  # Use incremental loading
    additional_metadata={"project": "refinire"},
    create_dictionary=True,  # Create domain dictionary
    create_knowledge_graph=True,  # Create knowledge graph
    dictionary_output_dir="./dictionaries",
    graph_output_dir="./graphs"
)
```

### rebuild_corpus_from_original

Rebuild corpus from existing original documents using knowledge artifacts.

```python
# Rebuild with existing dictionary
stats = corpus_manager.rebuild_corpus_from_original(
    corpus_name="knowledge_base",
    use_dictionary=True,  # Use existing dictionary for normalization
    dictionary_file_path="./dictionaries/knowledge_base_dictionary.md"
)

print(f"Documents processed: {stats.total_documents_created}")
print(f"Chunks created: {stats.total_chunks_created}")
```

```python
# Rebuild with both dictionary and knowledge graph
stats = corpus_manager.rebuild_corpus_from_original(
    corpus_name="knowledge_base",
    use_dictionary=True,
    use_knowledge_graph=True,
    dictionary_file_path="./dictionaries/knowledge_base_dictionary.md",
    graph_file_path="./graphs/knowledge_base_knowledge_graph.md",
    additional_metadata={"version": "2.0"}
)
```

### clear_corpus

Clear all documents from the corpus.

```python
# Clear all documents from both document and vector stores
corpus_manager.clear_corpus()
print("Corpus cleared successfully")
```

## Method Parameters Reference

### import_original_documents Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_name` | `str` | Required | Name of the corpus (used in metadata and output filenames) |
| `directory` | `str` | Required | Directory path to import from |
| `glob` | `str` | `"**/*"` | Glob pattern to match files |
| `use_multithreading` | `bool` | `True` | Whether to use multithreading for file processing |
| `force_reload` | `bool` | `False` | Force reload all files ignoring incremental cache |
| `additional_metadata` | `Optional[Dict[str, Any]]` | `None` | Additional metadata to add to all imported documents |
| `tracking_file_path` | `Optional[str]` | `None` | Path to store file tracking data for incremental loading |
| `create_dictionary` | `bool` | `False` | Whether to create domain dictionary after import |
| `create_knowledge_graph` | `bool` | `False` | Whether to create knowledge graph after import |
| `dictionary_output_dir` | `Optional[str]` | `None` | Directory to save dictionary file |
| `graph_output_dir` | `Optional[str]` | `None` | Directory to save graph file |

### rebuild_corpus_from_original Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus_name` | `str` | Required | Name of the corpus for metadata |
| `use_dictionary` | `bool` | `True` | Whether to use existing dictionary for normalization |
| `use_knowledge_graph` | `bool` | `False` | Whether to use existing knowledge graph for normalization |
| `dictionary_file_path` | `Optional[str]` | `None` | Path to existing dictionary file to use |
| `graph_file_path` | `Optional[str]` | `None` | Path to existing knowledge graph file to use |
| `additional_metadata` | `Optional[Dict[str, Any]]` | `None` | Additional metadata to add during rebuild |
| `stage_configs` | `Optional[Dict[str, Any]]` | `None` | Configuration for each processing stage |

## File Naming Convention

CorpusManager follows a consistent file naming convention for tracking and knowledge artifacts:

- **Tracking file**: `{corpus_name}_track.json`
- **Dictionary file**: `{corpus_name}_dictionary.md`
- **Knowledge graph file**: `{corpus_name}_knowledge_graph.md`

### Constructor with Environment Variables

Create CorpusManager with environment variable support.

```python
# Set environment variables
import os
os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s"
os.environ["REFINIRE_RAG_DOCUMENT_STORES"] = "sqlite"

# Create from environment
corpus_manager = CorpusManager()
```

### get_corpus_info

Get comprehensive information about the corpus manager configuration.

```python
info = corpus_manager.get_corpus_info()

print(f"Document Store: {info['document_store']['type']}")
print("Retrievers:")
for retriever in info['retrievers']:
    print(f"  - {retriever['type']}: {retriever['capabilities']}")
print(f"Total documents: {info['stats']['total_documents_created']}")
```

### get_retrievers_by_type

Get retrievers filtered by type.

```python
# Get vector-based retrievers
vector_retrievers = corpus_manager.get_retrievers_by_type("vector")

# Get keyword-based retrievers  
keyword_retrievers = corpus_manager.get_retrievers_by_type("keyword")
```

### add_retriever

Add a new retriever to the corpus manager.

```python
from refinire_rag.registry import PluginRegistry

# Create and add new retriever
new_retriever = PluginRegistry.create_plugin('retrievers', 'faiss')
corpus_manager.add_retriever(new_retriever)
```

### remove_retriever

Remove a retriever by index.

```python
# Remove retriever at index 1
success = corpus_manager.remove_retriever(1)
print(f"Removal successful: {success}")
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REFINIRE_DIR` | `"./refinire"` | Base directory for Refinire files |
| `REFINIRE_RAG_DOCUMENT_STORES` | `"sqlite"` | Document store plugins to use |
| `REFINIRE_RAG_RETRIEVERS` | `"inmemory_vector"` | Retriever plugins to use (comma-separated) |

## Multi-Retriever Support

CorpusManager supports multiple retrievers simultaneously:

```python
# Environment-based configuration
os.environ["REFINIRE_RAG_RETRIEVERS"] = "chroma,bm25s,faiss"

corpus_manager = CorpusManager()

# Check configured retrievers
info = corpus_manager.get_corpus_info()
print(f"Active retrievers: {len(info['retrievers'])}")

# Import documents (processed by all retrievers)
stats = corpus_manager.import_original_documents(
    corpus_name="multi_retriever_corpus",
    directory="./documents"
)

# Each retriever will process and store the documents according to its capabilities
```

## Plugin Integration

CorpusManager automatically integrates with the plugin system:

```python
# List available retriever plugins
from refinire_rag.registry import PluginRegistry
available = PluginRegistry.list_available_plugins('retrievers')
print(f"Available retriever plugins: {available}")

# Create corpus manager with specific plugins
os.environ["REFINIRE_RAG_RETRIEVERS"] = ",".join(available)
corpus_manager = CorpusManager()
```

## CorpusStats Return Object

The `import_original_documents` and `rebuild_corpus_from_original` methods return a `CorpusStats` object:

```python
@dataclass
class CorpusStats:
    total_files_processed: int = 0
    total_documents_created: int = 0
    total_chunks_created: int = 0
    total_processing_time: float = 0.0
    pipeline_stages_executed: int = 0
    documents_by_stage: Dict[str, int] = None
    errors_encountered: int = 0
```

### Usage Example

```python
stats = corpus_manager.import_original_documents(
    corpus_name="docs",
    directory="/path/to/docs"
)

print(f"Files processed: {stats.total_files_processed}")
print(f"Documents created: {stats.total_documents_created}")
print(f"Processing time: {stats.total_processing_time:.2f}s")
print(f"Errors: {stats.errors_encountered}")
```

## Usage Patterns

### Basic Document Import

```python
from refinire_rag.application import CorpusManager
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore

# Initialize CorpusManager
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()
corpus_manager = CorpusManager(doc_store, vector_store)

# Import documents
stats = corpus_manager.import_original_documents(
    corpus_name="knowledge_base",
    directory="/path/to/documents"
)

print(f"Imported {stats.total_documents_created} documents")
```

### Advanced Workflow with Knowledge Artifacts

```python
# Step 1: Import with knowledge artifact creation
stats = corpus_manager.import_original_documents(
    corpus_name="tech_docs",
    directory="/docs",
    create_dictionary=True,
    create_knowledge_graph=True
)

# Step 2: Later rebuild with improvements
rebuild_stats = corpus_manager.rebuild_corpus_from_original(
    corpus_name="tech_docs",
    use_dictionary=True,
    use_knowledge_graph=True
)

print(f"Rebuilt {rebuild_stats.total_documents_created} documents")
```

## Best Practices

1. **Use Incremental Loading**: Set `force_reload=False` for large document collections to enable incremental processing
2. **Create Knowledge Artifacts**: Use `create_dictionary=True` and `create_knowledge_graph=True` for improved corpus quality
3. **Monitor Progress**: Check `CorpusStats` for processing metrics and error counts
4. **Handle File Patterns**: Use glob patterns to filter specific file types
5. **Organize Output**: Specify custom directories for dictionaries and graphs

## Complete Workflow Example

```python
from refinire_rag.application import CorpusManager
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore

def build_knowledge_base():
    # Initialize storage
    doc_store = SQLiteDocumentStore("knowledge.db")
    vector_store = InMemoryVectorStore()
    corpus_manager = CorpusManager(doc_store, vector_store)
    
    try:
        # Initial import with knowledge artifacts
        stats = corpus_manager.import_original_documents(
            corpus_name="company_docs",
            directory="/company/documents",
            glob="**/*.{md,txt,pdf}",
            create_dictionary=True,
            create_knowledge_graph=True,
            additional_metadata={"version": "1.0"}
        )
        
        print(f"✅ Import complete: {stats.total_documents_created} documents")
        
        # Later: rebuild with improvements
        if stats.errors_encountered == 0:
            rebuild_stats = corpus_manager.rebuild_corpus_from_original(
                corpus_name="company_docs",
                use_dictionary=True,
                use_knowledge_graph=True
            )
            print(f"✅ Rebuild complete: {rebuild_stats.total_documents_created} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Run the workflow
success = build_knowledge_base()
```