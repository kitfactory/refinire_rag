# QueryEngine - Query Processing Engine

Use case class that manages query processing and answer generation.

## Overview

QueryEngine receives user queries and manages the following integrated processing:

1. **Query Normalization** - Unifying expression variations
2. **Document Retrieval** - Searching relevant documents
3. **Re-ranking** - Re-evaluating search results
4. **Answer Generation** - Generating answers using LLM

```python
from refinire_rag.use_cases.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader

# Create components
retriever = SimpleRetriever(vector_store, embedder)
reranker = SimpleReranker()
reader = SimpleReader()

# Create QueryEngine
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader,
    reranker=reranker,
    config=QueryEngineConfig()
)
```

## QueryEngineConfig

Configuration class that controls QueryEngine behavior.

```python
class QueryEngineConfig(BaseModel):
    enable_query_normalization: bool = True      # Enable query normalization
    auto_detect_corpus_state: bool = True       # Auto-detect corpus state
    retriever_top_k: int = 10                   # Top K search results
    reranker_top_k: int = 5                     # Top K after re-ranking
    include_sources: bool = True                # Include source information
    include_confidence: bool = True             # Include confidence score
    include_processing_metadata: bool = False   # Include processing metadata
    query_timeout: float = 30.0                 # Query timeout (seconds)
```

### Configuration Examples

```python
# Basic configuration
config = QueryEngineConfig()

# Custom configuration
config = QueryEngineConfig(
    enable_query_normalization=True,
    retriever_top_k=20,
    reranker_top_k=3,
    include_processing_metadata=True
)

# Apply to QueryEngine
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader,
    reranker=reranker,
    config=config
)
```

## Main Methods

### answer

Generates an answer to a query.

```python
def answer(self, query: str) -> QueryResult:
    """
    Generate answer to query
    
    Args:
        query: User query
        
    Returns:
        QueryResult: Answer result
    """
    
# Example usage
result = query_engine.answer("What is RAG?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Number of sources: {len(result.sources)}")
```

### search

Executes document search only (without answer generation).

```python
def search(self, query: str) -> List[SearchResult]:
    """
    Execute document search
    
    Args:
        query: Search query
        
    Returns:
        List[SearchResult]: Search results list
    """
    
# Example usage
results = query_engine.search("vector search")
for result in results:
    print(f"Document ID: {result.document_id}, Score: {result.score}")
```

### get_engine_stats

Gets engine statistics.

```python
stats = query_engine.get_engine_stats()
print(f"Queries processed: {stats['queries_processed']}")
print(f"Average response time: {stats['average_response_time']}")
print(f"Normalization rate: {stats['normalization_rate']}")
```

## Query Normalization

When normalization is applied to the corpus, queries are automatically normalized.

```python
# Configure normalization dictionary
normalizer_config = NormalizerConfig(
    dictionary_file_path="dictionary.md",
    normalize_variations=True,
    whole_word_only=False
)
query_engine.normalizer = Normalizer(normalizer_config)

# Query processing
result = query_engine.answer("Tell me about retrieval-enhanced generation")
# → Internally normalized to "Tell me about retrieval-augmented generation"

print(f"Original query: {result.query}")
print(f"Normalized: {result.normalized_query}")
```

## QueryResult

Model representing query processing results.

```python
class QueryResult:
    query: str                    # Original query
    answer: str                   # Generated answer
    sources: List[SearchResult]   # Referenced sources
    confidence: float             # Answer confidence (0.0-1.0)
    metadata: Dict[str, Any]      # Metadata
    processing_time: float        # Processing time (seconds)
    normalized_query: str         # Normalized query
```

### Using Results

```python
result = query_engine.answer("What are the advantages of RAG?")

# Basic information
print(f"Question: {result.query}")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.2%}")

# Source information
print("\nReferenced sources:")
for i, source in enumerate(result.sources[:3], 1):
    print(f"{i}. {source.metadata.get('title', 'Unknown')}")
    print(f"   Score: {source.score:.3f}")
    print(f"   Content: {source.content[:100]}...")

# Metadata
if result.metadata.get('query_normalized'):
    print(f"\nQuery normalized: {result.normalized_query}")
print(f"Processing time: {result.processing_time:.3f} seconds")
```

## Component Customization

### Custom Retriever

```python
from refinire_rag.retrieval.base import Retriever

class CustomRetriever(Retriever):
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Custom search logic
        pass

# Usage
custom_retriever = CustomRetriever()
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=custom_retriever,
    reader=reader
)
```

### Custom Reader

```python
from refinire_rag.retrieval.base import Reader

class CustomReader(Reader):
    def generate_answer(
        self, 
        query: str, 
        contexts: List[str]
    ) -> str:
        # Custom answer generation logic
        pass
```

## Error Handling

```python
try:
    result = query_engine.answer(query)
except TimeoutError:
    print("Query processing timed out")
except RetrievalError as e:
    print(f"Retrieval error: {e}")
except GenerationError as e:
    print(f"Generation error: {e}")
```

## Performance Optimization

### Caching

```python
# QueryEngine with cache
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader,
    cache_enabled=True,
    cache_ttl=3600  # 1 hour
)
```

### Batch Processing

```python
# Batch processing multiple queries
queries = ["Question 1", "Question 2", "Question 3"]
results = query_engine.batch_answer(queries)
```

## Best Practices

1. **Appropriate top_k settings**: Balance between accuracy and performance
2. **Utilize normalization**: Especially effective for Japanese queries
3. **Timeout settings**: Prevent long processing times
4. **Error handling**: Improve user experience

```python
# Recommended settings example
config = QueryEngineConfig(
    enable_query_normalization=True,
    retriever_top_k=15,    # Get sufficient candidates
    reranker_top_k=3,      # Finally narrow down to 3
    include_sources=True,   # For debugging and reliability
    query_timeout=10.0      # 10 second timeout
)
```