# models - Data Model Definitions

Basic data model definitions used in refinire-rag.

## Document

The basic model representing a document.

```python
from refinire_rag.models.document import Document

class Document(BaseModel):
    """Document model"""
    
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
```

### Attributes

- `id` (str): Unique identifier for the document
- `content` (str): Document text content
- `metadata` (Dict[str, Any]): Metadata (title, author, category, etc.)
- `created_at` (Optional[datetime]): Creation timestamp
- `updated_at` (Optional[datetime]): Update timestamp

### Example Usage

```python
# Create a document
doc = Document(
    id="doc1",
    content="RAG is a retrieval-augmented generation technology.",
    metadata={
        "title": "RAG Overview",
        "category": "Technology",
        "processing_stage": "original"
    }
)

# Update metadata
doc.metadata["tags"] = ["AI", "Search", "Generation"]
```

## Chunk

Model representing a chunk (document fragment).

```python
from refinire_rag.models.chunk import Chunk

class Chunk(BaseModel):
    """Chunk model"""
    
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
```

### Attributes

- `id` (str): Unique identifier for the chunk
- `document_id` (str): ID of the source document
- `content` (str): Chunk text content
- `chunk_index` (int): Order of chunk within document
- `metadata` (Dict[str, Any]): Metadata
- `start_char` (Optional[int]): Starting character position in source document
- `end_char` (Optional[int]): Ending character position in source document

### Example Usage

```python
# Create a chunk
chunk = Chunk(
    id="chunk1",
    document_id="doc1",
    content="RAG is a retrieval-augmented generation technology.",
    chunk_index=0,
    start_char=0,
    end_char=50,
    metadata={"overlap": True}
)
```

## EmbeddingResult

Model representing embedding vector results.

```python
from refinire_rag.models.embedding import EmbeddingResult

class EmbeddingResult(BaseModel):
    """Embedding result model"""
    
    text: str
    vector: np.ndarray
    model_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Attributes

- `text` (str): Text that was embedded
- `vector` (np.ndarray): Embedding vector
- `model_name` (str): Name of the model used
- `metadata` (Dict[str, Any]): Additional metadata

### Example Usage

```python
# Create embedding result
result = EmbeddingResult(
    text="RAG is a retrieval-augmented generation technology.",
    vector=np.array([0.1, 0.2, 0.3, ...]),
    model_name="tfidf",
    metadata={"dimension": 768}
)
```

## QueryResult

Model representing query processing results.

```python
from refinire_rag.models.query import QueryResult

class QueryResult(BaseModel):
    """Query result model"""
    
    query: str
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time: Optional[float] = None
    normalized_query: Optional[str] = None
```

### Attributes

- `query` (str): Original query
- `answer` (str): Generated answer
- `sources` (List[SearchResult]): Referenced sources
- `confidence` (float): Answer confidence (0.0-1.0)
- `metadata` (Dict[str, Any]): Processing metadata
- `processing_time` (Optional[float]): Processing time in seconds
- `normalized_query` (Optional[str]): Normalized query

### Example Usage

```python
# Create query result
result = QueryResult(
    query="What is RAG?",
    answer="RAG is a retrieval-augmented generation technology that combines LLMs with external knowledge.",
    sources=[search_result1, search_result2],
    confidence=0.85,
    processing_time=0.234,
    normalized_query="What is retrieval-augmented generation?"
)
```

## SearchResult

Model representing search results.

```python
from refinire_rag.models.search import SearchResult

class SearchResult(BaseModel):
    """Search result model"""
    
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None
```

### Attributes

- `document_id` (str): Document ID
- `content` (str): Search result content
- `score` (float): Relevance score
- `metadata` (Dict[str, Any]): Metadata
- `chunk_id` (Optional[str]): Chunk ID (for chunk-based search)

## ProcessingStats

Model representing processing statistics.

```python
from refinire_rag.models.stats import ProcessingStats

class ProcessingStats(BaseModel):
    """Processing statistics model"""
    
    total_documents_created: int = 0
    total_chunks_created: int = 0
    total_processing_time: float = 0.0
    documents_by_stage: Dict[str, int] = Field(default_factory=dict)
    errors_encountered: int = 0
    pipeline_stages_executed: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### Attributes

- `total_documents_created` (int): Total number of documents created
- `total_chunks_created` (int): Total number of chunks created
- `total_processing_time` (float): Total processing time in seconds
- `documents_by_stage` (Dict[str, int]): Document count by stage
- `errors_encountered` (int): Number of errors encountered
- `pipeline_stages_executed` (int): Number of pipeline stages executed
- `metadata` (Dict[str, Any]): Additional statistics