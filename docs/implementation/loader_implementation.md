# Loader Implementation - refinire-rag

This document describes the implementation of the Loader system in refinire-rag, which provides flexible document loading capabilities with parallel processing support and extension-based delegation.

## Architecture Overview

The Loader system follows a modular architecture with:

1. **Base Interfaces**: Abstract classes defining the contract for loaders and metadata generators
2. **Universal Loader**: Main entry point that delegates to specialized loaders based on file extension
3. **Specialized Loaders**: Format-specific implementations for different file types
4. **Metadata Generation**: Flexible system for enriching documents with additional metadata
5. **Parallel Processing**: Support for concurrent loading with progress tracking

## Core Components

### Document Model

The `Document` class represents a loaded document with minimal required metadata fields:

```python
@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]  # Flexible metadata with 4 required fields
```

**Required metadata fields:**
- `path`: str - File path
- `created_at`: str - ISO 8601 timestamp
- `file_type`: str - File extension
- `size_bytes`: int - File size in bytes

All other metadata is completely flexible and user-defined.

### Base Loader Interface

```python
class Loader(ABC):
    @abstractmethod
    def load_single(self, path: Union[str, Path]) -> Document:
        """Load a single document from path"""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        pass
    
    def load_batch(self, paths: List[Union[str, Path]], 
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> LoadingResult:
        """Load multiple documents with optional parallel processing"""
        pass
```

### Metadata Generator Interface

```python
class MetadataGenerator(ABC):
    @abstractmethod
    def generate_metadata(self, required_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional metadata from required fields"""
        pass
```

## Usage Examples

### Basic Usage

```python
from refinire_rag import UniversalLoader

# Simple loading
loader = UniversalLoader()
document = loader.load("document.pdf")

print(f"Loaded: {document.path}")
print(f"Content length: {len(document.content)}")
print(f"Metadata: {document.metadata}")
```

### With Metadata Generation

```python
from refinire_rag import UniversalLoader, PathBasedMetadataGenerator

# Define path-based metadata rules
path_rules = {
    "*/docs/public/*": {
        "access_group": "public",
        "classification": "open"
    },
    "*/docs/internal/*": {
        "access_group": "employees",
        "classification": "internal"
    },
    "*/docs/technical/*": {
        "access_group": "engineers",
        "classification": "technical",
        "tags": ["technical", "engineering"]
    }
}

# Create metadata generator
metadata_gen = PathBasedMetadataGenerator(path_rules)

# Create loader with metadata generation
loader = UniversalLoader(metadata_generator=metadata_gen)

# Load document - metadata will be automatically enriched
doc = loader.load("docs/technical/api_guide.pdf")

print(f"Access group: {doc.metadata['access_group']}")  # 'engineers'
print(f"Classification: {doc.metadata['classification']}")  # 'technical'
print(f"Tags: {doc.metadata['tags']}")  # ['technical', 'engineering']
```

### Parallel Batch Loading

```python
from refinire_rag import UniversalLoader, LoadingConfig

# Configure parallel loading
config = LoadingConfig(
    parallel=True,
    max_workers=4,
    skip_errors=True,
    timeout_per_file=30.0
)

loader = UniversalLoader(config=config)

# Define progress callback
def progress_callback(completed: int, total: int):
    percentage = (completed / total) * 100
    print(f"Progress: {completed}/{total} ({percentage:.1f}%)")

# Load multiple documents
file_paths = ["doc1.pdf", "doc2.md", "doc3.txt", "doc4.html"]
result = loader.load_batch(file_paths, progress_callback=progress_callback)

print(f"Loaded {result.successful_count} documents successfully")
print(f"Failed to load {result.failed_count} documents")
print(f"Success rate: {result.success_rate:.1f}%")
print(f"Total time: {result.total_time_seconds:.2f} seconds")

# Access loaded documents
for doc in result.documents:
    print(f"- {doc.path}: {len(doc.content)} characters")
```

### Async Loading

```python
import asyncio
from refinire_rag import UniversalLoader

async def load_documents_async():
    loader = UniversalLoader()
    
    file_paths = ["doc1.pdf", "doc2.md", "doc3.txt"]
    result = await loader.load_batch_async(file_paths)
    
    print(f"Async loading completed: {result.summary()}")

# Run async loading
asyncio.run(load_documents_async())
```

### Custom Specialized Loaders

```python
from refinire_rag.loaders import SpecializedLoader

class CustomXMLLoader(SpecializedLoader):
    """Custom loader for XML files"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract text content from XML file"""
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Extract all text content
        text_parts = []
        for elem in root.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
        
        return '\n'.join(text_parts)
    
    def supported_formats(self) -> List[str]:
        return ['.xml']

# Register custom loader
loader = UniversalLoader()
loader.register_loader('.xml', CustomXMLLoader)

# Now can load XML files
doc = loader.load("data.xml")
```

## Supported File Formats

The UniversalLoader supports the following formats out of the box:

| Format | Extension | Loader Class | Dependencies |
|--------|-----------|--------------|--------------|
| Plain Text | `.txt` | TextLoader | None |
| Markdown | `.md`, `.markdown` | MarkdownLoader | None |
| PDF | `.pdf` | PDFLoader | PyPDF2 |
| HTML | `.html`, `.htm` | HTMLLoader | BeautifulSoup4 |
| JSON | `.json` | JSONLoader | None |
| CSV | `.csv` | CSVLoader | None |

### Optional Advanced Loaders

Additional loaders can be installed as extensions:

- **DoclingLoader**: Advanced document processing (install separately)
- **UnstructuredLoader**: Multi-format document processing (install separately)
- **ExcelLoader**: Excel file support (install separately)

## Configuration Options

### LoadingConfig

```python
@dataclass
class LoadingConfig:
    parallel: bool = True                    # Enable parallel processing
    max_workers: Optional[int] = None        # Number of workers (default: CPU count)
    use_multiprocessing: bool = False        # Use processes instead of threads
    chunk_size: int = 10                     # Batch size for parallel processing
    timeout_per_file: Optional[float] = None # Timeout per file in seconds
    skip_errors: bool = True                 # Continue on individual file errors
```

### Performance Tuning

- **Parallel Processing**: Enabled by default for multiple files
- **Thread vs Process**: Threads for I/O-bound tasks, processes for CPU-bound
- **Error Handling**: Skip individual file errors to continue batch processing
- **Timeouts**: Prevent hanging on problematic files

## Error Handling

The Loader system provides comprehensive error handling:

```python
from refinire_rag.exceptions import LoaderError

try:
    doc = loader.load("problematic_file.pdf")
except LoaderError as e:
    print(f"Failed to load document: {e}")

# Batch loading with error recovery
result = loader.load_batch(file_paths)
if result.has_errors:
    print(f"Errors occurred:")
    for path, error in zip(result.failed_paths, result.errors):
        print(f"  {path}: {error}")
```

## Extension and Customization

### Creating Custom Metadata Generators

```python
class ContentBasedMetadataGenerator(MetadataGenerator):
    """Generate metadata based on content analysis"""
    
    def generate_metadata(self, required_metadata: Dict[str, Any]) -> Dict[str, Any]:
        # Analyze content to extract metadata
        return {
            "language": "ja",
            "estimated_reading_time": 5,
            "content_type": "technical"
        }
```

### Subpackage Loaders

Create specialized loaders in subpackages:

```
refinire_rag/
├── loaders/
│   ├── extensions/
│   │   ├── docling/
│   │   │   └── __init__.py  # DoclingLoader
│   │   ├── unstructured/
│   │   │   └── __init__.py  # UnstructuredLoader
│   │   └── excel/
│   │       └── __init__.py  # ExcelLoader
```

## Testing

Run the test suite to verify the implementation:

```bash
python -m pytest tests/unit/test_loader.py
python -m pytest tests/unit/test_loaders_detailed.py
```

The test covers:
- Document creation and validation
- Metadata generation
- Universal loader functionality
- Configuration and results

## Future Enhancements

Planned improvements include:

1. **Streaming Support**: For very large files
2. **Caching**: Document and metadata caching
3. **Validation**: Content validation and sanitization
4. **Monitoring**: Detailed performance metrics
5. **Cloud Storage**: Support for S3, Azure Blob, etc.

## Integration with CorpusManager

The Loader system integrates seamlessly with CorpusManager through the DocumentProcessor architecture:

```python
from refinire_rag import CorpusManager, UniversalLoader, PathBasedMetadataGenerator

# Create configured loader
metadata_gen = PathBasedMetadataGenerator(path_rules)
loader = UniversalLoader(metadata_generator=metadata_gen)

# Use in CorpusManager configuration
config = CorpusManagerConfig(
    document_store=doc_store,
    vector_store=vector_store,
    processors=[
        loader,  # As DocumentProcessor
        Normalizer(norm_config),
        TokenBasedChunker(chunk_config)
    ]
)

corpus_manager = CorpusManager(config)
results = corpus_manager.process_corpus(["documents/"])
```

This ensures consistent document loading throughout the RAG pipeline with proper metadata enrichment and error handling.