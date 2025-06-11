# refinire-rag Tests

This directory contains the test suite for refinire-rag.

## Test Structure

### Unit Tests (`tests/unit/`)
Tests for individual components and modules:

- `test_loader.py` - Basic document loading functionality
- `test_loaders_detailed.py` - Detailed loader testing with various formats
- `test_document_store.py` - Document storage and retrieval
- `test_embedding.py` - Embedding generation and TF-IDF functionality
- `test_chunking.py` - Document chunking algorithms
- `test_vector_store.py` - Vector storage and similarity search
- `test_vector_simple.py` - Simple vector operations
- `test_document_processor.py` - Document processor interface
- `test_improved_processor.py` - Enhanced processor functionality
- `test_processor_comprehensive.py` - Comprehensive processor testing
- `test_document_store_loader.py` - Document store loader integration

### Integration Tests (`tests/integration/`)
Tests for complete workflows and component integration:

- `test_corpus_manager.py` - Complete RAG pipeline from loading to embedding storage
- `test_document_processor_integration.py` - Document processor pipeline integration

## Running Tests

### Run All Tests
```bash
python -m pytest tests/
```

### Run Unit Tests Only
```bash
python -m pytest tests/unit/
```

### Run Integration Tests Only
```bash
python -m pytest tests/integration/
```

### Run Specific Test File
```bash
python -m pytest tests/unit/test_loader.py
```

### Run with Verbose Output
```bash
python -m pytest tests/ -v
```

## Legacy Test Execution
Some tests can still be run directly as scripts:

```bash
# Run individual test files
python tests/unit/test_loader.py
python tests/integration/test_corpus_manager.py
```

## Test Coverage

The test suite covers:

- ✅ Document loading and processing
- ✅ Embedding generation (TF-IDF)
- ✅ Vector storage and retrieval
- ✅ Document chunking
- ✅ Complete RAG pipeline integration
- ✅ DocumentProcessor unified architecture
- ✅ Error handling and edge cases

## Writing New Tests

When adding new functionality:

1. **Unit tests**: Add to `tests/unit/` for individual component testing
2. **Integration tests**: Add to `tests/integration/` for workflow testing
3. Follow existing naming convention: `test_<component>.py`
4. Include both positive and negative test cases
5. Test error handling and edge cases

## Test Dependencies

Tests require:
- Python 3.10+
- pytest (optional, but recommended)
- All refinire-rag dependencies

Tests are designed to be self-contained and create temporary files/directories as needed.