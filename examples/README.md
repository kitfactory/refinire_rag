# DocumentProcessor Examples

This directory contains comprehensive examples demonstrating the DocumentProcessor system and its new configuration pattern where each processor defines its own config class.

## Examples Overview

### 1. `document_processor_example.py`
**Basic DocumentProcessor usage and configuration**

Demonstrates:
- Basic processor usage with default configuration
- Custom configuration creation and usage
- Configuration serialization/deserialization
- Dynamic config discovery
- Error handling strategies

**Key concepts:**
- Each processor defines its own config class via `get_config_class()`
- Config validation ensures type safety
- Processors can use provided config or instance config
- Graceful error handling vs strict error handling

**Run:**
```bash
python examples/document_processor_example.py
```

### 2. `custom_processor_example.py`
**Creating custom DocumentProcessor implementations**

Demonstrates:
- Text normalization processor with custom config
- Document enrichment processor with metadata analysis
- Document splitting processor with flexible splitting strategies
- Custom pipeline combining multiple processors

**Key concepts:**
- Inheriting from `DocumentProcessor` base class
- Defining custom `DocumentProcessorConfig` subclasses
- Implementing `get_config_class()` and `process()` methods
- Real-world processing examples (normalization, enrichment, splitting)

**Run:**
```bash
python examples/custom_processor_example.py
```

### 3. `pipeline_example.py`
**Advanced pipeline configurations and database integration**

Demonstrates:
- Simple pipelines without database storage
- Comprehensive multi-stage pipelines
- Database integration with lineage tracking
- Error recovery and handling strategies
- Pipeline statistics and performance monitoring

**Key concepts:**
- `DocumentPipeline` for chaining processors
- Database storage with `SQLiteDocumentStore`
- Intermediate result storage and lineage tracking
- Metadata search and filtering
- Pipeline performance analysis

**Run:**
```bash
python examples/pipeline_example.py
```

## Configuration Pattern

The new configuration pattern ensures type safety and self-documentation:

### 1. Define Custom Config Class
```python
@dataclass
class MyProcessorConfig(DocumentProcessorConfig):
    """Custom configuration for my processor"""
    my_parameter: str = "default_value"
    threshold: float = 0.8
    enable_feature: bool = True
```

### 2. Implement Processor Class
```python
class MyCustomProcessor(DocumentProcessor):
    """Custom document processor"""
    
    @classmethod
    def get_config_class(cls) -> Type[MyProcessorConfig]:
        return MyProcessorConfig
    
    def process(self, document: Document, config: Optional[MyProcessorConfig] = None) -> List[Document]:
        proc_config = config or self.config
        # Process document using config...
        return [processed_document]
```

### 3. Use Processor
```python
# With default config
processor = MyCustomProcessor()

# With custom config
custom_config = MyProcessorConfig(threshold=0.95, enable_feature=False)
processor = MyCustomProcessor(custom_config)

# Override config during processing
temp_config = MyProcessorConfig(threshold=0.7)
results = processor.process(document, temp_config)
```

## Key Features Demonstrated

### Type Safety
- Each processor defines its required configuration type
- Runtime validation prevents incorrect config usage
- IDE support for config fields and types

### Self-Documentation
- Config classes clearly show processor requirements
- Dynamic discovery of processor capabilities
- Automatic validation and serialization

### Flexibility
- Default configs for quick usage
- Custom configs for specific needs
- Runtime config override for temporary changes

### Pipeline Integration
- Processors work seamlessly in pipelines
- Each processor maintains its own configuration
- Pipeline statistics track individual processor performance

### Database Integration
- Store all processing results with lineage tracking
- Search by metadata and processing stage
- Track document transformation history

### Error Handling
- Graceful error handling (continue on errors)
- Strict error handling (fail fast on errors)
- Comprehensive error reporting and statistics

## Running the Examples

### Prerequisites
```bash
# Ensure you have the refinire-rag package installed
cd /path/to/refinire-rag
pip install -e .
```

### Run Individual Examples
```bash
# Basic usage
python examples/document_processor_example.py

# Custom processors
python examples/custom_processor_example.py

# Advanced pipelines
python examples/pipeline_example.py
```

### Run All Examples
```bash
# Run all examples sequentially
python examples/document_processor_example.py && \
python examples/custom_processor_example.py && \
python examples/pipeline_example.py
```

## Example Output

Each example produces detailed output showing:
- Document processing steps
- Configuration usage
- Processing results and statistics
- Error handling demonstrations
- Performance metrics

Example output includes:
- Original document content and metadata
- Processing configuration details
- Intermediate and final processing results
- Pipeline statistics and timing
- Database storage and lineage information

## Next Steps

After running these examples, you can:

1. **Create Your Own Processors**: Use the patterns shown to implement domain-specific processors
2. **Build Complex Pipelines**: Combine multiple processors for sophisticated document processing workflows
3. **Integrate with Database**: Use `SQLiteDocumentStore` for persistent storage and lineage tracking
4. **Scale Up**: Apply the patterns to larger document collections
5. **Customize Error Handling**: Implement specific error recovery strategies for your use case

## Best Practices

Based on the examples:

1. **Always define custom config classes** for processors with specific requirements
2. **Use graceful error handling** in production pipelines
3. **Store intermediate results** when debugging or analyzing pipeline behavior
4. **Track lineage** for document provenance and debugging
5. **Monitor pipeline statistics** for performance optimization
6. **Validate configurations** before processing large document sets

## Related Documentation

- `docs/backend_interfaces.md` - DocumentProcessor interface specification
- `docs/processor_config_example.md` - Configuration pattern documentation (Japanese)
- `src/refinire_rag/processing/` - DocumentProcessor implementation
- `src/refinire_rag/chunking/` - Chunker implementation as example processor