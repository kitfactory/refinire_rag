# refinire-rag Examples

## Overview / 概要

This directory contains comprehensive examples demonstrating how to use refinire-rag for building production-ready RAG (Retrieval-Augmented Generation) systems. The examples range from simple quick-start scripts to complete enterprise-ready implementations.

このディレクトリには、本番対応のRAG（検索拡張生成）システム構築のためのrefinire-ragの使用方法を実証する包括的な例が含まれています。例は、シンプルなクイックスタートスクリプトから完全な企業対応実装まで多岐にわたります。

## 🚀 Core Tutorial Examples / コアチュートリアル例

These examples accompany the main tutorial series and demonstrate the complete RAG workflow:

### **[complete_rag_tutorial.py](complete_rag_tutorial.py)**
**Complete End-to-End RAG Tutorial / 完全エンドツーエンドRAGチュートリアル**

The most comprehensive example that demonstrates the entire RAG lifecycle:
- Part 1: Corpus creation with multiple strategies
- Part 2: Query engine configuration and testing  
- Part 3: Comprehensive evaluation and reporting

```bash
python complete_rag_tutorial.py
```

**Features:**
- Automated knowledge base creation
- Performance comparison across approaches
- Detailed evaluation metrics
- Comprehensive reporting

### **[tutorial_part1_corpus_creation_example.py](tutorial_part1_corpus_creation_example.py)**
**Part 1: Corpus Creation Examples / パート1: コーパス作成例**

Demonstrates corpus building with different strategies:
- Simple RAG: Basic document processing
- Semantic RAG: Dictionary-based normalization
- Knowledge RAG: Full pipeline with knowledge graphs

### **[tutorial_part2_query_engine_example.py](tutorial_part2_query_engine_example.py)**
**Part 2: Query Engine Examples / パート2: クエリエンジン例**

Shows query processing and answer generation:
- Basic query operations
- Advanced configurations (performance vs accuracy)
- Error handling and debugging
- Performance monitoring

### **[tutorial_part3_evaluation_example.py](tutorial_part3_evaluation_example.py)**
**Part 3: Evaluation Examples / パート3: 評価例**

Comprehensive RAG system evaluation:
- Automated QA pair generation
- Multi-metric performance assessment
- Contradiction detection
- Advanced evaluation techniques

## 📖 Basic Examples / 基本例

### Quick Start / クイックスタート

#### **[simple_rag_test.py](simple_rag_test.py)**
Minimal working RAG system in under 20 lines:
```python
from refinire_rag import create_simple_rag

rag = create_simple_rag("documents/")
answer = rag.query("What is this about?")
print(answer)
```

#### **[quickstart_guide.py](quickstart_guide.py)**
Expanded quick start with error handling and configuration options.

### Component Examples / コンポーネント例

#### **[corpus_manager_demo.py](corpus_manager_demo.py)**
Comprehensive CorpusManager demonstration:
- Multiple corpus building strategies
- Custom pipeline configurations
- Performance monitoring

#### **[query_engine_demo.py](query_engine_demo.py)**  
QueryEngine configuration and usage:
- Component initialization
- Query processing
- Result analysis

#### **[quality_lab_example.py](quality_lab_example.py)**
QualityLab evaluation system:
- QA pair generation
- Performance evaluation
- Report generation

## 🏢 Enterprise Examples / 企業例

### **[tutorial_05_enterprise_usage.py](tutorial_05_enterprise_usage.py)**
Production-ready enterprise patterns:
- Department-level data isolation
- Scalable processing pipelines
- Production monitoring
- Security considerations

### **[incremental_loading_demo.py](incremental_loading_demo.py)**
Efficient incremental document processing:
- Change detection
- Batch processing
- Performance optimization

### **[evaluation_example.py](evaluation_example.py)**
Advanced evaluation techniques:
- Custom metrics
- Automated testing
- Performance benchmarking

## 🔧 DocumentProcessor Examples / DocumentProcessor例

This section contains comprehensive examples demonstrating the DocumentProcessor system and its configuration pattern where each processor defines its own config class.

### **[document_processor_example.py](document_processor_example.py)**
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

## 🚀 Getting Started / はじめに

### Prerequisites / 前提条件
```bash
# Install refinire-rag
pip install refinire-rag

# Set up environment
export OPENAI_API_KEY="your-api-key"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
```

### Quick Test / クイックテスト
```bash
# Run the complete tutorial (recommended)
python complete_rag_tutorial.py

# Or start with simple examples
python simple_rag_test.py
python quickstart_guide.py
```

### Individual Components / 個別コンポーネント
```bash
# Test corpus creation
python tutorial_part1_corpus_creation_example.py

# Test query engine
python tutorial_part2_query_engine_example.py

# Test evaluation
python tutorial_part3_evaluation_example.py
```

### Enterprise Patterns / 企業パターン
```bash
# Enterprise usage patterns
python tutorial_05_enterprise_usage.py

# Incremental loading
python incremental_loading_demo.py
```

## 📚 Learning Path / 学習パス

### For Beginners / 初心者向け
1. **Start here:** `complete_rag_tutorial.py`
2. **Basic components:** Tutorial Part 1, 2, 3 examples
3. **Simple examples:** `simple_rag_test.py`, `quickstart_guide.py`
4. **Component focus:** Individual component examples

### For Developers / 開発者向け
1. **Architecture understanding:** `document_processor_example.py`
2. **Component integration:** `pipeline_example.py`
3. **Custom development:** `custom_processor_example.py`
4. **Complete workflow:** `complete_rag_tutorial.py`

### For Enterprise Users / 企業ユーザー向け
1. **Complete understanding:** `complete_rag_tutorial.py`
2. **Production patterns:** `tutorial_05_enterprise_usage.py`
3. **Scalability:** `incremental_loading_demo.py`
4. **Evaluation:** `quality_lab_example.py`

## 💡 Tips and Best Practices / ヒントとベストプラクティス

### Performance / パフォーマンス
- Use incremental loading for large document sets
- Monitor memory usage with vector operations
- Implement appropriate caching strategies

### Production / 本番環境
- Follow enterprise usage patterns from Tutorial 5
- Implement comprehensive error handling
- Add monitoring and logging

### Development / 開発
- Start with the complete tutorial for best understanding
- Use individual examples for specific learning
- Customize examples for your specific domain

## 🆘 Troubleshooting / トラブルシューティング

### Common Issues / よくある問題
1. **Missing API keys:** Check environment variable setup
2. **Import errors:** Ensure refinire-rag is properly installed
3. **Performance issues:** Review corpus size and configuration
4. **Memory errors:** Use incremental processing for large datasets

### Getting Help / ヘルプの取得
- Start with the complete tutorial for comprehensive examples
- Review individual component examples for specific issues
- Consult the main documentation in `docs/`
- Run examples with debug output enabled

## 📖 Related Documentation / 関連ドキュメント

- [Tutorial Documentation](../docs/tutorials/) - Complete tutorial series
- [API Reference](../docs/api/) - Detailed API documentation
- [Architecture Guide](../docs/design/architecture.md) - System design
- `docs/backend_interfaces.md` - DocumentProcessor interface specification
- `docs/processor_config_example.md` - Configuration pattern documentation
- `src/refinire_rag/processing/` - DocumentProcessor implementation

---

**Happy coding! / コーディングを楽しんでください！**

These examples provide comprehensive coverage of refinire-rag capabilities. Start with the complete tutorial for the best learning experience, then explore specific examples based on your needs.

これらの例は、refinire-ragの機能を包括的にカバーしています。最良の学習体験のために完全チュートリアルから始め、その後ニーズに基づいて特定の例を探索してください。