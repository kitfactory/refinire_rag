# Part 1: コーパス作成（インデックス）チュートリアル

## Overview / 概要

This tutorial demonstrates how to create and manage a document corpus using refinire-rag's CorpusManager. The CorpusManager provides flexible corpus building with preset configurations, stage selection, and custom pipelines.

このチュートリアルでは、refinire-ragのCorpusManagerを使用してドキュメントコーパスを作成・管理する方法を説明します。CorpusManagerは、プリセット設定、ステージ選択、カスタムパイプラインによる柔軟なコーパス構築を提供します。

## Learning Objectives / 学習目標

- Understand different corpus building approaches / 異なるコーパス構築アプローチの理解
- Learn preset configurations (Simple, Semantic, Knowledge RAG) / プリセット設定の学習（Simple、Semantic、Knowledge RAG）
- Master stage selection for custom workflows / カスタムワークフロー用のステージ選択のマスター
- Create custom pipelines for specialized processing / 専用処理のためのカスタムパイプライン作成

## Prerequisites / 前提条件

```bash
# Install refinire-rag
pip install refinire-rag

# Set environment variables (if using LLM features)
export OPENAI_API_KEY="your-api-key"
export REFINIRE_RAG_LLM_MODEL="gpt-4o-mini"
```

## Quick Start Example / クイックスタート例

```python
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore

# Initialize storage components
doc_store = SQLiteDocumentStore("documents.db")
vector_store = InMemoryVectorStore()

# Create simple RAG corpus (Load → Chunk → Vector)
manager = CorpusManager.create_simple_rag(doc_store, vector_store)
stats = manager.build_corpus(["documents/"])

print(f"Processed {stats.total_files_processed} files")
print(f"Created {stats.total_chunks_created} chunks")
```

## 1. Preset Configurations / プリセット設定

### 1.1 Simple RAG
Basic pipeline for quick prototyping / プロトタイピング用の基本パイプライン

```python
# Simple RAG: Load → Chunk → Vector
simple_manager = CorpusManager.create_simple_rag(doc_store, vector_store)
simple_stats = simple_manager.build_corpus(file_paths)

print(f"Simple RAG Results:")
print(f"- Files: {simple_stats.total_files_processed}")
print(f"- Chunks: {simple_stats.total_chunks_created}")
print(f"- Time: {simple_stats.total_processing_time:.3f}s")
```

### 1.2 Semantic RAG
Enhanced with dictionary-based normalization / 辞書ベース正規化で強化

```python
# Semantic RAG: Load → Dictionary → Normalize → Chunk → Vector
semantic_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
semantic_stats = semantic_manager.build_corpus(file_paths)

print(f"Semantic RAG Results:")
print(f"- Normalized expressions using domain dictionary")
print(f"- Better semantic consistency")
```

### 1.3 Knowledge RAG
Full pipeline with knowledge graph generation / 知識グラフ生成を含む完全パイプライン

```python
# Knowledge RAG: Load → Dictionary → Graph → Normalize → Chunk → Vector
knowledge_manager = CorpusManager.create_knowledge_rag(doc_store, vector_store)
knowledge_stats = knowledge_manager.build_corpus(file_paths)

print(f"Knowledge RAG Results:")
print(f"- Generated domain-specific dictionary")
print(f"- Built knowledge graph relationships")
print(f"- Enhanced semantic understanding")
```

## 2. Stage Selection Approach / ステージ選択アプローチ

### 2.1 Custom Stage Selection
Select specific processing stages / 特定の処理ステージを選択

```python
from refinire_rag.processing.dictionary_maker import DictionaryMakerConfig
from refinire_rag.processing.chunker import ChunkingConfig
from refinire_rag.loader.loader import LoaderConfig

# Configure individual stages
stage_configs = {
    "loader_config": LoaderConfig(),
    "dictionary_config": DictionaryMakerConfig(
        dictionary_file_path="custom_dictionary.md",
        focus_on_technical_terms=True,
        extract_abbreviations=True
    ),
    "chunker_config": ChunkingConfig(
        chunk_size=256,
        overlap=32,
        split_by_sentence=True
    )
}

# Execute selected stages only
corpus_manager = CorpusManager(doc_store, vector_store)
selected_stages = ["load", "dictionary", "chunk", "vector"]

stage_stats = corpus_manager.build_corpus(
    file_paths=file_paths,
    stages=selected_stages,
    stage_configs=stage_configs
)

print(f"Selected stages: {selected_stages}")
print(f"Generated dictionary with technical terms")
```

### 2.2 Available Processing Stages / 利用可能な処理ステージ

| Stage | Purpose | Output |
|-------|---------|--------|
| `load` | File loading and conversion | Documents in DocumentStore |
| `dictionary` | Domain-specific term extraction | Dictionary file (.md) |
| `graph` | Relationship extraction | Knowledge graph file (.md) |
| `normalize` | Expression normalization | Normalized documents |
| `chunk` | Text segmentation | Document chunks |
| `vector` | Embedding generation | Vector embeddings |

## 3. Custom Pipeline Approach / カスタムパイプラインアプローチ

### 3.1 Multi-Stage Custom Pipeline
Create sophisticated processing workflows / 高度な処理ワークフローの作成

```python
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.processing.document_store_processor import DocumentStoreProcessor
from refinire_rag.loader.text_loader import TextLoader
from refinire_rag.processing.dictionary_maker import DictionaryMaker
from refinire_rag.processing.normalizer import Normalizer
from refinire_rag.processing.chunker import Chunker

# Define custom pipelines
custom_pipelines = [
    # Stage 1: Load and store original documents
    DocumentPipeline([
        TextLoader(LoaderConfig()),
        DocumentStoreProcessor(doc_store)
    ]),
    
    # Stage 2: Extract dictionary from originals
    DocumentPipeline([
        DocumentStoreLoader(doc_store, config=DocumentStoreLoaderConfig(
            processing_stage="original"
        )),
        DictionaryMaker(DictionaryMakerConfig(
            dictionary_file_path="pipeline_dictionary.md"
        ))
    ]),
    
    # Stage 3: Normalize and store
    DocumentPipeline([
        DocumentStoreLoader(doc_store, config=DocumentStoreLoaderConfig(
            processing_stage="original"
        )),
        Normalizer(NormalizerConfig(
            dictionary_file_path="pipeline_dictionary.md"
        )),
        DocumentStoreProcessor(doc_store)
    ]),
    
    # Stage 4: Chunk normalized documents
    DocumentPipeline([
        DocumentStoreLoader(doc_store, config=DocumentStoreLoaderConfig(
            processing_stage="normalized"
        )),
        Chunker(ChunkingConfig(chunk_size=128, overlap=16))
    ])
]

# Execute custom pipelines
pipeline_stats = corpus_manager.build_corpus(
    file_paths=file_paths,
    custom_pipelines=custom_pipelines
)
```

## 4. File Format Support / ファイル形式サポート

### 4.1 Supported File Types / サポートファイル形式

```python
# Text files
text_files = ["document.txt", "readme.md"]

# CSV files
csv_files = ["data.csv", "records.csv"]

# JSON files
json_files = ["config.json", "data.json"]

# HTML files
html_files = ["webpage.html", "documentation.html"]

# All formats in one corpus
all_files = text_files + csv_files + json_files + html_files
stats = manager.build_corpus(all_files)
```

### 4.2 Directory Processing / ディレクトリ処理

```python
# Process entire directories
directory_paths = [
    "documents/",
    "knowledge_base/",
    "technical_docs/"
]

# Incremental loading (only changed files)
from refinire_rag.loader.incremental_directory_loader import IncrementalDirectoryLoader
from refinire_rag.loader.file_tracker import FileTracker

tracker = FileTracker("file_tracking.json")
incremental_loader = IncrementalDirectoryLoader(tracker=tracker)

# Only process new/modified files
incremental_stats = manager.build_corpus(
    file_paths=directory_paths,
    use_incremental=True
)
```

## 5. Advanced Configuration / 高度な設定

### 5.1 Dictionary Maker Configuration / 辞書作成設定

```python
dictionary_config = DictionaryMakerConfig(
    dictionary_file_path="domain_dictionary.md",
    focus_on_technical_terms=True,
    extract_abbreviations=True,
    include_definitions=True,
    min_term_frequency=2,
    max_terms_per_document=50,
    llm_model="gpt-4o-mini"
)
```

### 5.2 Chunking Configuration / チャンク設定

```python
chunking_config = ChunkingConfig(
    chunk_size=512,           # Characters per chunk
    overlap=50,               # Overlap between chunks
    split_by_sentence=True,   # Preserve sentence boundaries
    min_chunk_size=100,       # Minimum chunk size
    separators=["\n\n", "\n", ".", "!", "?"]  # Splitting separators
)
```

### 5.3 Normalization Configuration / 正規化設定

```python
normalizer_config = NormalizerConfig(
    dictionary_file_path="domain_dictionary.md",
    case_sensitive=False,
    preserve_formatting=True,
    expand_abbreviations=True,
    normalize_numbers=True
)
```

## 6. Monitoring and Statistics / 監視と統計

### 6.1 Processing Statistics / 処理統計

```python
# Get detailed statistics
stats = manager.build_corpus(file_paths)

print(f"Processing Summary:")
print(f"- Total files processed: {stats.total_files_processed}")
print(f"- Documents created: {stats.total_documents_created}")
print(f"- Chunks created: {stats.total_chunks_created}")
print(f"- Processing time: {stats.total_processing_time:.3f}s")
print(f"- Pipeline stages: {stats.pipeline_stages_executed}")
print(f"- Documents by stage: {stats.documents_by_stage}")
print(f"- Errors encountered: {len(stats.errors_encountered)}")
```

### 6.2 Error Handling / エラーハンドリング

```python
try:
    stats = manager.build_corpus(file_paths)
    
    if stats.errors_encountered:
        print(f"Encountered {len(stats.errors_encountered)} errors:")
        for error in stats.errors_encountered:
            print(f"- {error['file']}: {error['message']}")
    
except Exception as e:
    print(f"Corpus building failed: {e}")
    # Handle specific error types
    if "FileNotFoundError" in str(e):
        print("Check file paths and permissions")
    elif "LLMError" in str(e):
        print("Check LLM configuration and API keys")
```

## 7. Best Practices / ベストプラクティス

### 7.1 Performance Optimization / パフォーマンス最適化

```python
# For large corpora, use incremental loading
manager = CorpusManager(doc_store, vector_store)
stats = manager.build_corpus(
    file_paths=large_directory_paths,
    use_incremental=True,
    batch_size=100  # Process in batches
)

# Use appropriate chunk sizes
# Small chunks (128-256): Better for precise retrieval
# Large chunks (512-1024): Better for context preservation
```

### 7.2 Storage Considerations / ストレージ考慮事項

```python
# For production: Use persistent storage
doc_store = SQLiteDocumentStore("production_documents.db")

# For development: Use in-memory storage
doc_store = SQLiteDocumentStore(":memory:")

# For departments: Separate databases
hr_store = SQLiteDocumentStore("data/hr_documents.db")
sales_store = SQLiteDocumentStore("data/sales_documents.db")
```

### 7.3 Quality Assurance / 品質保証

```python
# Validate corpus after building
if stats.total_chunks_created == 0:
    print("Warning: No chunks created. Check file formats and content.")

if stats.errors_encountered:
    print(f"Warning: {len(stats.errors_encountered)} files failed processing")

# Check generated artifacts
import os
if os.path.exists("domain_dictionary.md"):
    print("✓ Dictionary successfully generated")
if os.path.exists("knowledge_graph.md"):
    print("✓ Knowledge graph successfully generated")
```

## 8. Complete Example / 完全な例

```python
#!/usr/bin/env python3
"""
Complete corpus creation example
完全なコーパス作成例
"""

from pathlib import Path
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore

def main():
    # Setup / セットアップ
    doc_store = SQLiteDocumentStore("my_corpus.db")
    vector_store = InMemoryVectorStore()
    
    # Create sample documents / サンプル文書作成
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)
    
    (sample_dir / "ai_overview.txt").write_text("""
    Artificial Intelligence (AI) is the simulation of human intelligence 
    in machines programmed to think and learn like humans. Key applications 
    include machine learning, natural language processing, and computer vision.
    """)
    
    (sample_dir / "machine_learning.txt").write_text("""
    Machine Learning (ML) is a subset of AI that enables computers to learn 
    without explicit programming. Popular algorithms include neural networks, 
    decision trees, and support vector machines.
    """)
    
    # Build corpus with semantic RAG / セマンティックRAGでコーパス構築
    print("Building corpus with semantic RAG...")
    manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    
    stats = manager.build_corpus([str(sample_dir)])
    
    # Results / 結果
    print(f"\n✅ Corpus Creation Complete:")
    print(f"   Files processed: {stats.total_files_processed}")
    print(f"   Documents created: {stats.total_documents_created}")
    print(f"   Chunks created: {stats.total_chunks_created}")
    print(f"   Processing time: {stats.total_processing_time:.3f}s")
    
    # Validate results / 結果検証
    total_docs = doc_store.count_documents()
    total_vectors = vector_store.count()
    
    print(f"\n📊 Storage Validation:")
    print(f"   Documents in store: {total_docs}")
    print(f"   Vectors in store: {total_vectors}")
    
    return True

if __name__ == "__main__":
    success = main()
    print("\n🎉 Tutorial completed successfully!" if success else "\n❌ Tutorial failed")
```

## Next Steps / 次のステップ

After creating your corpus, proceed to:
コーパス作成後、次に進む：

1. **Part 2: Query Engine** - Learn to search and retrieve from your corpus
   **Part 2: Query Engine** - コーパスからの検索と取得を学習
2. **Part 3: Evaluation** - Evaluate your RAG system performance
   **Part 3: Evaluation** - RAGシステムのパフォーマンス評価

## Resources / リソース

- [CorpusManager API Documentation](../api/corpus_manager.md)
- [Processing Configuration Reference](../development/processor_config_example.md)
- [Example Scripts](../../examples/)