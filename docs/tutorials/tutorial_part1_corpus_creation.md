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

## Core Concepts / 基本概念

### What is a Document Corpus? / ドキュメントコーパスとは？

A **document corpus** is a structured collection of processed documents that serves as the foundation for RAG (Retrieval-Augmented Generation) systems. Think of it as a searchable knowledge base where:

**ドキュメントコーパス**は、RAG（検索拡張生成）システムの基礎となる、処理済み文書の構造化されたコレクションです。以下のような検索可能な知識ベースと考えてください：

- **Raw documents** are converted into machine-readable format / **生文書**を機械可読形式に変換
- **Text is segmented** into meaningful chunks for better retrieval / **テキストを分割**して検索精度を向上
- **Semantic embeddings** are generated for similarity search / **セマンティック埋め込み**を生成して類似検索を実現
- **Metadata** is extracted and indexed for filtering / **メタデータ**を抽出・インデックス化してフィルタリングを可能に

### Why Different Corpus Building Approaches? / なぜ異なるコーパス構築アプローチが必要？

Different use cases require different processing strategies:

異なるユースケースには、異なる処理戦略が必要です：

#### 1. **Simple RAG** - Quick Prototyping / クイックプロトタイピング
- **When to use**: Testing, demos, simple applications / **使用場面**: テスト、デモ、シンプルなアプリケーション
- **Processing**: Load → Chunk → Vectorize / **処理**: 読み込み → チャンク化 → ベクトル化
- **Benefits**: Fast setup, minimal configuration / **利点**: 高速セットアップ、最小設定

#### 2. **Semantic RAG** - Enhanced Understanding / 理解力向上
- **When to use**: Domain-specific content, terminology consistency / **使用場面**: 専門分野コンテンツ、用語統一
- **Processing**: Load → Dictionary → Normalize → Chunk → Vectorize / **処理**: 読み込み → 辞書 → 正規化 → チャンク化 → ベクトル化
- **Benefits**: Better semantic consistency, domain adaptation / **利点**: セマンティック一貫性向上、ドメイン適応

#### 3. **Knowledge RAG** - Advanced Analytics / 高度な分析
- **When to use**: Complex relationships, knowledge discovery / **使用場面**: 複雑な関係性、知識発見
- **Processing**: Load → Dictionary → Graph → Normalize → Chunk → Vectorize / **処理**: 読み込み → 辞書 → グラフ → 正規化 → チャンク化 → ベクトル化
- **Benefits**: Relationship extraction, enhanced reasoning / **利点**: 関係抽出、推論能力向上

### Document Processing Pipeline / 文書処理パイプライン

Understanding the document processing stages is crucial for choosing the right approach:

適切なアプローチを選択するには、文書処理ステージの理解が重要です：

```
📄 Raw Documents → 🔄 Processing Pipeline → 🗂️ Searchable Corpus
                        ↓
              [Load] → [Analyze] → [Normalize] → [Chunk] → [Vectorize]
```

Each stage serves a specific purpose:

各ステージには特定の目的があります：

| Stage | Purpose | What It Does | When to Use |
|-------|---------|--------------|-------------|
| **Load** | Import | File parsing and document creation / ファイル解析と文書作成 | Always required / 常に必要 |
| **Dictionary** | Analyze | Extract domain terms and abbreviations / 専門用語・略語抽出 | Domain-specific content / 専門分野コンテンツ |
| **Graph** | Relate | Build knowledge relationships / 知識関係の構築 | Complex documents / 複雑な文書 |
| **Normalize** | Standardize | Unify terminology variations / 用語バリエーションの統一 | Inconsistent terminology / 用語の不統一 |
| **Chunk** | Segment | Break text into optimal pieces / テキストを最適な断片に分割 | Always required / 常に必要 |
| **Vector** | Index | Generate semantic embeddings / セマンティック埋め込み生成 | Always required / 常に必要 |

### Choosing Your Approach / アプローチの選択

**Decision Tree** for selecting the right corpus building approach:

適切なコーパス構築アプローチを選択するための**決定ツリー**：

```
🤔 What type of content are you processing?
   あなたが処理するコンテンツのタイプは？

├── 📝 General documents, quick start needed
│   一般的な文書、クイックスタートが必要
│   → Use Simple RAG
│
├── 🏥 Domain-specific with technical terms  
│   技術用語を含む専門分野
│   → Use Semantic RAG
│
└── 🔬 Complex documents with relationships
    関係性を含む複雑な文書
    → Use Knowledge RAG
```

## Quick Start Example / クイックスタート例

This example demonstrates the simplest way to create a corpus using the **Simple RAG** approach:

この例では、**Simple RAG**アプローチを使用してコーパスを作成する最もシンプルな方法を示します：

```python
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore

# 🗄️ STEP 1: Initialize storage components
#    ステップ1: ストレージコンポーネントの初期化
doc_store = SQLiteDocumentStore("documents.db")        # Stores document metadata / 文書メタデータを保存
vector_store = InMemoryVectorStore()                   # Stores vector embeddings / ベクトル埋め込みを保存

# 🔧 STEP 2: Create Simple RAG corpus manager
#    ステップ2: Simple RAGコーパスマネージャーの作成
#    This implements: Load → Chunk → Vector pipeline
#    これは以下を実装: 読み込み → チャンク化 → ベクトル化 パイプライン
manager = CorpusManager.create_simple_rag(doc_store, vector_store)

# 📄 STEP 3: Process documents and build corpus
#    ステップ3: 文書を処理してコーパスを構築
stats = manager.build_corpus(["documents/"])

# 📊 STEP 4: Review processing results
#    ステップ4: 処理結果の確認
print(f"✅ Processed {stats.total_files_processed} files")
print(f"✅ Created {stats.total_chunks_created} chunks")
```

**What happened behind the scenes? / 内部で何が起こったのか？**
1. **File loading** - Documents were parsed and converted to internal format / **ファイル読み込み** - 文書が解析され内部形式に変換
2. **Chunking** - Large texts were split into optimal-sized pieces / **チャンク化** - 大きなテキストが最適なサイズに分割
3. **Vectorization** - Each chunk was converted to semantic embeddings / **ベクトル化** - 各チャンクがセマンティック埋め込みに変換
4. **Storage** - Results were saved to the database and vector store / **保存** - 結果がデータベースとベクトルストアに保存

## Implementation Examples / 実装例

Now let's see how each approach translates the concepts into working code:

各アプローチが概念をどのように動作するコードに変換するかを見てみましょう：

### 1.1 Simple RAG Implementation / Simple RAG実装
**Concept**: Quick prototyping with minimal processing / **概念**: 最小処理でのクイックプロトタイピング

```python
# 🎯 CONCEPT MAPPING: Simple RAG = Load → Chunk → Vector
#    概念マッピング: Simple RAG = 読み込み → チャンク化 → ベクトル化

simple_manager = CorpusManager.create_simple_rag(doc_store, vector_store)
simple_stats = simple_manager.build_corpus(file_paths)

print(f"Simple RAG Results:")
print(f"- Files: {simple_stats.total_files_processed}")      # 📄 Load stage output / 読み込みステージ出力
print(f"- Chunks: {simple_stats.total_chunks_created}")      # ✂️ Chunk stage output / チャンクステージ出力  
print(f"- Time: {simple_stats.total_processing_time:.3f}s")  # ⏱️ Processing efficiency / 処理効率
```

**When to use this implementation / この実装を使用するタイミング:**
- ✅ Testing new document collections / 新しい文書コレクションのテスト
- ✅ Rapid prototyping / 高速プロトタイピング
- ✅ Simple content without domain-specific terminology / 専門用語のないシンプルなコンテンツ

### 1.2 Semantic RAG Implementation / Semantic RAG実装
**Concept**: Enhanced understanding through terminology normalization / **概念**: 用語正規化による理解力向上

```python
# 🎯 CONCEPT MAPPING: Semantic RAG = Load → Dictionary → Normalize → Chunk → Vector
#    概念マッピング: Semantic RAG = 読み込み → 辞書 → 正規化 → チャンク化 → ベクトル化

semantic_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
semantic_stats = semantic_manager.build_corpus(file_paths)

print(f"Semantic RAG Results:")
print(f"- Files: {semantic_stats.total_files_processed}")           # 📄 Load stage output
print(f"- Dictionary terms: {semantic_stats.dictionary_terms}")     # 📚 Dictionary stage output  
print(f"- Normalized chunks: {semantic_stats.total_chunks_created}") # 🔄 Normalize + Chunk output
print(f"- Better semantic consistency achieved")                     # 🎯 Core benefit
```

**What the additional processing achieves / 追加処理で達成されること:**
1. **Dictionary creation** - Domain-specific terms are automatically extracted / **辞書作成** - 専門分野の用語が自動抽出
2. **Normalization** - Terminology variations are unified (e.g., "AI" = "Artificial Intelligence") / **正規化** - 用語のバリエーションが統一（例: "AI" = "Artificial Intelligence"）
3. **Enhanced search** - Better matching between queries and content / **検索向上** - クエリとコンテンツ間のマッチング向上

**When to use this implementation / この実装を使用するタイミング:**
- ✅ Medical, legal, or technical documents / 医療・法律・技術文書
- ✅ Content with many abbreviations / 略語が多いコンテンツ
- ✅ Multi-language or mixed terminology / 多言語や混合用語

### 1.3 Knowledge RAG Implementation / Knowledge RAG実装
**Concept**: Advanced analytics with relationship discovery / **概念**: 関係発見による高度な分析

```python
# 🎯 CONCEPT MAPPING: Knowledge RAG = Load → Dictionary → Graph → Normalize → Chunk → Vector
#    概念マッピング: Knowledge RAG = 読み込み → 辞書 → グラフ → 正規化 → チャンク化 → ベクトル化

knowledge_manager = CorpusManager.create_knowledge_rag(doc_store, vector_store)
knowledge_stats = knowledge_manager.build_corpus(file_paths)

print(f"Knowledge RAG Results:")
print(f"- Files: {knowledge_stats.total_files_processed}")          # 📄 Load stage output
print(f"- Dictionary terms: {knowledge_stats.dictionary_terms}")    # 📚 Dictionary stage output
print(f"- Relationships: {knowledge_stats.graph_relationships}")    # 🕸️ Graph stage output
print(f"- Knowledge chunks: {knowledge_stats.total_chunks_created}") # 🧠 Final enriched chunks
```

**What the complete pipeline achieves / 完全パイプラインで達成されること:**
1. **Relationship extraction** - Entities and their connections are identified / **関係抽出** - エンティティとその接続が特定
2. **Knowledge graph** - Structured representation of domain knowledge / **知識グラフ** - ドメイン知識の構造化表現
3. **Enhanced reasoning** - Better understanding of context and implications / **推論向上** - コンテキストと含意の理解向上

**When to use this implementation / この実装を使用するタイミング:**
- ✅ Research documents with complex relationships / 複雑な関係性を持つ研究文書
- ✅ Business processes and workflows / ビジネスプロセスとワークフロー
- ✅ Educational content requiring deep understanding / 深い理解が必要な教育コンテンツ

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