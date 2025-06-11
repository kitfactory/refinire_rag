# CorpusManager - コーパス管理

ドキュメントコーパスの構築と管理を行うユースケースクラスです。

## 概要

CorpusManagerは、RAGシステムのためのドキュメントコーパスを構築する3つのアプローチを提供します：

1. **プリセット設定** - 事前定義されたパイプライン
2. **ステージ選択** - カスタムステージの組み合わせ
3. **カスタムパイプライン** - 完全にカスタマイズされたパイプライン

```python
from refinire_rag.use_cases.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore

# ストレージの初期化
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# CorpusManagerの作成
corpus_manager = CorpusManager(doc_store, vector_store)
```

## プリセット設定

### create_simple_rag

最もシンプルなRAGパイプラインを作成します。

```python
# Simple RAG: Load → Chunk → Vector
manager = CorpusManager.create_simple_rag(doc_store, vector_store)

# コーパス構築
stats = manager.build_corpus(["doc1.txt", "doc2.txt"])
```

### create_semantic_rag

正規化機能を含むセマンティックRAGパイプラインを作成します。

```python
# Semantic RAG: Load → Dictionary → Normalize → Chunk → Vector
manager = CorpusManager.create_semantic_rag(doc_store, vector_store)

# デフォルト設定の上書き
manager = CorpusManager.create_semantic_rag(
    doc_store, 
    vector_store,
    chunk_size=300,
    dictionary_path="custom_dict.md"
)
```

### create_knowledge_rag

知識グラフを含む高度なRAGパイプラインを作成します。

```python
# Knowledge RAG: Load → Dictionary → Graph → Normalize → Chunk → Vector
manager = CorpusManager.create_knowledge_rag(doc_store, vector_store)
```

## ステージ選択アプローチ

個別ステージを選択してカスタムパイプラインを構築します。

```python
# 利用可能なステージ
stages = ["load", "dictionary", "normalize", "graph", "chunk", "vector"]

# ステージ別設定
stage_configs = {
    "loader_config": LoaderConfig(),
    "dictionary_config": DictionaryMakerConfig(
        dictionary_file_path="domain_dict.md",
        focus_on_technical_terms=True
    ),
    "normalizer_config": NormalizerConfig(
        dictionary_file_path="domain_dict.md",
        whole_word_only=False
    ),
    "chunker_config": ChunkingConfig(
        chunk_size=500,
        overlap=50
    ),
    "vector_config": VectorStoreProcessorConfig(
        embedder_type="tfidf"
    )
}

# コーパス構築
stats = corpus_manager.build_corpus(
    file_paths=["doc1.txt", "doc2.txt"],
    stages=["load", "normalize", "chunk", "vector"],
    stage_configs=stage_configs
)
```

## カスタムパイプライン

完全にカスタマイズされたパイプラインを使用します。

```python
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.processing.normalizer import Normalizer
from refinire_rag.processing.chunker import Chunker

# カスタムパイプラインの作成
custom_pipelines = [
    # パイプライン1: 原文処理
    DocumentPipeline([
        TextLoader(config),
        DocumentStoreProcessor(doc_store)
    ]),
    
    # パイプライン2: 正規化
    DocumentPipeline([
        DocumentStoreLoader(doc_store),
        Normalizer(norm_config),
        DocumentStoreProcessor(doc_store)
    ]),
    
    # パイプライン3: チャンキングとベクトル化
    DocumentPipeline([
        DocumentStoreLoader(doc_store),
        Chunker(chunk_config),
        VectorStoreProcessor(vector_store, vector_config)
    ])
]

# コーパス構築
stats = corpus_manager.build_corpus(
    file_paths=["doc1.txt", "doc2.txt"],
    custom_pipelines=custom_pipelines
)
```

## メソッド

### build_corpus

コーパスを構築します。

```python
def build_corpus(
    self,
    file_paths: List[str],
    preset: Optional[str] = None,
    stages: Optional[List[str]] = None,
    stage_configs: Optional[Dict[str, Any]] = None,
    custom_pipelines: Optional[List[DocumentPipeline]] = None
) -> ProcessingStats:
    """
    コーパスを構築する
    
    Args:
        file_paths: 処理するファイルパスのリスト
        preset: プリセット名 ("simple", "semantic", "knowledge")
        stages: 実行するステージのリスト
        stage_configs: ステージ別の設定
        custom_pipelines: カスタムパイプラインのリスト
        
    Returns:
        ProcessingStats: 処理統計
    """
```

### get_corpus_stats

コーパスの統計情報を取得します。

```python
stats = corpus_manager.get_corpus_stats()
print(f"総ドキュメント数: {stats['total_documents']}")
print(f"総チャンク数: {stats['total_chunks']}")
print(f"総ベクトル数: {stats['total_vectors']}")
```

### clear_corpus

コーパスをクリアします。

```python
# 全データをクリア
corpus_manager.clear_corpus()

# 特定のステージのみクリア
corpus_manager.clear_corpus(stages=["vector"])
```

## ProcessingStats

処理結果の統計情報です。

```python
class ProcessingStats:
    total_documents_created: int  # 作成されたドキュメント総数
    total_chunks_created: int     # 作成されたチャンク総数
    total_processing_time: float  # 総処理時間（秒）
    documents_by_stage: Dict[str, int]  # ステージ別ドキュメント数
    errors_encountered: int       # エラー数
    pipeline_stages_executed: int # 実行されたステージ数
    metadata: Dict[str, Any]      # 追加メタデータ
```

## エラーハンドリング

```python
try:
    stats = corpus_manager.build_corpus(file_paths)
except FileNotFoundError as e:
    print(f"ファイルが見つかりません: {e}")
except ProcessingError as e:
    print(f"処理エラー: {e}")
    print(f"失敗したステージ: {e.stage}")
    print(f"失敗したドキュメント: {e.document_id}")
```

## ベストプラクティス

1. **小規模コーパス**: Simple RAGから始める
2. **日本語コーパス**: Semantic RAGで正規化を活用
3. **専門分野**: カスタム辞書とKnowledge RAGを使用
4. **大規模処理**: バッチ処理とエラーハンドリングを実装

```python
# バッチ処理の例
batch_size = 100
for i in range(0, len(all_files), batch_size):
    batch = all_files[i:i+batch_size]
    stats = corpus_manager.build_corpus(batch)
    print(f"バッチ {i//batch_size + 1} 完了: {stats.total_documents_created} 文書")
```