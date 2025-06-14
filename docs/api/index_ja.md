# refinire-rag API リファレンス

refinire-ragライブラリのAPIリファレンスドキュメントです。

## パッケージ構成

### コアモジュール

- [models](models.md) - データモデル定義
- [processing](processing.md) - ドキュメント処理パイプライン
- [storage](storage.md) - ストレージインターフェース
- [embedding](embedding.md) - 埋め込み生成
- [retrieval](retrieval.md) - 検索と回答生成
- [loaders](loaders.md) - ファイルローダー

### アプリケーションクラス

- [CorpusManager](corpus_manager.md) - コーパス管理
- [QueryEngine](query_engine.md) - クエリ処理エンジン
- [QualityLab](quality_lab.md) - 品質評価（予定）

## クイックリファレンス

### 基本的な使い方

```python
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore

# ストレージの初期化
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# Simple RAGの作成
corpus_manager = CorpusManager.create_simple_rag(doc_store, vector_store)

# コーパスの構築
stats = corpus_manager.build_corpus(["document1.txt", "document2.txt"])

# クエリエンジンの作成
query_engine = QueryEngine(
    document_store=doc_store,
    vector_store=vector_store,
    retriever=retriever,
    reader=reader
)

# 質問応答
result = query_engine.answer("RAGとは何ですか？")
```

### 高度な使い方

```python
# カスタムパイプラインの構築
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.processing.normalizer import Normalizer
from refinire_rag.processing.chunker import Chunker

pipeline = DocumentPipeline([
    Normalizer(config),
    Chunker(config)
])

# ステージ選択によるコーパス構築
stats = corpus_manager.build_corpus(
    file_paths=files,
    stages=["load", "dictionary", "normalize", "chunk", "vector"],
    stage_configs={
        "normalizer_config": NormalizerConfig(...),
        "chunker_config": ChunkingConfig(...)
    }
)
```

## 設計原則

1. **DocumentProcessorパターン**: 全ての処理はDocumentProcessorインターフェースを実装
2. **パイプライン構成**: 処理をパイプラインとして組み合わせ可能
3. **設定の外部化**: 全ての設定はConfigクラスで管理
4. **型安全性**: Pydanticを使用した型定義

## バージョン情報

- 現在のバージョン: 0.1.0
- Python要件: 3.10以上