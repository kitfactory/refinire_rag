# 統合アーキテクチャ移行ガイド

本ガイドでは、refinire-ragの統合アーキテクチャ（2024年12月アップデート）への移行方法を説明します。

## 変更の概要

### 主な変更点

1. **VectorStoreProcessor廃止**: ラッパークラスを削除
2. **VectorStore統合**: DocumentProcessor + Indexer + Retriever を統合実装
3. **KeywordSearch統合**: DocumentProcessor + Indexer + Retriever を統合実装
4. **直接パイプライン使用**: ストアを直接DocumentPipelineで使用可能

### アーキテクチャの比較

#### 従来のアーキテクチャ（2024年11月以前）
```
DocumentPipeline → VectorStoreProcessor → VectorStore
                        ↑ ラッパークラス
```

#### 新しい統合アーキテクチャ（2024年12月以降）
```
DocumentPipeline → VectorStore (DocumentProcessor + Indexer + Retriever)
                        ↑ 統合されたクラス
```

## 移行手順

### 1. インポートの更新

#### 従来のコード
```python
from refinire_rag.processing.vector_store_processor import VectorStoreProcessor
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
```

#### 新しいコード
```python
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
# VectorStoreProcessorのインポートは不要
```

### 2. VectorStoreの初期化

#### 従来のコード
```python
# ストアとプロセッサーを別々に作成
vector_store = InMemoryVectorStore()
embedder = OpenAIEmbedder(config)
processor = VectorStoreProcessor(vector_store, config)
processor.set_embedder(embedder)

# パイプラインにプロセッサーを追加
pipeline = DocumentPipeline([processor])
```

#### 新しいコード
```python
# ストアを直接使用
vector_store = InMemoryVectorStore()
embedder = OpenAIEmbedder(config)
vector_store.set_embedder(embedder)  # 直接設定

# パイプラインにストアを直接追加
pipeline = DocumentPipeline([vector_store])
```

### 3. CorpusManagerでの使用

#### 従来のコード
```python
# VectorStoreProcessorを経由
vector_store = InMemoryVectorStore()
processor = VectorStoreProcessor(vector_store)

corpus_manager = CorpusManager(
    document_store=doc_store,
    vector_store=vector_store,
    processors=[processor]  # プロセッサーを指定
)
```

#### 新しいコード
```python
# VectorStoreを直接使用
vector_store = InMemoryVectorStore()
vector_store.set_embedder(embedder)

corpus_manager = CorpusManager(
    document_store=doc_store,
    vector_store=vector_store  # 直接指定、プロセッサー不要
)
```

### 4. カスタムパイプラインの構築

#### 従来のコード
```python
# 複雑な構成が必要
vector_store = InMemoryVectorStore()
processor = VectorStoreProcessor(vector_store, config)

pipeline = DocumentPipeline([
    Loader(loader_config),
    Normalizer(normalizer_config),
    Chunker(chunker_config),
    processor  # ラッパークラス
])
```

#### 新しいコード
```python
# シンプルな構成
vector_store = InMemoryVectorStore()
vector_store.set_embedder(embedder)

pipeline = DocumentPipeline([
    Loader(loader_config),
    Normalizer(normalizer_config),
    Chunker(chunker_config),
    vector_store  # 直接使用
])
```

### 5. 検索機能の使用

#### 従来のコード
```python
# Retrieverを別途作成
vector_store = InMemoryVectorStore()
retriever = SimpleRetriever(vector_store, embedder)

# QueryEngineで使用
query_engine = QueryEngine(
    retriever=retriever,
    vector_store=vector_store
)
```

#### 新しいコード
```python
# VectorStore自体がRetrieverインターフェースを実装
vector_store = InMemoryVectorStore()
vector_store.set_embedder(embedder)

# 直接検索可能
results = vector_store.retrieve("search query", limit=5)

# QueryEngineでの使用
query_engine = QueryEngine(
    retriever=vector_store,  # VectorStore自体を使用
    vector_store=vector_store
)
```

## 新機能の活用

### 1. 統合されたインデックス機能

```python
vector_store = InMemoryVectorStore()
vector_store.set_embedder(embedder)

# 単一文書のインデックス
vector_store.index_document(document)

# 複数文書のバッチインデックス
vector_store.index_documents(documents)

# 文書の更新
vector_store.update_document(updated_document)

# 文書の削除
vector_store.remove_document(document_id)

# インデックスのクリア
vector_store.clear_index()

# 文書数の取得
count = vector_store.get_document_count()
```

### 2. 統合された検索機能

```python
vector_store = InMemoryVectorStore()
vector_store.set_embedder(embedder)

# テキストベース検索
results = vector_store.retrieve("search query", limit=10)

# メタデータフィルタ付き検索
results = vector_store.retrieve(
    "search query", 
    limit=10,
    metadata_filter={"category": "technology"}
)

# メタデータのみでの検索
results = vector_store.search_by_metadata(
    {"author": "John Doe"}, 
    limit=5
)
```

### 3. DocumentPipelineでの統計追跡

```python
pipeline = DocumentPipeline([chunker, vector_store])
results = pipeline.process_documents(documents)

# パイプライン統計の取得
pipeline_stats = pipeline.get_pipeline_stats()

# VectorStore固有の統計
vector_stats = vector_store.get_processing_stats()
print(f"Vectors stored: {vector_stats['vectors_stored']}")
print(f"Searches performed: {vector_stats['searches_performed']}")
```

## 互換性に関する注意

### 破壊的変更

1. **VectorStoreProcessor**: 完全に削除されました
2. **KeywordSearchProcessor**: 削除されました（元々存在していませんでした）
3. **プロセッサーラッパーパターン**: サポートされません

### 移行が必要なコード

- `VectorStoreProcessor`を使用しているすべてのコード
- `DocumentPipeline`で`VectorStoreProcessor`を使用しているコード
- カスタムプロセッサーラッパーを作成しているコード

### 後方互換性が保たれる部分

- `VectorStore`の基本的なメソッド（`add_vector`, `search_similar`など）
- `Document`およびデータモデル
- `DocumentPipeline`の基本的な動作
- 設定クラスの構造

## トラブルシューティング

### よくあるエラーと解決方法

#### エラー: `ImportError: cannot import name 'VectorStoreProcessor'`

**原因**: VectorStoreProcessorが削除されました

**解決方法**: 
```python
# 削除
from refinire_rag.processing.vector_store_processor import VectorStoreProcessor

# VectorStoreを直接使用
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
```

#### エラー: `TypeError: VectorStore object is not callable`

**原因**: VectorStoreProcessorの代わりにVectorStoreを使おうとしています

**解決方法**:
```python
# 間違い
processor = VectorStoreProcessor(vector_store)

# 正しい
vector_store = InMemoryVectorStore()
vector_store.set_embedder(embedder)
```

#### エラー: `AttributeError: 'VectorStore' object has no attribute 'process'`

**原因**: 古いVectorStoreインスタンスを使用している可能性があります

**解決方法**: 最新版のVectorStoreを確認し、DocumentProcessorを継承していることを確認してください

## パフォーマンスの改善

統合アーキテクチャにより以下の改善が期待できます：

1. **メモリ使用量削減**: ラッパークラスが不要
2. **処理速度向上**: 直接的なメソッド呼び出し
3. **コード複雑性の低下**: より少ないクラス数
4. **設定の簡素化**: 直接的な設定方法

## 移行チェックリスト

- [ ] `VectorStoreProcessor`のインポートを削除
- [ ] `VectorStore`を直接`DocumentPipeline`で使用
- [ ] `vector_store.set_embedder()`でエンベッダーを設定
- [ ] `CorpusManager`のコンストラクタを更新
- [ ] `QueryEngine`でVectorStoreを直接Retrieverとして使用
- [ ] テストコードを新しいAPIに合わせて更新
- [ ] 統計取得方法を新しいメソッドに更新

## サポート

移行に関する質問や問題がある場合は、以下のリソースを参照してください：

- [GitHub Issues](https://github.com/your-repo/refinire-rag/issues)
- [ドキュメント](./architecture.md)
- [API リファレンス](./api/)
- [チュートリアル](./tutorials/)

この移行により、より簡潔で保守しやすいコードベースが実現できます。