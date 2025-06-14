# DocumentStoreLoader使用ガイド

## 概要

DocumentStoreLoaderは、既存のDocumentStoreから文書をロードするためのLoaderクラスです。様々なロード戦略と堅牢なエラーハンドリングを提供します。

## 基本的な使用方法

### 1. フルロード（全文書をロード）

```python
from refinire_rag.loader.document_store_loader import DocumentStoreLoader, DocumentLoadConfig, LoadStrategy
from refinire_rag.storage.document_store import DocumentStore

# DocumentStoreの準備
document_store = YourDocumentStore()

# フルロード設定
config = DocumentLoadConfig(strategy=LoadStrategy.FULL)
loader = DocumentStoreLoader(document_store, config)

# 全文書をロード
result = loader.load_all()
print(f"ロード済み: {result.loaded_count}, エラー: {result.error_count}")

# またはDocumentProcessorインターフェースで
for document in loader.process([]):
    print(f"文書ID: {document.id}")
```

### 2. フィルタードロード（条件に基づくロード）

```python
from datetime import datetime, timedelta

# メタデータフィルター
config = DocumentLoadConfig(
    strategy=LoadStrategy.FILTERED,
    metadata_filters={"type": "research_paper", "status": "published"},
    max_documents=100
)

# 内容検索
config = DocumentLoadConfig(
    strategy=LoadStrategy.FILTERED,
    content_query="machine learning",
    max_documents=50
)

loader = DocumentStoreLoader(document_store, config)
documents = list(loader.process([]))
```

### 3. インクリメンタルロード（日時ベース）

```python
# 最近1週間に更新された文書
config = DocumentLoadConfig(
    strategy=LoadStrategy.INCREMENTAL,
    modified_after=datetime.now() - timedelta(days=7),
    batch_size=50
)

loader = DocumentStoreLoader(document_store, config)
result = loader.load_all()
```

### 4. 特定IDロード

```python
# 特定の文書IDのみロード
config = DocumentLoadConfig(
    strategy=LoadStrategy.ID_LIST,
    document_ids=["doc1", "doc2", "doc3"],
    validate_documents=True
)

loader = DocumentStoreLoader(document_store, config)
documents = list(loader.process([]))
```

### 5. ページングロード

```python
# 大量データを分割処理
config = DocumentLoadConfig(
    strategy=LoadStrategy.PAGINATED,
    batch_size=100,
    max_documents=1000,
    sort_by="created_at",
    sort_order="desc"
)

loader = DocumentStoreLoader(document_store, config)

# バッチごとに処理
for document in loader.process([]):
    # 1つずつ処理
    process_document(document)
```

## 高度な機能

### エラーハンドリング

```python
from refinire_rag.exceptions import DocumentStoreError, LoaderError, ConfigurationError

try:
    config = DocumentLoadConfig(
        strategy=LoadStrategy.FILTERED,
        metadata_filters={"date": {"$gte": "2024-01-01"}},
        validate_documents=True
    )
    
    loader = DocumentStoreLoader(document_store, config)
    result = loader.load_all()
    
    if result.has_errors:
        print("エラーが発生しました:")
        for error in result.errors:
            print(f"  - {error}")
    
    print(f"成功率: {result.success_rate:.2%}")
    
except ConfigurationError as e:
    print(f"設定エラー: {e}")
except DocumentStoreError as e:
    print(f"ストレージエラー: {e}")
except LoaderError as e:
    print(f"ローダーエラー: {e}")
```

### 文書数の事前確認

```python
loader = DocumentStoreLoader(document_store, config)

# ロード対象文書数を事前確認
count = loader.count_matching_documents()
print(f"ロード予定文書数: {count}")

# ローダーの設定サマリー
summary = loader.get_load_summary()
print(f"戦略: {summary['strategy']}")
print(f"バッチサイズ: {summary['batch_size']}")
print(f"フィルター有無: {summary['has_metadata_filters']}")
```

## 設定オプション

### DocumentLoadConfig のパラメータ

| パラメータ | デフォルト | 説明 |
|------------|-----------|------|
| `strategy` | `LoadStrategy.FULL` | ロード戦略 |
| `metadata_filters` | `None` | メタデータフィルター辞書 |
| `content_query` | `None` | 内容検索クエリ |
| `document_ids` | `None` | 特定文書IDリスト |
| `modified_after` | `None` | この日時以降に更新された文書 |
| `modified_before` | `None` | この日時以前に更新された文書 |
| `batch_size` | `100` | バッチサイズ |
| `max_documents` | `None` | 最大ロード文書数 |
| `sort_by` | `"created_at"` | ソートフィールド |
| `sort_order` | `"desc"` | ソート順序（"asc"/"desc"） |
| `validate_documents` | `True` | 文書検証の有効/無効 |

### LoadStrategy の選択肢

| 戦略 | 説明 | 使用場面 |
|------|------|---------|
| `FULL` | 全文書をロード | 初期データロード、バックアップ |
| `FILTERED` | フィルター条件でロード | 特定条件の文書のみ処理 |
| `INCREMENTAL` | 日時ベースでロード | 増分更新、定期処理 |
| `ID_LIST` | 特定IDでロード | 個別文書の再処理 |
| `PAGINATED` | ページング付きロード | 大量データの分割処理 |

## ベストプラクティス

### 1. 大量データの処理

```python
# メモリ効率的な処理
config = DocumentLoadConfig(
    strategy=LoadStrategy.PAGINATED,
    batch_size=100,  # 適切なバッチサイズ
    validate_documents=False  # 高速化のため検証を無効化
)

loader = DocumentStoreLoader(document_store, config)

# イテレーターで逐次処理
for document in loader.process([]):
    try:
        # 文書処理
        result = process_document(document)
        save_result(result)
    except Exception as e:
        logger.error(f"文書処理エラー {document.id}: {e}")
```

### 2. エラー処理とログ

```python
import logging

logger = logging.getLogger(__name__)

def safe_load_documents(document_store, config):
    try:
        loader = DocumentStoreLoader(document_store, config)
        result = loader.load_all()
        
        logger.info(f"ロード完了: {result.get_summary()}")
        
        if result.has_errors:
            logger.warning(f"エラーが {result.error_count} 件発生")
            for error in result.errors[-5:]:  # 最新5件のエラーをログ
                logger.error(f"ロードエラー: {error}")
        
        return result
        
    except Exception as e:
        logger.error(f"致命的エラー: {e}")
        raise
```

### 3. 設定の検証

```python
def create_validated_loader(document_store, **kwargs):
    """検証済みローダーを作成"""
    try:
        config = DocumentLoadConfig(**kwargs)
        config.validate()  # 事前検証
        
        loader = DocumentStoreLoader(document_store, config)
        
        # 接続テスト
        count = loader.count_matching_documents()
        if count == 0:
            logger.warning("ロード対象文書が見つかりません")
        else:
            logger.info(f"ロード対象: {count} 文書")
        
        return loader
        
    except Exception as e:
        logger.error(f"ローダー作成失敗: {e}")
        raise
```

## トラブルシューティング

### よくあるエラーと解決方法

1. **ConfigurationError**: 設定が無効
   ```python
   # ID_LIST戦略でdocument_idsが未指定
   config = DocumentLoadConfig(
       strategy=LoadStrategy.ID_LIST,
       document_ids=["doc1", "doc2"]  # 必須
   )
   ```

2. **ValidationError**: 文書検証失敗
   ```python
   # 検証を無効化するか、データを修正
   config = DocumentLoadConfig(validate_documents=False)
   ```

3. **DocumentStoreError**: ストレージエラー
   ```python
   # 接続やアクセス権限を確認
   # リトライロジックを実装
   ```

DocumentStoreLoaderを使用することで、様々なシナリオでDocumentStoreから効率的かつ安全に文書をロードできます。