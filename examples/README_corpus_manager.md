# CorpusManager Tutorial and Examples

CorpusManagerを使用した文書処理パイプラインの完全なチュートリアルとサンプル集です。

## Overview / 概要

CorpusManagerは、文書の読み込みから埋め込み生成・保存まで、完全なRAG（Retrieval-Augmented Generation）パイプラインを提供します。

**完全なワークフロー：**
```
📁 Documents → 🔄 Processing → ✂️ Chunking → 🔤 Embedding → 🗄️ Storage → 🔍 Search
```

## Quick Start / クイックスタート

### 1. 最速で始める (30秒)

```bash
python examples/quickstart_guide.py
```

**または最小限のコード：**

```python
from refinire_rag import CorpusManager

# 1行でセットアップ
corpus_manager = CorpusManager()

# ディレクトリ全体を処理
results = corpus_manager.process_corpus("/path/to/your/documents")

# 検索テスト
search_results = corpus_manager.search_documents("your query")
```

### 2. 完全チュートリアル (15分)

```bash
python examples/corpus_manager_tutorial.py
```

5つのステップで段階的に学習：
- Step 1: 基本的な使用法
- Step 2: カスタム設定
- Step 3: 永続ストレージ
- Step 4: 高度な機能
- Step 5: 本番環境での例

## Examples Overview / サンプル概要

### 📖 Tutorial Files / チュートリアルファイル

| ファイル | 内容 | 実行時間 | レベル |
|---------|------|----------|--------|
| `quickstart_guide.py` | 3つのクイック例 | 3分 | 初心者 |
| `corpus_manager_tutorial.py` | 完全チュートリアル | 15分 | 中級 |
| `document_processor_example.py` | 処理パイプライン詳細 | 5分 | 中級 |
| `embedding_example.py` | 埋め込み比較 | 5分 | 中級 |

### 🎯 What You'll Learn / 学習内容

**基礎編：**
- ✅ 文書の自動読み込み（複数形式対応）
- ✅ テキストチャンキング（検索最適化）
- ✅ 埋め込み生成（TF-IDF、OpenAI）
- ✅ データベース保存（SQLite）
- ✅ 基本的な検索機能

**応用編：**
- ✅ カスタム処理パイプライン
- ✅ バッチ処理と進捗監視
- ✅ エラーハンドリング戦略
- ✅ パフォーマンス最適化
- ✅ 本番環境デプロイ

## Step-by-Step Guide / ステップバイステップガイド

### Step 1: Basic Usage / 基本使用法

```python
from refinire_rag import CorpusManager

# デフォルト設定で開始
corpus_manager = CorpusManager()

# 文書処理（ファイルまたはディレクトリ）
results = corpus_manager.process_corpus("path/to/documents")

# 結果確認
print(f"処理完了: {results['documents_loaded']} 文書")
print(f"埋め込み生成: {results['documents_embedded']} 文書")
```

### Step 2: Custom Configuration / カスタム設定

```python
from refinire_rag import (
    CorpusManager, CorpusManagerConfig, 
    ChunkingConfig, TFIDFEmbedder, TFIDFEmbeddingConfig
)

# カスタム設定
config = CorpusManagerConfig(
    # チャンキング設定
    chunking_config=ChunkingConfig(
        chunk_size=256,      # チャンクサイズ
        overlap=32,          # オーバーラップ
        split_by_sentence=True  # 文境界で分割
    ),
    
    # 埋め込み設定
    embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
        max_features=5000,   # 語彙サイズ
        ngram_range=(1, 2),  # 1-gram, 2-gram
        min_df=2             # 最小文書頻度
    )),
    
    # 処理オプション
    batch_size=20,
    enable_progress_reporting=True
)

corpus_manager = CorpusManager(config)
results = corpus_manager.process_corpus("documents/")
```

### Step 3: Persistent Storage / 永続ストレージ

```python
from refinire_rag import SQLiteDocumentStore

# データベース設定
db_path = "corpus_database.db"
document_store = SQLiteDocumentStore(db_path)

config = CorpusManagerConfig(
    document_store=document_store,
    store_intermediate_results=True,  # 中間結果も保存
    
    # モデル永続化
    embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
        model_path="tfidf_model.pkl",
        auto_save_model=True
    ))
)

corpus_manager = CorpusManager(config)
results = corpus_manager.process_corpus("documents/")

# 後でデータベースとモデルを再利用可能
```

### Step 4: Production Setup / 本番設定

```python
# 本番環境用の堅牢な設定
config = CorpusManagerConfig(
    # 最適化されたチャンキング
    chunking_config=ChunkingConfig(
        chunk_size=384,
        overlap=64,
        split_by_sentence=True,
        min_chunk_size=100,
        max_chunk_size=600
    ),
    
    # 高性能埋め込み
    embedder=TFIDFEmbedder(TFIDFEmbeddingConfig(
        max_features=10000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 3),
        model_path="production_model.pkl"
    )),
    
    # 堅牢なエラーハンドリング
    fail_on_error=False,
    max_errors=100,
    
    # パフォーマンス設定
    batch_size=50,
    enable_progress_reporting=True
)
```

## Configuration Options / 設定オプション

### ChunkingConfig / チャンキング設定

```python
ChunkingConfig(
    chunk_size=512,           # チャンクサイズ（トークン数）
    overlap=50,               # オーバーラップ（トークン数）
    split_by_sentence=True,   # 文境界で分割
    min_chunk_size=50,        # 最小チャンクサイズ
    max_chunk_size=1024       # 最大チャンクサイズ
)
```

### TFIDFEmbeddingConfig / TF-IDF設定

```python
TFIDFEmbeddingConfig(
    max_features=10000,       # 最大語彙数
    min_df=2,                 # 最小文書頻度
    max_df=0.95,              # 最大文書頻度（比率）
    ngram_range=(1, 2),       # N-gramの範囲
    remove_stopwords=True,    # ストップワード除去
    model_path="model.pkl",   # モデル保存パス
    auto_save_model=True      # 自動保存
)
```

### CorpusManagerConfig / 全体設定

```python
CorpusManagerConfig(
    enable_chunking=True,               # チャンキング有効化
    enable_embedding=True,              # 埋め込み有効化
    store_intermediate_results=True,    # 中間結果保存
    batch_size=100,                     # バッチサイズ
    enable_progress_reporting=True,     # 進捗レポート
    fail_on_error=False,                # エラー時継続
    max_errors=10                       # 最大エラー数
)
```

## Performance Guidelines / パフォーマンスガイドライン

### 推奨設定

**小規模（< 1,000文書）：**
```python
chunk_size=256, batch_size=10, max_features=1000
```

**中規模（1,000-10,000文書）：**
```python
chunk_size=384, batch_size=50, max_features=5000
```

**大規模（> 10,000文書）：**
```python
chunk_size=512, batch_size=100, max_features=10000
```

### パフォーマンス最適化

1. **バッチサイズ調整**：メモリと速度のバランス
2. **チャンクサイズ最適化**：検索精度と処理速度
3. **語彙サイズ制限**：メモリ使用量とモデル品質
4. **進捗監視**：長時間処理の可視化
5. **エラーハンドリング**：堅牢性の確保

## Supported File Formats / 対応ファイル形式

| 形式 | 拡張子 | 説明 |
|------|--------|------|
| Text | `.txt` | プレーンテキスト |
| Markdown | `.md`, `.markdown` | Markdown形式 |
| JSON | `.json` | JSON構造化データ |
| CSV | `.csv` | カンマ区切り値 |
| PDF | `.pdf` | PDF文書（要追加ライブラリ） |
| HTML | `.html`, `.htm` | HTML文書 |

## Error Handling / エラーハンドリング

### 設定例

```python
# 優雅なエラー処理（推奨）
config = CorpusManagerConfig(
    fail_on_error=False,    # エラーで停止しない
    max_errors=100          # 最大100エラーまで許容
)

# 厳密なエラー処理
config = CorpusManagerConfig(
    fail_on_error=True      # 最初のエラーで停止
)
```

### 一般的なエラーと対処法

1. **ファイル読み込みエラー**：`ignore_errors=True`
2. **埋め込み生成エラー**：`fail_on_error=False`
3. **メモリ不足**：`batch_size`を小さく
4. **語彙不足**：`min_df`を小さく

## Monitoring and Statistics / 監視と統計

### 処理結果の確認

```python
results = corpus_manager.process_corpus("documents/")

print(f"成功: {results['success']}")
print(f"読み込み: {results['documents_loaded']}")
print(f"処理済み: {results['documents_processed']}")
print(f"埋め込み: {results['documents_embedded']}")
print(f"保存済み: {results['documents_stored']}")
print(f"エラー: {results['total_errors']}")
print(f"処理時間: {results['total_processing_time']:.2f}秒")
```

### 詳細統計

```python
stats = corpus_manager.get_corpus_stats()

# 埋め込み統計
embedder_stats = stats.get('embedder_stats', {})
print(f"平均処理時間: {embedder_stats.get('average_processing_time', 0):.4f}秒")
print(f"キャッシュヒット率: {embedder_stats.get('cache_hit_rate', 0):.1%}")

# パイプライン統計
pipeline_stats = stats.get('pipeline_stats', {})
for processor, proc_stats in pipeline_stats.get('processor_stats', {}).items():
    print(f"{processor}: {proc_stats.get('total_time', 0):.3f}秒")
```

## Advanced Features / 高度な機能

### OpenAI埋め込み使用

```python
from refinire_rag import OpenAIEmbedder, OpenAIEmbeddingConfig

# OpenAI埋め込み設定（API Key必要）
embedder = OpenAIEmbedder(OpenAIEmbeddingConfig(
    model_name="text-embedding-3-small",
    batch_size=10
))

config = CorpusManagerConfig(embedder=embedder)
corpus_manager = CorpusManager(config)
```

### カスタム処理パイプライン

```python
from examples.custom_processor_example import (
    TextNormalizationProcessor,
    DocumentEnricher
)

config = CorpusManagerConfig(
    processors=[
        TextNormalizationProcessor(),  # テキスト正規化
        DocumentEnricher()             # メタデータ追加
    ]
)
```

### 検索とフィルタリング

```python
# 基本検索
results = corpus_manager.search_documents("machine learning", limit=5)

# 文書系譜の追跡
lineage = corpus_manager.get_document_lineage("document_id")

# 処理段階別の文書取得
chunked_docs = document_store.search_by_metadata({
    "processing_stage": "chunked"
})
```

## Troubleshooting / トラブルシューティング

### よくある問題

**1. 処理が遅い**
```python
# バッチサイズを大きく
config.batch_size = 100

# チャンクサイズを大きく
config.chunking_config.chunk_size = 512
```

**2. メモリ不足**
```python
# バッチサイズを小さく
config.batch_size = 10

# 語彙サイズを制限
config.embedder.config.max_features = 1000
```

**3. 検索結果が少ない**
```python
# 最小文書頻度を下げる
config.embedder.config.min_df = 1

# チャンクサイズを小さく
config.chunking_config.chunk_size = 128
```

**4. エラーが多発**
```python
# エラー許容設定
config.fail_on_error = False
config.max_errors = 1000

# ファイル読み込みエラーを無視
config.loading_config.ignore_errors = True
```

## Next Steps / 次のステップ

1. **QueryEngine実装**：セマンティック検索機能
2. **ベクトルデータベース**：高速類似度検索
3. **評価システム**：RAG品質測定
4. **スケーリング**：大規模文書処理
5. **本番デプロイ**：プロダクション環境

## Examples Summary / サンプル要約

| 機能 | 例 | ファイル |
|------|----|---------| 
| 基本使用法 | 最小設定での処理 | `quickstart_guide.py` |
| カスタム設定 | 詳細パラメータ調整 | `corpus_manager_tutorial.py` |
| 永続ストレージ | データベース保存 | `corpus_manager_tutorial.py` |
| 本番環境 | プロダクション設定 | `corpus_manager_tutorial.py` |
| カスタム処理 | 独自プロセッサ | `custom_processor_example.py` |
| 埋め込み比較 | TF-IDF vs OpenAI | `embedding_example.py` |

**🚀 今すぐ始める：**
```bash
python examples/quickstart_guide.py
```

**📚 詳しく学ぶ：**
```bash
python examples/corpus_manager_tutorial.py
```