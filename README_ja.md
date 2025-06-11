# refinire-rag

洗練されたRAGフレームワーク - 企業グレードの文書処理を楽々に。

## 🌟 なぜrefinire-rag？

従来のRAGフレームワークは強力ですが複雑です。refinire-ragは根本的なシンプルさと企業グレードの生産性でRAG開発体験を洗練します。

**[→ なぜrefinire-rag？完全版](docs/why_refinire_rag_ja.md)** | **[→ Why refinire-rag? Complete Story](docs/why_refinire_rag.md)**

### ⚡ 10倍シンプルな開発
```python
# LangChain: 50行以上の複雑なセットアップ
# refinire-rag: 本番対応RAGを5行で
manager = CorpusManager.create_simple_rag(doc_store, vector_store)
results = manager.process_corpus(["documents/"])
answer = query_engine.answer("これはどう動きますか？")
```

### 🏢 企業対応機能が内蔵
- **増分処理**: 10,000+文書を効率的に処理
- **日本語最適化**: 内蔵言語処理
- **アクセス制御**: 部署レベルデータ分離
- **本番監視**: 包括的可観測性
- **統一アーキテクチャ**: すべてが一つのパターン

## 概要

refinire-ragは、Refinireライブラリのサブパッケージとして、RAG（Retrieval-Augmented Generation）機能を提供します。モジュラーなアーキテクチャと統一されたDocumentProcessorインターフェースで、最大限の柔軟性を実現します。

## アーキテクチャ

### ユースケースクラス（Refinire Steps）
- **CorpusManager**: 文書ローディング、正規化、チャンク化、埋め込み生成、保存
- **QueryEngine**: 文書検索、再ランキング、回答生成（Refinire Stepを継承）
- **QualityLab**: 評価データ作成、自動RAG評価、矛盾検出、レポート生成

### DocumentProcessor統一アーキテクチャ
すべての文書処理機能が統一されたベースクラスを継承：

#### 文書処理プロセッサ
- **UniversalLoader**: 拡張子ベースのファイル読み込み
- **Normalizer**: 辞書ベースの用語正規化
- **TokenBasedChunker**: トークンベースのチャンク化
- **DictionaryMaker**: 用語・略語抽出とMD辞書更新
- **GraphBuilder**: 知識グラフ構築
- **VectorStoreProcessor**: 埋め込み生成とベクトル保存

#### 品質評価プロセッサ
- **TestSuite**: 評価ランナー
- **Evaluator**: メトリクス集計
- **ContradictionDetector**: 矛盾検出
- **InsightReporter**: 洞察レポート生成

## 🚀 クイックスタート

### インストール
```bash
pip install refinire-rag
```

### 30秒RAGシステム
```python
from refinire_rag import create_simple_rag

# ワンライナー企業RAG
rag = create_simple_rag("your_documents/")
answer = rag.query("これはどう動きますか？")
print(answer)
```

### 本番対応セットアップ
```python
from refinire_rag.use_cases import CorpusManager, QueryEngine
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore

# ストレージを設定
doc_store = SQLiteDocumentStore("corpus.db")
vector_store = InMemoryVectorStore()

# 増分処理でコーパス構築
manager = CorpusManager.create_simple_rag(doc_store, vector_store)
results = manager.process_corpus(["documents/"])

# 信頼性のあるクエリ
query_engine = QueryEngine(retriever, reranker, reader)
result = query_engine.answer("リモートワークに関する会社のポリシーは？")
```

### 企業機能
```python
# 増分更新（大規模コーパスで90%以上の時間削減）
incremental_loader = IncrementalLoader(document_store, cache_file=".cache.json")
results = incremental_loader.process_incremental(["documents/"])

# 部署レベルデータ分離（チュートリアル5パターン）
hr_rag = CorpusManager.create_simple_rag(hr_doc_store, hr_vector_store)
sales_rag = CorpusManager.create_simple_rag(sales_doc_store, sales_vector_store)

# 本番監視
stats = corpus_manager.get_corpus_stats()
```

## 📚 ドキュメント

### 🎯 チュートリアル
ステップバイステップでRAGシステム構築を学習 - シンプルなプロトタイプから企業デプロイメントまで。

- [チュートリアル概要](docs/tutorials/tutorial_overview_ja.md)
- [チュートリアル1: 基本的なRAGパイプライン](docs/tutorials/tutorial_01_basic_rag_ja.md)
- [チュートリアル2: コーパス管理とドキュメント処理](docs/tutorials/tutorial_02_corpus_management_ja.md)
- [チュートリアル3: クエリエンジンと回答生成](docs/tutorials/tutorial_03_query_engine_ja.md)
- [チュートリアル4: 高度な正規化とクエリ処理](docs/tutorials/tutorial_04_normalization_ja.md)
- [チュートリアル5: 企業での利用方法](docs/tutorials/tutorial_05_enterprise_usage_ja.md)
- [チュートリアル6: 増分文書ローディング](docs/tutorials/tutorial_06_incremental_loading_ja.md)

### 📖 APIリファレンス
各モジュールの詳細APIドキュメント。

- [APIリファレンス トップ](docs/api/index_ja.md)
- [models - データモデル定義](docs/api/models_ja.md)
- [processing - ドキュメント処理パイプライン](docs/api/processing_ja.md)
- [CorpusManager - コーパス管理](docs/api/corpus_manager_ja.md)
- [QueryEngine - クエリ処理エンジン](docs/api/query_engine_ja.md)

### 🏗️ アーキテクチャ・設計
システム設計思想と実装詳細。

- [設計文書](docs/concept.md)
- [アーキテクチャ概要](docs/architecture.md)
- [要件](docs/requirements.md)
- [機能仕様](docs/function_spec.md)
- [バックエンドインターフェース](docs/backend_interfaces.md)
- [ユースケースインターフェース](docs/usecase_interfaces.md)

## 🏆 フレームワーク比較

| 機能 | LangChain/LlamaIndex | refinire-rag | 優位性 |
|------|---------------------|---------------|--------|
| **開発速度** | 複雑なセットアップ | 5行セットアップ | **90%高速** |
| **企業機能** | カスタム開発 | 内蔵 | **すぐに利用可能** |
| **日本語処理** | 追加作業 | 最適化済み | **ネイティブサポート** |
| **増分更新** | 手動実装 | 自動 | **90%時間削減** |
| **コード一貫性** | コンポーネント固有API | 統一インターフェース | **保守が簡単** |
| **チーム生産性** | 急な学習曲線 | 単一パターン | **迅速なオンボーディング** |

## 主要機能

### DocumentProcessor統一モデル
- すべてが同じ`process(document) -> List[Document]`インターフェース
- 設定ベースの動作制御
- 統一されたエラーハンドリングとログ

### 増分処理
- ファイル変更検出（修正時刻、サイズ、ハッシュ）
- 新規・更新文書のみを処理
- 大規模文書コレクションの効率的な管理

### 日本語最適化
- 表現揺らぎの自動正規化
- 単語境界の適切な処理
- 和英混在文書の対応

### 企業対応設計
- 部署別データ分離パターン
- 包括的な監視・ログ
- 本番環境対応のエラーハンドリング

## 開発

### テスト実行
```bash
# 実装テスト実行
python -m pytest tests/

# サンプル実行
python examples/simple_rag_test.py
```

### プロジェクト構造
```
refinire-rag/
├── src/refinire_rag/          # メインパッケージ
│   ├── models/                # データモデル
│   ├── loaders/              # 文書ローディングシステム
│   ├── processing/           # 文書処理パイプライン
│   ├── storage/              # ストレージシステム
│   ├── use_cases/            # ユースケースクラス
│   └── retrieval/            # 検索・回答生成
├── docs/                     # アーキテクチャドキュメント
├── examples/                 # 使用例
└── tests/                    # テストスイート
```

## ライセンス

[ライセンス情報は追加予定]

---

**refinire-rag: 企業RAG開発が楽々になる場所。**