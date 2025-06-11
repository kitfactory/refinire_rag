# refinire-rag ドキュメント

refinire-ragの公式ドキュメントです。

## 📚 ドキュメント構成

### 🎯 チュートリアル
RAGシステムの構築方法を段階的に学習できます。

- **日本語版**
  - [チュートリアル概要](tutorials/tutorial_overview_ja.md)
  - [チュートリアル1: 基本的なRAGパイプライン](tutorials/tutorial_01_basic_rag_ja.md)
  - [チュートリアル2: コーパス管理とドキュメント処理](tutorials/tutorial_02_corpus_management_ja.md)
  - [チュートリアル3: クエリエンジンと回答生成](tutorials/tutorial_03_query_engine_ja.md)
  - [チュートリアル4: 高度な正規化とクエリ処理](tutorials/tutorial_04_normalization_ja.md)

- **English Version**
  - [Tutorial Overview](tutorials/tutorial_overview.md)
  - [Tutorial 1: Basic RAG Pipeline](tutorials/tutorial_01_basic_rag.md)

### 📖 APIリファレンス
各モジュールの詳細なAPIドキュメントです。

- **日本語版**
  - [APIリファレンス トップ](api/index_ja.md)
  - [models - データモデル定義](api/models_ja.md)
  - [processing - ドキュメント処理パイプライン](api/processing_ja.md)
  - [CorpusManager - コーパス管理](api/corpus_manager_ja.md)
  - [QueryEngine - クエリ処理エンジン](api/query_engine_ja.md)

- **English Version**
  - [API Reference Top](api/index.md)
  - [models - Data Model Definitions](api/models.md)
  - [processing - Document Processing Pipeline](api/processing.md)
  - [CorpusManager - Corpus Management](api/corpus_manager.md)
  - [QueryEngine - Query Processing Engine](api/query_engine.md)

### 🏗️ アーキテクチャ・設計
システムの設計思想と実装詳細です。

- [アーキテクチャ概要](architecture.md)
- [コンセプト](concept.md)
- [要件定義](requirements.md)
- [機能仕様](function_spec.md)
- [バックエンドインターフェース](backend_interfaces.md)
- [ユースケースインターフェース](usecase_interfaces.md)

## 🚀 クイックスタート

### インストール
```bash
pip install -e .
```

### 基本的な使い方
```python
from refinire_rag.use_cases.corpus_manager_new import CorpusManager
from refinire_rag.use_cases.query_engine import QueryEngine

# コーパスの構築
corpus_manager = CorpusManager.create_simple_rag(doc_store, vector_store)
stats = corpus_manager.build_corpus(["doc1.txt", "doc2.txt"])

# 質問応答
query_engine = QueryEngine(doc_store, vector_store, retriever, reader)
result = query_engine.answer("RAGとは何ですか？")
```

## 📝 ドキュメントの言語

- 🇬🇧 **英語**: ファイル名はそのまま（例: `tutorial_01_basic_rag.md`）
- 🇯🇵 **日本語**: ファイル名に `_ja` を付加（例: `tutorial_01_basic_rag_ja.md`）

## 🔗 関連リンク

- [GitHubリポジトリ](https://github.com/your-org/refinire-rag)
- [Issue Tracker](https://github.com/your-org/refinire-rag/issues)
- [ディスカッション](https://github.com/your-org/refinire-rag/discussions)

## 📄 ライセンス

このドキュメントはMITライセンスの下で公開されています。