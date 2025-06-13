# プロジェクト構造

## 考え方：
* メインのパッケージ(refinire.rag)にデータ、抽象クラス、ユースケースを集め、構造を明確化する。
* 抽象クラスごとにサブパッケージを設ける。例：refinire.rag.reranker　など
* 外部パッケージとして提供されるプラグインは抽象クラスを実装し、パッケージのエントリポイントを設定することで、抽象クラスのパッケージでインポートできる MyRerankerを作った場合、refinire.rag.reranker.MyRerankerとなる。

## モジュール構成

### 1. コア (`src/refinire/rag/`)

下記を refinire.ragパッケージのクラスとして提供

#### 1.1 データモデル（`models.py`）
- **Document**: 文書を表現するデータクラス
- **CorpusPipeline**: 旧DocumentPipeline
- **SearchResult**: 検索結果を表現するデータクラス
- **QueryResult**: 最終的なクエリ結果を表現するデータクラス

#### 1.2 抽象クラス（`base.py`）
- **Retriever**: すべての文書検索実装の統一インターフェース
- **Indexer**: 文書のインデックスと管理機能を提供
- **KeywordSearch**: キーワードベースの文書検索の基底クラス
- **VectorStore**: ベクトルベースの文書ストアの基底クラス
- **Embedder**: ベクトルベースの文書ストアの基底クラス
- **Reranker**: 検索結果の再ランキングを行う基底クラス
- **AnswerSynthesizer**: LLMを使用した回答合成の基底クラス
- **Loader**: 文書読み込みの基底クラス
- **DocumentProcessor**: 文書処理の基底クラス
- **CorpusStore**: 文書保存の実装クラス

### 2. サブパッケージ

#### 2.1 文書保存 (`src/refinire/rag/corpusstore/`)
- **SQLiteCorpusStore**: 文書処理の実装クラス

#### 2.2 文書ローダー (`src/refinire/rag/corpusstore/`)
- **CorpusLoader**: CorpusStoreから文書を取得する。


#### 2.3 検索システム（`src/refinire/rag/retriever/`）
VectorStore/KeywordStore含めて、このパッケージに。

- **SimpleRetriever**（`simple_retriever.py`）: ベクトル類似度検索を実行する基本的なRetriever実装
- **HybridRetriever**（`hybrid_retriever.py`）: 複数の検索手法を組み合わせるハイブリッド検索器

#### 2.4 検索システム（`src/refinire/rag/retriever/`）

- **SimpleReranker**（`simple_reranker.py`）: 基本的な再ランキング実装
- **SimpleReader**（`simple_reader.py`）: 基本的な文書読み込み実装

#### 2.3 評価システム（`src/refinire/rag/evaluation/`）
- **Evaluator**: 検索結果の評価を行うクラス
- **Metrics**: 評価指標を提供するクラス

### 3. ユーティリティ（`src/refinire/rag/utils/`）
- **logging**: ロギング関連のユーティリティ
- **validation**: 入力検証関連のユーティリティ

### 4. テスト（`tests/`）
- **test_chunking.py**: チャンキング機能のテスト
- **test_document_processor_integration.py**: 文書処理の統合テスト
- **test_corpus_manager.py**: コーパス管理の統合テスト

### 5. ドキュメント（`docs/`）
- **requirements.md**: 要件定義書
- **architecture.md**: アーキテクチャ設計書
- **function_spec.md**: 機能仕様書
- **unified_retrieval_api.md**: 統一検索APIの仕様
- **project_structure.md**: プロジェクト構造の説明
- **todo.md**: タスク管理 