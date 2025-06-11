# refinire-rag 設計文書

## 1. 概要

refinire-rag は、RAG (Retrieval-Augmented Generation) システムの開発・運用を支援する Python ライブラリである。本設計は「ユースケース＝RefinireライブラリのStep」「DocumentProcessor統一アーキテクチャ」を「設定ファイルやDI（依存性注入）」によって構成を切り替えて使用する。

また、Refinireのサブパッケージとして、RAG機能を提供する。
* Refinire: https://github.com/kitfactory/refinire

### 1.1 アーキテクチャの特徴

**DocumentProcessor統一モデル**
すべての文書処理機能（ローディング、正規化、チャンキング、埋め込み生成、評価など）は、統一されたDocumentProcessorベースクラスを継承し、単一の`process(document) -> List[Document]`インターフェースで実装される。

**DocumentPipeline**
複数のDocumentProcessorを連鎖させて、文書処理パイプラインを構築する。各プロセッサは独立してテスト・設定可能。

**増分処理対応**
IncrementalLoaderにより、ファイル変更検出と差分処理を行い、大規模文書コーパスの効率的な更新を実現。

## 2. ユースケースクラス

各ユースケースはRefinire Stepサブクラスである。各ユースケースごとに関連するDocumentProcessorをDIで可変にし、切り替える。

### 2.1 CorpusManager

* **責務**: 文書のロード、正規化、チャンク、Embedding 生成、保存、辞書/グラフの生成

* **主なメソッド**:

  * `add_documents(paths: List[str])`: 文書追加
  * `generate_dictionary(doc: Document)`: 用語抽出
  * `generate_graph(doc: Document)`: 関係グラフ作成

### 2.2 QueryEngine

* **責務**: クエリに対する文書検索、再ランキング、回答生成

* **主なメソッド**:
  * `answer(query: str, ctx:Context )`: クエリに対して RAG 生成を行う。
  
QueryEngineはrefinireのStepサブクラスとして実装し、answerメソッドがStep#run()メソッドと同等とする。

### 2.3 QualityLab

* **責務**: 評価データの作成、RAG 結果の自動評価、文書矛盾の検知、改善レポート生成
* **主なメソッド**:

  * `run_evaluation(test_set_path: str)`
  * `detect_conflicts(docs: List[Document])`
  * `generate_report(metrics: Dict)`

## 3. DocumentProcessor モジュール（統一アーキテクチャ）

すべての文書処理機能は`DocumentProcessor`ベースクラスを継承し、統一インターフェースで実装される。

### 3.1 文書処理プロセッサ

| プロセッサ名                | 責務                  | 入力 → 出力                        |
| --------------------- | ------------------- | ----------------------------- |
| UniversalLoader       | 外部ファイル → Document 化 | `trigger_doc` → `List[Document]`           |
| DictionaryMaker       | 用語・略語の抽出とMD辞書更新 | `Document` → `Document + dict.md更新`        |
| Normalizer            | 用語の置換・タグ付け          | `Document` → `Document`       |
| GraphBuilder          | 主語・述語・目的語の関係抽出とMDグラフ更新      | `Document` → `Document + graph.md更新`       |
| TokenBasedChunker     | トークンでチャンク化          | `Document` → `List[Document(chunks)]`    |
| VectorStoreProcessor  | 埋め込み生成とベクトルストア保存      | `Document` → `Document + vector_store更新`          |

### 3.2 品質評価プロセッサ

| プロセッサ名                | 責務                  | 入力 → 出力                        |
| --------------------- | ------------------- | ----------------------------- |
| TestSuite             | 評価ランナー              | `Document(test_queries)` → `List[Document(results)]`   |
| Evaluator             | メトリクス集計             | `Document(results)` → `Document(metrics)`         |
| ContradictionDetector | claim 抽出 + NLI 判定   | `Document` → `Document(conflicts)` |
| InsightReporter       | 閾値超過の解釈とレポート        | `Document(metrics)` → `Document(insights)`      |

### 3.3 クエリエンジンコンポーネント

QueryEngineで使用される独立コンポーネント（DocumentProcessorとは別系統）:

| コンポーネント名             | 責務                  | 入力 → 出力                        |
| --------------------- | ------------------- | ----------------------------- |
| Retriever             | 文書検索                | `query` → `List[SearchResult]`       |
| Reranker              | 候補再順位付け             | `query, List[SearchResult]` → `List[SearchResult]` |
| Reader                | 回答生成                | `query + contexts` → `answer`   |

## 4. 増分処理とDocumentPipeline

### 4.1 IncrementalLoader

大規模文書コーパスの効率的な管理のため、増分処理機能を提供：

- **ファイル変更検出**: 修正時刻、サイズ、コンテンツハッシュによる変更検出
- **差分処理**: 新規・更新文書のみを処理し、未変更文書をスキップ
- **キャッシュ管理**: JSONベースのファイル状態キャッシュ
- **クリーンアップ**: 削除されたファイルに対応する文書の自動削除

### 4.2 DocumentPipeline統合

CorpusManagerはDocumentPipelineベースで再実装され、以下の利点を提供：

- **モジュラー設計**: 各処理ステップを独立したプロセッサとして実装
- **設定可能性**: プロセッサの組み合わせと順序を動的に変更
- **エラーハンドリング**: 各プロセッサでの例外処理と継続性制御
- **パフォーマンス**: バッチ処理と並列処理対応

## 5. 対応状況と実装特徴

| 対応項目       | 実装内容                                                                                  |
| ---------- | ------------------------------------------------------------------------------------- |
| 統一アーキテクチャ | すべての処理機能をDocumentProcessorサブクラスとして実装、統一インターフェース提供 |
| 増分処理       | IncrementalLoaderによるファイル変更検出と差分処理で大規模コーパス効率更新対応              |
| Config 検証  | dataclassベースの設定とpydanticバリデーション。型安全性と設定エラーの即時検出                                  |
| ロギング/メトリクス | 各ユースケースとプロセッサでの詳細ログ記録とパフォーマンス統計 |
| 並列化        | LoaderのAsync対応、並列処理オプション、バッチ処理による大規模データ対応|
| エラー処理    | 階層化された例外定義とgracefulな失敗処理、継続可能な処理フロー |

