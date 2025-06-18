# 実施事項

## 0. LangChain同等機能実装（新規追加）

### 0.1 Phase 1: コア8割対応（P0優先度）
- [x] **PDFLoader**: PDF文書を読み込むLoaderを実装する（LangChain互換）- プラグイン開発中
- [x] **ChromaVectorStore**: Chromaベクターストアを実装する（LangChain互換）- プラグイン開発中  
- [x] **BM25Retriever**: BM25キーワード検索を実装する（LangChain互換）- プラグイン開発中

**次の優先度: 以下を実装**

- [x] **CSVLoader**: CSVファイルを読み込むLoaderを実装する（LangChain互換）- ✅ 実装済み
  - ファイル: `src/refinire_rag/loader/csv_loader.py`
  - 追加必要: 環境変数対応 (REFINIRE_RAG_CSV_DELIMITER, REFINIRE_RAG_CSV_ENCODING)

- [x] **TextFileLoader**: テキストファイルを読み込むLoaderを実装する（LangChain互換）- ✅ 実装済み  
  - ファイル: `src/refinire_rag/loader/text_loader.py`
  - 追加必要: 環境変数対応 (REFINIRE_RAG_TEXT_ENCODING)

- [x] **JSONLoader**: JSONファイルを読み込むLoaderを実装する（LangChain互換）- ✅ 実装済み
  - ファイル: `src/refinire_rag/loader/json_loader.py`
  - 追加必要: 環境変数対応

- [x] **HTMLLoader**: HTMLファイルを読み込むLoaderを実装する（LangChain以上）- ✅ 実装済み
  - ファイル: `src/refinire_rag/loader/html_loader.py`
  - 優位性: LangChainより高機能

- [x] **DirectoryLoader**: ディレクトリ内ファイル読み込みLoaderを実装する（LangChain互換）- ✅ 実装済み
  - ファイル: `src/refinire_rag/loader/directory_loader.py`

- [x] **RecursiveChunker**: 再帰的文字分割チャンカーを実装する（LangChain互換）- ✅ 実装完了
  - ファイル: `src/refinire_rag/processing/recursive_chunker.py`
  - 機能: 階層的分割、セパレータ設定、オーバーラップ制御、環境変数完全対応
  - 環境変数: REFINIRE_RAG_CHUNK_SIZE, REFINIRE_RAG_CHUNK_OVERLAP, REFINIRE_RAG_SEPARATORS
  - テスト: 15/15 PASSED (92%カバレッジ)

- [ ] **HuggingFaceEmbedder**: HuggingFace埋め込みを実装する（LangChain互換）
  - ファイル: `src/refinire_rag/processing/huggingface_embedder.py`
  - 機能: HuggingFaceモデル統合、ローカル実行
  - 環境変数: REFINIRE_RAG_HF_MODEL_NAME, REFINIRE_RAG_HF_DEVICE
  - 優先度: 🔶 **高優先** (ユーザー需要75%)

- [ ] **BufferMemory**: 会話履歴管理を実装する（LangChain互換）
  - ファイル: `src/refinire_rag/memory/buffer_memory.py`
  - 機能: 会話履歴保持、コンテキスト管理
  - 環境変数: REFINIRE_RAG_MEMORY_MAX_TOKENS, REFINIRE_RAG_MEMORY_TYPE
  - 優先度: 🔶 **高優先** (ユーザー需要80%)

**P1優先度（後続実装）:**

- [ ] **FaissVectorStore**: Faissベクターストアを実装する（LangChain互換）
  - ファイル: `src/refinire_rag/storage/faiss_vector_store.py`
  - 機能: Faiss統合、インデックス保存、高速検索
  - 環境変数: REFINIRE_RAG_FAISS_INDEX_PATH, REFINIRE_RAG_FAISS_INDEX_TYPE
  - 優先度: 🔸 **中優先** (ユーザー需要75%)

**Splitter環境変数対応（LangChain完全互換のため）:**

- [ ] **CharacterTextSplitter環境変数対応**: 文字分割の環境変数設定を追加
  - ファイル: `src/refinire_rag/splitter/character_splitter.py`
  - 環境変数: REFINIRE_RAG_CHARACTER_CHUNK_SIZE, REFINIRE_RAG_CHARACTER_OVERLAP
  - 機能: from_env()メソッド、統一メタデータ

- [ ] **TokenTextSplitter環境変数対応**: トークン分割の環境変数設定を追加
  - ファイル: `src/refinire_rag/splitter/token_splitter.py` 
  - 環境変数: REFINIRE_RAG_TOKEN_CHUNK_SIZE, REFINIRE_RAG_TOKEN_OVERLAP, REFINIRE_RAG_TOKEN_SEPARATOR
  - 機能: from_env()メソッド、統一メタデータ

- [ ] **MarkdownTextSplitter環境変数対応**: Markdown分割の環境変数設定を追加
  - ファイル: `src/refinire_rag/splitter/markdown_splitter.py`
  - 環境変数: REFINIRE_RAG_MD_CHUNK_SIZE, REFINIRE_RAG_MD_OVERLAP, REFINIRE_RAG_MD_HEADERS
  - 機能: from_env()メソッド、見出し保持制御

### 0.2 プラグインシステム拡張
- [ ] **PluginFactory拡張**: 新規プラグインの自動登録機能を実装する
  - ファイル: `src/refinire_rag/factories/plugin_factory.py`
  - 機能: 環境変数からの自動プラグイン作成、設定管理

- [ ] **PluginRegistry拡張**: LangChain互換プラグインの登録
  - ファイル: `src/refinire_rag/registry/plugin_registry.py`
  - 機能: 新規コンポーネントの登録、互換性管理

### 0.3 環境変数設定システム
- [ ] **環境変数設定ドキュメント**: 全環境変数の一覧と説明を作成
  - ファイル: `docs/environment_variables.md`
  - 内容: 全コンポーネントの環境変数、デフォルト値、設定例

- [ ] **設定検証システム**: 環境変数の妥当性チェック機能を実装
  - ファイル: `src/refinire_rag/config/env_validator.py`
  - 機能: 型チェック、必須項目確認、設定ガイダンス

### 0.4 Phase 1テスト実装
- [ ] **PDFLoaderテスト**: PDFローダーのテストケースを作成
  - ファイル: `tests/test_pdf_loader.py`
  - 内容: PDF読み込み、エラーハンドリング、環境変数テスト

- [ ] **ChromaVectorStoreテスト**: Chromaベクターストアのテストケースを作成
  - ファイル: `tests/test_chroma_vector_store.py`
  - 内容: 接続、データ保存、検索機能テスト

- [ ] **RecursiveChunkerテスト**: 再帰的チャンカーのテストケースを作成
  - ファイル: `tests/test_recursive_chunker.py`
  - 内容: 分割ロジック、オーバーラップ、環境変数設定テスト

- [ ] **BM25Retrieverテスト**: BM25検索のテストケースを作成
  - ファイル: `tests/test_bm25_retriever.py`
  - 内容: ランキング、パラメータ調整、検索精度テスト

- [ ] **統合テスト**: Phase 1コンポーネントの統合テストを作成
  - ファイル: `tests/integration/test_phase1_integration.py`
  - 内容: エンドツーエンド動作、環境変数による動作制御確認

### 0.5 LangChain移行ガイド
- [ ] **移行ガイド作成**: LangChainからrefinire-ragへの移行手順を作成
  - ファイル: `docs/migration_from_langchain.md`
  - 内容: コンポーネント対応表、設定変換、コード例

- [ ] **機能比較表**: LangChainとrefinire-ragの機能比較表を作成
  - ファイル: `docs/langchain_comparison.md`
  - 内容: 機能対応、性能比較、差別化ポイント

## 0. 例外設計とアーキテクチャ改善
- [x] **例外クラス体系の整備**: refinire_rag全体で使用する例外クラスの階層構造を設計・実装する。
  - ファイル: `src/refinire_rag/exceptions.py`
  - 内容: RefinireRAGError基底クラス、機能別例外クラス、例外ラッピング機能
- [x] **DocumentStoreLoader設計書**: DocumentStoreから文書をロードするLoaderの設計を策定する。
  - ファイル: `docs/document_store_loader_design.md`
  - 内容: 複数ロード戦略、フィルタリング、エラーハンドリング、設定管理
- [x] **DocumentStoreLoader実装**: DocumentStoreから文書をロードするLoaderクラスを実装する。
  - ファイル: `src/refinire_rag/loader/document_store_loader.py`
  - 内容: DocumentLoadConfig、LoadStrategy、統一例外処理
- [x] **DocumentStoreLoaderテスト**: DocumentStoreLoaderのテストケースを実装する。
  - ファイル: `tests/test_document_store_loader.py`
  - 内容: 各ロード戦略、エラーハンドリング、設定検証のテスト
- [x] **DocumentStoreLoader使用ガイド**: DocumentStoreLoaderの使用方法ドキュメントを作成する。
  - ファイル: `docs/document_store_loader_usage.md`
  - 内容: 基本使用法、高度な機能、ベストプラクティス、トラブルシューティング
- [ ] **既存コードの例外統合**: 既存のコードで新しい例外クラスを使用するよう修正する。
  - 対象: loader、storage、processing配下のクラス
  - 内容: 例外の置き換え、エラーハンドリングの改善

## 0.1. インクリメンタルローダー実装（完了済み）
- [x] **ファイル追跡モデル**: ファイル変更検出用のモデルクラスを実装する。
  - ファイル: `src/refinire_rag/loader/models/file_info.py`, `change_set.py`, `sync_result.py`, `filter_config.py`
  - 内容: FileInfo、ChangeSet、SyncResult、FilterConfigクラス
- [x] **フィルターシステム**: ファイルフィルタリング用の基底クラスと具体的フィルターを実装する。
  - ファイル: `src/refinire_rag/loader/filters/`配下
  - 内容: BaseFilter、ExtensionFilter、DateFilter、PathFilter、SizeFilter
- [x] **FileTracker**: ディレクトリスキャン間のファイル変更を検出するクラスを実装する。
  - ファイル: `src/refinire_rag/loader/file_tracker.py`
  - 内容: MD5ハッシュ比較、変更検出、永続化機能
- [x] **IncrementalDirectoryLoader**: 変更されたファイルのみを処理するLoaderを実装する。
  - ファイル: `src/refinire_rag/loader/incremental_directory_loader.py`
  - 内容: 増分同期、フィルター統合、DocumentStore連携
- [x] **インクリメンタルローダーテスト**: 各コンポーネントのテストケースを実装する。
  - ファイル: `tests/test_incremental_loader.py`
  - 内容: フィルター、ファイル追跡、ローダー統合テスト

## 1. モデルと機能クラスの実装
- [x] **Document**: 文書データを示すクラスを実装し、メタデータやID情報を含める。
  - ファイル: `src/refinire/rag/models/document.py`
- [x] **QAPair**: 主にDocumentから生成されたQA情報を表現するクラスを実装する。
  - ファイル: `src/refinire/rag/models/qa_pair.py`
- [x] **EvaluationResult**: RAGの評価結果を表すクラスを実装する。
  - ファイル: `src/refinire/rag/models/evaluation_result.py`
- [x] **CorpusStore**: Embeddings/Indexとなる前の文書をメタデータとともに保存するクラスを実装する。
  - ファイル: `src/refinire/rag/corpusstore.py`
- [x] **SQLiteCorpusStore**: CorpusStoreの実装クラスとして、SQLiteを使用した文書保存を実装する。
  - ファイル: `src/refinire/rag/corpus_store/sqlite_corpus_store.py`
- [ ] **DocumentProcessor**: 文書を処理するインターフェースを実装し、すべての文書処理機能を単一の`process`メソッドで実装する。
  - ファイル: `src/refinire/rag/documentprocessor.py`
- [ ] **Loader**: 文書を処理するインターフェースを実装し、`process`メソッドでロード結果を返却する。
  - ファイル: `src/refinire/rag/loader.py`
- [ ] **Retriever**: メタデータの条件とクエリ文から文書の検索を担当するクラスを実装する。
  - ファイル: `src/refinire/rag/retriever.py`
- [ ] **Indexer**: 与えられた文書、メタデータから文書を検索可能な状態にするクラスを実装する。
  - ファイル: `src/refinire/rag/indexer.py`
- [ ] **KeywordSearch**: RetrieverとIndexerを継承したクラスを実装し、BM25などのキーワード検索型に対応する。
  - ファイル: `src/refinire/rag/keywordsearch.py`
- [ ] **VectorStore**: RetrieverとIndexerを継承したクラスを実装し、ベクトル検索に対応する。
  - ファイル: `src/refinire/rag/vectorstore.py`
- [ ] **Embedder**: VectorSearchに必要な情報をEmbeddingsを提供するクラスを実装する。
  - ファイル: `src/refinire/rag/embedder.py`
- [ ] **OpenAIEmbedder**: OpenAIを使用したEmbedderの実装クラスを作成する。
  - ファイル: `src/refinire/rag/embedder/openai_embedder.py`
- [ ] **Reranker**: Retrieverから取得された返却結果を再ランキングするクラスを実装する。
  - ファイル: `src/refinire/rag/reranker.py`

## 2. テストの作成と実行
### 2.1 テストの作成
- [ ] **Document**: 文書データクラスのテストを作成する。
  - ファイル: `tests/test_models.py`
- [ ] **QAPair**: QA情報クラスのテストを作成する。
  - ファイル: `tests/test_models.py`
- [ ] **EvaluationResult**: 評価結果クラスのテストを作成する。
  - ファイル: `tests/test_models.py`
- [ ] **CorpusStore**: 文書保存クラスのテストを作成する。
  - ファイル: `tests/test_corpusstore.py`
- [ ] **SQLiteCorpusStore**: SQLiteを使用した文書保存クラスのテストを作成する。
  - ファイル: `tests/test_corpusstore.py`
- [ ] **DocumentProcessor**: 文書処理クラスのテストを作成する。
  - ファイル: `tests/test_documentprocessor.py`
- [ ] **Loader**: 文書ロードクラスのテストを作成する。
  - ファイル: `tests/test_loader.py`
- [ ] **Retriever**: 文書検索クラスのテストを作成する。
  - ファイル: `tests/test_retriever.py`
- [ ] **Indexer**: 文書インデックスクラスのテストを作成する。
  - ファイル: `tests/test_indexer.py`
- [ ] **KeywordSearch**: キーワード検索クラスのテストを作成する。
  - ファイル: `tests/test_keywordsearch.py`
- [ ] **VectorStore**: ベクトル検索クラスのテストを作成する。
  - ファイル: `tests/test_vectorstore.py`
- [ ] **Embedder**: 埋め込みクラスのテストを作成する。
  - ファイル: `tests/test_embedder.py`
- [ ] **OpenAIEmbedder**: OpenAIを使用した埋め込みクラスのテストを作成する。
  - ファイル: `tests/test_embedder.py`
- [ ] **Reranker**: 再ランキングクラスのテストを作成する。
  - ファイル: `tests/test_reranker.py`

### 2.2 テストの実行
- [ ] **Document**: 文書データクラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_models.py`
- [ ] **QAPair**: QA情報クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_models.py`
- [ ] **EvaluationResult**: 評価結果クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_models.py`
- [ ] **CorpusStore**: 文書保存クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_corpusstore.py`
- [ ] **SQLiteCorpusStore**: SQLiteを使用した文書保存クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_corpusstore.py`
- [ ] **DocumentProcessor**: 文書処理クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_documentprocessor.py`
- [ ] **Loader**: 文書ロードクラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_loader.py`
- [ ] **Retriever**: 文書検索クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_retriever.py`
- [ ] **Indexer**: 文書インデックスクラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_indexer.py`
- [ ] **KeywordSearch**: キーワード検索クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_keywordsearch.py`
- [ ] **VectorStore**: ベクトル検索クラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_vectorstore.py`
- [ ] **Embedder**: 埋め込みクラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_embedder.py`
- [ ] **OpenAIEmbedder**: OpenAIを使用した埋め込みクラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_embedder.py`
- [ ] **Reranker**: 再ランキングクラスのテストを実行し、パスすることを確認する。
  - ファイル: `tests/test_reranker.py`

### 2.1.1 スプリッターの実装
- [ ] **CharacterTextSplitter**: 文字（デフォルト '\n\n'）を基準に分割する基本的なスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/character_splitter.py`
- [ ] **RecursiveCharacterTextSplitter**: 階層的に複数レベルのセパレータ（段落→文→語）で分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/recursive_character_splitter.py`
- [ ] **TokenTextSplitter**: トークン数を単位に分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/token_splitter.py`
- [ ] **MarkdownTextSplitter**: Markdownドキュメントを見出しベースで分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/markdown_splitter.py`
- [ ] **MarkdownHeaderTextSplitter**: 指定した見出しレベルで分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/markdown_header_splitter.py`
- [ ] **HTMLHeaderTextSplitter**: HTMLの見出しタグを基準に分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/html_header_splitter.py`
- [ ] **HTMLSectionSplitter**: HTMLのセクションを詳細に分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/html_section_splitter.py`
- [ ] **RecursiveJsonSplitter**: JSONデータの階層構造を保持したまま分割するスプリッターを実装する。
  - ファイル: `src/refinire/rag/processor/recursive_json_splitter.py`

### 2.1.2 スプリッターのテスト作成
- [ ] **CharacterTextSplitter**: 文字分割スプリッターのテストを作成する。
  - ファイル: `tests/test_character_splitter.py`
- [ ] **RecursiveCharacterTextSplitter**: 再帰的文字分割スプリッターのテストを作成する。
  - ファイル: `tests/test_recursive_character_splitter.py`
- [ ] **TokenTextSplitter**: トークン分割スプリッターのテストを作成する。
  - ファイル: `tests/test_token_splitter.py`
- [ ] **MarkdownTextSplitter**: Markdown分割スプリッターのテストを作成する。
  - ファイル: `tests/test_markdown_splitter.py`
- [ ] **MarkdownHeaderTextSplitter**: Markdown見出し分割スプリッターのテストを作成する。
  - ファイル: `tests/test_markdown_header_splitter.py`
- [ ] **HTMLHeaderTextSplitter**: HTML見出し分割スプリッターのテストを作成する。
  - ファイル: `tests/test_html_header_splitter.py`
- [ ] **HTMLSectionSplitter**: HTMLセクション分割スプリッターのテストを作成する。
  - ファイル: `tests/test_html_section_splitter.py`
- [ ] **RecursiveJsonSplitter**: JSON分割スプリッターのテストを作成する。
  - ファイル: `tests/test_recursive_json_splitter.py`

### 2.1.3 Loaderの実装
- [ ] **TextLoader**: テキストファイルを読み込むLoaderを実装する。
  - ファイル: `src/refinire/rag/loader/text_loader.py`
  - 機能: テキストファイルを読み込み、Documentオブジェクトを生成
  - 依存: なし（標準ライブラリのみ）

- [ ] **DirectoryLoader**: ディレクトリ内のファイルを再帰的に読み込むLoaderを実装する。
  - ファイル: `src/refinire/rag/loader/directory_loader.py`
  - 機能: 指定されたディレクトリ内のファイルを再帰的に読み込み
  - 依存: なし（標準ライブラリのみ）

- [ ] **CSVLoader**: CSVファイルを読み込むLoaderを実装する。
  - ファイル: `src/refinire/rag/loader/csv_loader.py`
  - 機能: CSVファイルを読み込み、各行をDocumentオブジェクトとして生成
  - 依存: なし（標準ライブラリのcsvモジュール）

- [ ] **JSONLoader**: JSONファイルを読み込むLoaderを実装する。
  - ファイル: `src/refinire/rag/loader/json_loader.py`
  - 機能: JSONファイルを読み込み、構造化データをDocumentオブジェクトとして生成
  - 依存: なし（標準ライブラリのjsonモジュール）

### 2.1.4 Loaderのテスト作成
- [ ] **TextLoader**: テキストファイルローダーのテストを作成する。
  - ファイル: `tests/test_text_loader.py`
  - テスト内容: テキストファイルの読み込み、エンコーディング処理、メタデータの保持

- [ ] **DirectoryLoader**: ディレクトリローダーのテストを作成する。
  - ファイル: `tests/test_directory_loader.py`
  - テスト内容: ディレクトリ内のファイル再帰的読み込み、フィルタリング機能

- [ ] **CSVLoader**: CSVファイルローダーのテストを作成する。
  - ファイル: `tests/test_csv_loader.py`
  - テスト内容: CSVファイルの読み込み、カラムマッピング、データ型変換

- [ ] **JSONLoader**: JSONファイルローダーのテストを作成する。
  - ファイル: `tests/test_json_loader.py`
  - テスト内容: JSONファイルの読み込み、ネストされた構造の処理、スキーマ検証

## 3. ユースケースの実装
- [ ] **CorpusManager**: 文書のロード、正規化、チャンク、Embedding生成、保存、辞書/グラフの生成を担当するクラスを実装する。
  - ファイル: `src/refinire/rag/corpusmanager.py`
- [ ] **QueryEngine**: クエリに対する文書検索と再ランキングを行うクラスを実装する。
  - ファイル: `src/refinire/rag/queryengine.py`
- [ ] **QualityLab**: RAGの評価データの作成、評価、評価レポート生成を担当するクラスを実装する。
  - ファイル: `src/refinire/rag/qualitylab.py`

## 4. ドキュメントの更新
- [ ] 実装したクラスやメソッドに関するドキュメントを更新し、使用方法や機能を明確にする。
  - ファイル: `docs/requirements.md`, `docs/architecture.md`, `docs/function_spec.md`
