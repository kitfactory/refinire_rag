# 実施事項

## 1. モデルと機能クラスの実装
- [ ] **Document**: 文書データを示すクラスを実装し、メタデータやID情報を含める。
  - ファイル: `src/refinire/rag/models.py`
- [ ] **QAPair**: 主にDocumentから生成されたQA情報を表現するクラスを実装する。
  - ファイル: `src/refinire/rag/models.py`
- [ ] **EvaluationResult**: RAGの評価結果を表すクラスを実装する。
  - ファイル: `src/refinire/rag/models.py`
- [ ] **CorpusStore**: Embeddings/Indexとなる前の文書をメタデータとともに保存するクラスを実装する。
  - ファイル: `src/refinire/rag/corpusstore.py`
- [ ] **SQLiteCorpusStore**: CorpusStoreの実装クラスとして、SQLiteを使用した文書保存を実装する。
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
