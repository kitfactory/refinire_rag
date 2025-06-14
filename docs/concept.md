# refinire-rag 設計文書

## 1. 概要

refinire-rag は、RAG (Retrieval-Augmented Generation) システムの開発・運用を支援する Python ライブラリである。本設計はRag機能を＝RefinireライブラリのStepとして提供する。

DocumentProcessor統一アーキテクチャ、を「DI（依存性注入）」によって構成を切り替えて使用する。

また、Refinireのサブパッケージとして、RAG機能を提供する。
* Refinire: https://github.com/kitfactory/refinire

これはentry_point


### 1.1 アーキテクチャの特徴

大きく3つのアプリケーションに分けて、処理を実現する。

## 2. アプリケーションクラス

Ragの主要機能は3つある。
この主要機能をそれぞれのアプリケーション・クラスを窓口として対応する。

* エンベッディングやインデックスの作成：CorpusManager
* 実際の検索：
* 品質指標の取得

### 2.1 CorpusManager

* **責務**: 文書のロード、正規化、チャンク、Embedding 生成、保存、辞書/グラフの生成

* **主なメソッド**:

  * `add_documents(paths: List[str])`: 文書追加
  * `generate_dictionary(doc: Document)`: 用語抽出
  * `generate_graph(doc: Document)`: 関係グラフ作成

### 2.2 QueryEngine

* **責務**: クエリに対する文書検索、再ランキングした情報を返却する

* **主なメソッド**:
  * `run(query: str, ctx:Context )`: クエリに対して RAG 生成を行う。
  
QueryEngineはrefinireのStepサブクラスとして実装されている。Step#run()メソッドと同等とする。この出力をrefinireと連結することで様々なエージェントに組み込み、品質計測付きのRagとして実現が可能である。

### 2.3 QualityLab

* **責務**: RAGの評価データの作成、RAGの評価、評価レポート生成

* **主なメソッド**:
  * `build_qa_pair()`: CorpusStoreのデータを用いて、処理を行う。
  * `run_evaluation(test_set_path: str)`

## 3. 主要なデータ

主要なデータクラス、refinire.rag.modelsパッケージにて提供される。

### 3.1. Document

文書データを示す。メタデータやid情報などを含む。あるDocumentが処理されると１つまたは複数のDocumentが生成される。CorpusStoreにほぞんされる。

### 3.2. QAPair

主にDocumentから生成されたQA情報。必要に応じて生成で作

### 3.3. EvaluationResult

RAGの評価結果を表したクラス。

## 3. 主要なクラス

アプリケーションは機能部品で実現される。重要な機能部品クラスを記載する。これらは抽象クラスである。抽象クラスはrefinire.ragパッケージに配備される。

### 3.1 CorpusStore

* **責務**: Embeddings/Indexとなる前の文書をメタデータとともに保存する。

* **主な実装クラス**: SQLiteCorpusStore


### 3.2. DocumentProcessor

* **責務**: 文書を処理するインターフェースである。すべての文書処理機能（ローディング、正規化、チャンキング、埋め込み生成、評価など）はDocumentProcessorベースクラスを継承し、単一の`process(input:List[Document]) -> Iterator[Document]`インターフェースで実装される。

## 3.3. Loader

* **責務**: 文書を処理するインターフェースである。DocumentProcessorベースクラスを継承し、単一の`process(input:List[Document]) -> Iterator[Document]`インターフェースで実装される。実体は入力を無視して、ロード結果をIteratorに返却する。

## 3.4. Retriever

メタデータの条件とクエリ文から文書の検索を担当する

## 3.5. Indexer

与えられた文書、メタデータから文書を検索可能な状態にする

## 3.6. KeywordSearch

RetrieverとIndexerを継承したクラス。実質的にはVectorStoreと差異はないが、技術が明確に異なるため、BM25s等のKeywordSearch型には、こちらを利用する。

## 3.7. VectorStore

RetrieverとIndexerを継承したクラス。実質的にはVectorStoreと差異はないが、技術が明確に異なるため、BM25s等のKeywordSearch型には、こちらを利用する。

## 3.8. Embedder

VectorSearchに必要な情報をEmbeddingsを提供する。指定テキストの他、埋め込みの次元数を返却する。


## 3.9. Reranker

Retrieverから取得された返却結果をRerank結果を返却する。


