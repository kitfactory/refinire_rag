# チュートリアル1: 基本的なRAGパイプライン

このチュートリアルでは、refinire-ragを使用して最もシンプルなRAGシステムを構築します。

## 学習目標

- RAGの基本概念を理解する
- 簡単なドキュメントコーパスを構築する
- 基本的なクエリ処理と回答生成を実行する

## RAGの基本構成

RAG（Retrieval-Augmented Generation）システムは以下の要素から構成されます：

```
文書 → [埋め込み] → ベクトルストア
                      ↓
クエリ → [埋め込み] → [検索] → [再ランキング] → [回答生成] → 回答
```

## ステップ1: 基本的なセットアップ

まず、必要なモジュールをインポートします：

```python
from refinire_rag.use_cases.corpus_manager_new import CorpusManager
from refinire_rag.use_cases.query_engine import QueryEngine
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.document import Document
```

## ステップ2: サンプルドキュメントの作成

RAGシステムで使用するサンプルドキュメントを作成します：

```python
def create_sample_documents():
    """サンプルドキュメントを作成"""
    
    documents = [
        Document(
            id="doc1",
            content="""
            RAG（Retrieval-Augmented Generation）は、検索拡張生成技術です。
            大規模言語モデル（LLM）と外部知識ベースを組み合わせ、
            より正確で根拠のある回答を生成します。
            主な利点は、ハルシネーションの減少、知識の更新容易性、
            専門ドメインへの適応性です。
            """,
            metadata={"title": "RAG概要", "category": "技術"}
        ),
        
        Document(
            id="doc2",
            content="""
            ベクトル検索は、意味的類似性に基づく検索技術です。
            文書やクエリを高次元ベクトル空間に埋め込み、
            コサイン類似度などを使用して関連性を計算します。
            従来のキーワード検索では発見できない
            文脈的に関連する情報を見つけることができます。
            """,
            metadata={"title": "ベクトル検索", "category": "技術"}
        ),
        
        Document(
            id="doc3",
            content="""
            大規模言語モデル（LLM）は、自然言語処理の中核技術です。
            GPT、Claude、Geminiなどの先進モデルが存在し、
            文章生成、翻訳、要約、質疑応答など
            幅広いタスクに対応できます。
            企業では、カスタマーサポート、コンテンツ生成、
            文書解析などの用途で活用されています。
            """,
            metadata={"title": "大規模言語モデル", "category": "技術"}
        )
    ]
    
    return documents
```

## ステップ3: ストレージの初期化

ドキュメントストアとベクトルストアを初期化します：

```python
def setup_storage():
    """ストレージを初期化"""
    
    # ドキュメントストア（メタデータと原文を保存）
    document_store = SQLiteDocumentStore(":memory:")
    
    # ベクトルストア（埋め込みベクトルを保存）
    vector_store = InMemoryVectorStore()
    
    return document_store, vector_store
```

## ステップ4: シンプルなコーパス構築

最もシンプルなRAGパイプライン（Load → Chunk → Vector）でコーパスを構築します：

```python
def build_simple_corpus(documents, document_store, vector_store):
    """シンプルなコーパスを構築"""
    
    print("📚 コーパスを構築中...")
    
    # Simple RAGマネージャーを作成
    corpus_manager = CorpusManager.create_simple_rag(
        document_store, 
        vector_store
    )
    
    # ドキュメントを手動でストアに追加（実際のファイルパスの代わり）
    for doc in documents:
        document_store.store_document(doc)
    
    # ベクトル埋め込みを手動で作成
    embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=embedder_config)
    
    # コーパスでembedderを訓練
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    
    # 各ドキュメントのベクトルを生成してストア
    from refinire_rag.storage.vector_store import VectorEntry
    
    for doc in documents:
        embedding_result = embedder.embed_text(doc.content)
        vector_entry = VectorEntry(
            document_id=doc.id,
            content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            embedding=embedding_result.vector.tolist(),
            metadata=doc.metadata
        )
        vector_store.add_vector(vector_entry)
    
    print(f"✅ {len(documents)}件のドキュメントでコーパスを構築しました")
    return embedder
```

## ステップ5: クエリエンジンの作成

検索と回答生成を行うクエリエンジンを作成します：

```python
def create_query_engine(document_store, vector_store, embedder):
    """クエリエンジンを作成"""
    
    print("🤖 クエリエンジンを作成中...")
    
    # 検索・回答生成コンポーネントを作成
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # クエリエンジンを作成
    query_engine = QueryEngine(
        document_store=document_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker
    )
    
    print("✅ クエリエンジンを作成しました")
    return query_engine
```

## ステップ6: 質疑応答のテスト

作成したRAGシステムでいくつかの質問をテストします：

```python
def test_questions(query_engine):
    """質問をテストする"""
    
    questions = [
        "RAGとは何ですか？",
        "ベクトル検索の仕組みを教えて",
        "LLMの主な用途は？",
        "RAGの利点を説明してください"
    ]
    
    print("\\n" + "="*60)
    print("🔍 質疑応答テスト")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\\n📌 質問 {i}: {question}")
        print("-" * 40)
        
        try:
            result = query_engine.answer(question)
            
            print(f"🤖 回答:")
            print(f"   {result.answer}")
            
            print(f"\\n📊 詳細:")
            print(f"   - 処理時間: {result.metadata.get('processing_time', 0):.3f}秒")
            print(f"   - 参考文書数: {result.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result.confidence:.3f}")
            
            if result.sources:
                print(f"   - 主な参考文書: {result.sources[0].metadata.get('title', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ エラー: {e}")
```

## ステップ7: 完全なサンプルプログラム

以下が完全なサンプルプログラムです：

```python
#!/usr/bin/env python3
"""
チュートリアル1: 基本的なRAGパイプライン
"""

def main():
    """メイン関数"""
    
    print("🚀 基本的なRAGパイプライン チュートリアル")
    print("="*60)
    
    try:
        # ステップ1: サンプルドキュメント作成
        documents = create_sample_documents()
        print(f"📝 {len(documents)}件のサンプルドキュメントを作成")
        
        # ステップ2: ストレージ初期化
        document_store, vector_store = setup_storage()
        print("💾 ストレージを初期化")
        
        # ステップ3: コーパス構築
        embedder = build_simple_corpus(documents, document_store, vector_store)
        
        # ステップ4: クエリエンジン作成
        query_engine = create_query_engine(document_store, vector_store, embedder)
        
        # ステップ5: 質疑応答テスト
        test_questions(query_engine)
        
        print("\\n🎉 チュートリアル1が完了しました！")
        print("\\n次は [チュートリアル2: コーパス管理とドキュメント処理] に進みましょう。")
        
    except Exception as e:
        print(f"\\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
```

## 実行方法

このサンプルプログラムを実行するには：

```bash
# チュートリアルディレクトリに移動
cd tutorials

# プログラムを実行
python tutorial_01_basic_rag.py
```

## 期待される出力

プログラムを実行すると、以下のような出力が表示されます：

```
🚀 基本的なRAGパイプライン チュートリアル
============================================================
📝 3件のサンプルドキュメントを作成
💾 ストレージを初期化
📚 コーパスを構築中...
✅ 3件のドキュメントでコーパスを構築しました
🤖 クエリエンジンを作成中...
✅ クエリエンジンを作成しました

============================================================
🔍 質疑応答テスト
============================================================

📌 質問 1: RAGとは何ですか？
----------------------------------------
🤖 回答:
   RAG（Retrieval-Augmented Generation）は、検索拡張生成技術です。
   大規模言語モデルと外部知識ベースを組み合わせ、より正確で根拠のある回答を生成します。

📊 詳細:
   - 処理時間: 0.002秒
   - 参考文書数: 3
   - 信頼度: 0.250
   - 主な参考文書: RAG概要

...
```

## 理解度チェック

このチュートリアルで学んだ内容を確認しましょう：

1. **RAGの基本構成要素**は何ですか？
   - ドキュメントストア、ベクトルストア、検索、回答生成

2. **Simple RAGパイプライン**の処理順序は？
   - Load（読み込み）→ Chunk（分割）→ Vector（ベクトル化）

3. **QueryEngine**の主な役割は？
   - クエリを受け取り、検索・回答生成を統合管理

## 次のステップ

基本的なRAGパイプラインが理解できたら、[チュートリアル2: コーパス管理とドキュメント処理](tutorial_02_corpus_management.md)に進んで、より高度なドキュメント処理機能を学習しましょう。

## トラブルシューティング

### よくある問題

1. **ImportError**: モジュールが見つからない
   ```bash
   pip install -e .
   ```

2. **TF-IDFエラー**: コーパスが小さすぎる
   ```python
   # min_df=1に設定
   embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
   ```

3. **メモリエラー**: 大きなドキュメント
   ```python
   # チャンクサイズを小さくする
   chunk_config = ChunkingConfig(chunk_size=200)
   ```