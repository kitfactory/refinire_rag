#!/usr/bin/env python3
"""
チュートリアル1: 基本的なRAGパイプライン

このサンプルでは、refinire-ragを使用して最もシンプルなRAGシステムを構築し、
基本的な質疑応答機能をテストします。
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.document import Document
from refinire_rag.storage.vector_store import VectorEntry


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


def setup_storage():
    """ストレージを初期化"""
    
    # ドキュメントストア（メタデータと原文を保存）
    document_store = SQLiteDocumentStore(":memory:")
    
    # ベクトルストア（埋め込みベクトルを保存）
    vector_store = InMemoryVectorStore()
    
    return document_store, vector_store


def build_simple_corpus(documents, document_store, vector_store):
    """シンプルなコーパスを構築"""
    
    print("📚 コーパスを構築中...")
    
    # ドキュメントを手動でストアに追加
    for doc in documents:
        document_store.store_document(doc)
    
    # ベクトル埋め込みを手動で作成
    embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=embedder_config)
    
    # コーパスでembedderを訓練
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    
    # 各ドキュメントのベクトルを生成してストア
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


def test_questions(query_engine):
    """質問をテストする"""
    
    questions = [
        "RAGとは何ですか？",
        "ベクトル検索の仕組みを教えて",
        "LLMの主な用途は？",
        "RAGの利点を説明してください"
    ]
    
    print("\n" + "="*60)
    print("🔍 質疑応答テスト")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📌 質問 {i}: {question}")
        print("-" * 40)
        
        try:
            result = query_engine.answer(question)
            
            print(f"🤖 回答:")
            # 改行で分割して見やすく表示
            answer_lines = result.answer.split('\n')
            for line in answer_lines:
                if line.strip():
                    print(f"   {line.strip()}")
            
            print(f"\n📊 詳細:")
            print(f"   - 処理時間: {result.metadata.get('processing_time', 0):.3f}秒")
            print(f"   - 参考文書数: {result.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result.confidence:.3f}")
            
            if result.sources:
                print(f"   - 主な参考文書: {result.sources[0].metadata.get('title', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ エラー: {e}")


def show_corpus_stats(document_store, vector_store):
    """コーパスの統計情報を表示"""
    
    print("\n" + "="*60)
    print("📊 コーパス統計情報")
    print("="*60)
    
    # ドキュメント数
    try:
        # SQLiteDocumentStoreの内部実装を使用して文書数を取得
        cursor = document_store.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📄 保存文書数: {doc_count}")
    except:
        print(f"📄 保存文書数: 取得できませんでした")
    
    # ベクトル数
    vector_count = len(vector_store._vectors) if hasattr(vector_store, '_vectors') else 0
    print(f"🔢 ベクトル数: {vector_count}")
    
    # メタデータサンプル
    if vector_count > 0:
        sample_vector = next(iter(vector_store._vectors.values()))
        print(f"📏 ベクトル次元: {len(sample_vector.embedding)}")
        print(f"📋 サンプルメタデータ: {sample_vector.metadata}")


def main():
    """メイン関数"""
    
    print("🚀 基本的なRAGパイプライン チュートリアル")
    print("="*60)
    print("このチュートリアルでは、refinire-ragを使用して")
    print("最もシンプルなRAGシステムを構築し、質疑応答機能をテストします。")
    
    try:
        # ステップ1: サンプルドキュメント作成
        print("\n📝 ステップ1: サンプルドキュメント作成")
        documents = create_sample_documents()
        print(f"✅ {len(documents)}件のサンプルドキュメントを作成")
        
        # ステップ2: ストレージ初期化
        print("\n💾 ステップ2: ストレージ初期化")
        document_store, vector_store = setup_storage()
        print("✅ ドキュメントストアとベクトルストアを初期化")
        
        # ステップ3: コーパス構築
        print("\n🏗️ ステップ3: コーパス構築")
        embedder = build_simple_corpus(documents, document_store, vector_store)
        
        # ステップ4: クエリエンジン作成
        print("\n⚙️ ステップ4: クエリエンジン作成")
        query_engine = create_query_engine(document_store, vector_store, embedder)
        
        # ステップ5: コーパス統計情報
        show_corpus_stats(document_store, vector_store)
        
        # ステップ6: 質疑応答テスト
        print("\n🧪 ステップ5: 質疑応答テスト")
        test_questions(query_engine)
        
        # 成功メッセージ
        print("\n🎉 チュートリアル1が完了しました！")
        print("\n📚 学習内容:")
        print("   ✅ RAGの基本構成要素（ドキュメントストア、ベクトルストア）")
        print("   ✅ Simple RAGパイプライン（Load → Chunk → Vector）")
        print("   ✅ QueryEngine を使った質疑応答")
        print("   ✅ TF-IDF埋め込みとベクトル検索")
        
        print("\n🚀 次のステップ:")
        print("   • チュートリアル2: コーパス管理とドキュメント処理")
        print("   • より高度なドキュメント処理機能")
        print("   • マルチステージパイプライン")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        print("\n🔧 トラブルシューティング:")
        print("   1. 依存関係がインストールされているか確認: pip install -e .")
        print("   2. Python 3.10以上を使用しているか確認")
        print("   3. メモリ容量が十分か確認")
        
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)