#!/usr/bin/env python3
"""
Simple RAG Workflow Test

This is a simplified test that manually builds a corpus and tests the QueryEngine
without the complex pipeline system, to verify core RAG functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.retrieval import SimpleRetrieverConfig, SimpleRerankerConfig, SimpleReaderConfig
from refinire_rag.models.document import Document
from refinire_rag.embedding import TFIDFEmbedder
from refinire_rag.storage.vector_store import VectorEntry


def create_sample_documents() -> list:
    """Create sample documents for testing"""
    
    docs = [
        Document(
            id="doc1",
            content="""
RAG（Retrieval-Augmented Generation）は、検索拡張生成技術の実装です。
このシステムでは、LLM（Large Language Model）と外部知識ベースを組み合わせ、
より正確で信頼性の高い回答を生成します。

主要な利点：
- ハルシネーション（幻覚）の減少
- リアルタイムな知識更新
- 専門ドメインへの適応
- 回答の根拠となるソースの提供

RAGシステムは企業の文書検索、カスタマーサポート、
研究支援などの用途で広く活用されています。
""",
            metadata={"title": "RAG Overview", "type": "overview"}
        ),
        
        Document(
            id="doc2", 
            content="""
ベクトル検索（Vector Search）は、RAGシステムの中核技術です。
文書をベクトル空間に埋め込み、意味的類似性に基づいて検索を行います。

技術要素：
- 文書の埋め込み（Document Embedding）
- ベクトルデータベース（Vector Database）
- 類似度計算（通常はコサイン類似度）
- 近似最近傍探索（ANN: Approximate Nearest Neighbor）

人気のあるベクトルDB：
- Chroma: オープンソースのベクトルデータベース
- Pinecone: マネージドベクトル検索サービス
- Faiss: Facebook AI開発の類似性検索ライブラリ
- Weaviate: GraphQLインターフェースを持つベクトルDB
""",
            metadata={"title": "Vector Search", "type": "technical"}
        ),
        
        Document(
            id="doc3",
            content="""
LLM統合は、RAGシステムの回答生成フェーズで重要な役割を果たします。
検索された文書を文脈として使用し、自然で有用な回答を生成します。

統合のポイント：
- プロンプトエンジニアリング: 効果的な指示文の設計
- 文脈長制限: LLMのトークン制限に対応
- 温度設定: 創造性と正確性のバランス
- ストリーミング応答: リアルタイムな回答生成

主要なLLMプロバイダー：
- OpenAI GPT-4: 高品質な言語理解と生成
- Anthropic Claude: 長文対応と安全性重視
- Google Gemini: マルチモーダル対応
- ローカルLLM: プライバシーとコスト削減
""",
            metadata={"title": "LLM Integration", "type": "technical"}
        )
    ]
    
    return docs


def setup_vector_store(documents: list) -> InMemoryVectorStore:
    """Manually setup vector store with embeddings"""
    
    print("📚 Setting up vector store with embeddings...")
    
    # Create vector store and embedder
    vector_store = InMemoryVectorStore()
    from refinire_rag.embedding import TFIDFEmbeddingConfig
    config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)  # Adjust for small corpus
    embedder = TFIDFEmbedder(config=config)
    
    # Fit embedder on document corpus
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    print(f"   ✅ Fitted TF-IDF embedder on {len(corpus_texts)} documents")
    
    # Add documents to vector store
    for doc in documents:
        # Generate embedding
        embedding_result = embedder.embed_text(doc.content)
        
        # Create vector entry
        vector_entry = VectorEntry(
            document_id=doc.id,
            content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            embedding=embedding_result.vector.tolist(),
            metadata=doc.metadata
        )
        
        # Add to vector store
        vector_store.add_vector(vector_entry)
        print(f"   ✅ Added embedding for {doc.id}: {doc.metadata.get('title', 'No title')}")
    
    print(f"📊 Vector store setup completed with {len(documents)} documents")
    return vector_store


def setup_document_store(documents: list) -> SQLiteDocumentStore:
    """Setup document store"""
    
    print("📄 Setting up document store...")
    
    doc_store = SQLiteDocumentStore(":memory:")
    
    # Add documents to document store
    for doc in documents:
        doc_store.store_document(doc)
        print(f"   ✅ Stored document {doc.id}: {doc.metadata.get('title', 'No title')}")
    
    print(f"📊 Document store setup completed with {len(documents)} documents")
    return doc_store


def test_retrieval_components(vector_store: InMemoryVectorStore, documents: list):
    """Test individual retrieval components"""
    
    print("\n" + "="*60)
    print("🔧 COMPONENT TESTING")
    print("="*60)
    
    # Test Retriever
    print("\n📌 Testing Retriever...")
    from refinire_rag.embedding import TFIDFEmbeddingConfig
    config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=config)
    
    # Fit embedder with document corpus
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    
    query = "RAGとは何ですか？"
    search_results = retriever.retrieve(query, limit=3)
    
    print(f"   Query: {query}")
    print(f"   Results: {len(search_results)}")
    for i, result in enumerate(search_results, 1):
        print(f"     {i}. Doc {result.document_id} (score: {result.score:.3f})")
        print(f"        {result.document.content[:100]}...")
    
    # Test Reranker
    print("\n📌 Testing Reranker...")
    reranker = SimpleReranker()
    reranked_results = reranker.rerank(query, search_results)
    
    print(f"   Reranked from {len(search_results)} to {len(reranked_results)} results")
    for i, result in enumerate(reranked_results, 1):
        print(f"     {i}. Doc {result.document_id} (score: {result.score:.3f})")
    
    # Test Reader
    print("\n📌 Testing Reader...")
    reader = SimpleReader()
    answer = reader.read(query, reranked_results)
    
    print(f"   Generated answer ({len(answer)} chars):")
    print(f"     {answer}")
    
    return search_results, reranked_results, answer


def test_query_engine(doc_store: SQLiteDocumentStore, vector_store: InMemoryVectorStore):
    """Test complete QueryEngine workflow"""
    
    print("\n" + "="*60)
    print("🤖 QUERY ENGINE TESTING") 
    print("="*60)
    
    # Create components  
    from refinire_rag.embedding import TFIDFEmbeddingConfig
    config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=config)
    
    # Fit embedder - we need to recreate the corpus texts
    # Since we know the documents, recreate them
    documents = create_sample_documents()
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # Create QueryEngine
    query_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker
    )
    
    print(f"✅ QueryEngine initialized")
    print(f"   Corpus state: {query_engine.corpus_state}")
    
    # Test queries
    queries = [
        "RAGとは何ですか？",
        "ベクトル検索の仕組みを教えて",
        "LLM統合のポイントは？",
        "おすすめのベクトルデータベースを教えて"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📌 Query {i}: {query}")
        print("-" * 40)
        
        try:
            result = query_engine.answer(query)
            
            print(f"🤖 回答:")
            # Split answer into lines for better formatting
            answer_lines = result.answer.split('\n')
            for line in answer_lines:
                if line.strip():
                    print(f"   {line}")
            
            print(f"\n📊 メタデータ:")
            print(f"   - 処理時間: {result.metadata.get('processing_time', 0):.3f}s")
            print(f"   - ソース数: {result.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result.confidence:.3f}")
            
            if result.sources:
                print(f"   - ソース:")
                for j, source in enumerate(result.sources[:3], 1):
                    source_title = source.metadata.get('title', f'Document {source.document_id}')
                    print(f"     {j}. {source_title} (score: {source.score:.3f})")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Display engine statistics
    print(f"\n📈 Engine Statistics:")
    stats = query_engine.get_engine_stats()
    print(f"   - Total queries: {stats.get('queries_processed', 0)}")
    print(f"   - Average response time: {stats.get('average_response_time', 0):.3f}s")
    print(f"   - Average retrieval count: {stats.get('average_retrieval_count', 0):.1f}")


def main():
    """Main test function"""
    
    print("🚀 Simple RAG Workflow Test")
    print("="*60)
    print("Testing core RAG functionality with manually built corpus")
    
    try:
        # Create sample documents
        print("\n📝 Creating sample documents...")
        documents = create_sample_documents()
        print(f"Created {len(documents)} sample documents")
        
        # Setup stores
        vector_store = setup_vector_store(documents)
        doc_store = setup_document_store(documents)
        
        # Test individual components
        test_retrieval_components(vector_store, documents)
        
        # Test complete QueryEngine
        test_query_engine(doc_store, vector_store)
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)