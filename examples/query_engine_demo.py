#!/usr/bin/env python3
"""
QueryEngine Demo - Complete RAG pipeline demonstration

This example demonstrates the QueryEngine with automatic corpus state detection,
query normalization, and answer generation using simple component implementations.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.retrieval import SimpleRetrieverConfig, SimpleRerankerConfig, SimpleReaderConfig
from refinire_rag.models.document import Document


def create_sample_files(temp_dir: Path) -> list:
    """Create sample files for RAG demo"""
    
    files = []
    
    # Sample file 1: RAG basics
    file1 = temp_dir / "rag_basics.txt"
    file1.write_text("""
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
""", encoding='utf-8')
    files.append(str(file1))
    
    # Sample file 2: Vector search
    file2 = temp_dir / "vector_search.txt"
    file2.write_text("""
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
""", encoding='utf-8')
    files.append(str(file2))
    
    # Sample file 3: LLM integration
    file3 = temp_dir / "llm_integration.txt"
    file3.write_text("""
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
""", encoding='utf-8')
    files.append(str(file3))
    
    return files


def setup_corpus(temp_dir: Path, file_paths: list):
    """Setup corpus with semantic RAG pipeline"""
    
    print("\n📚 Setting up corpus with semantic RAG...")
    
    # Initialize stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Create semantic RAG corpus
    corpus_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    stats = corpus_manager.build_corpus(file_paths)
    
    print(f"✅ Corpus setup completed:")
    print(f"   - Documents: {stats.total_documents_created}")
    print(f"   - Chunks: {stats.total_chunks_created}")
    print(f"   - Processing time: {stats.total_processing_time:.3f}s")
    
    return doc_store, vector_store


def demo_basic_queries(query_engine: QueryEngine):
    """Demo basic query processing"""
    
    print("\n" + "="*60)
    print("🔍 BASIC QUERY DEMO")
    print("="*60)
    
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
            print(f"   {result.answer}")
            print(f"")
            print(f"📊 メタデータ:")
            print(f"   - 処理時間: {result.metadata.get('processing_time', 0):.3f}s")
            print(f"   - ソース数: {result.metadata.get('source_count', 0)}")
            print(f"   - 信頼度: {result.confidence:.3f}")
            print(f"   - クエリ正規化: {'Yes' if result.metadata.get('query_normalized') else 'No'}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")


def demo_engine_stats(query_engine: QueryEngine):
    """Demo engine statistics"""
    
    print("\n" + "="*60)
    print("📈 ENGINE STATISTICS")
    print("="*60)
    
    stats = query_engine.get_engine_stats()
    
    print(f"\n🔧 Engine Configuration:")
    config_info = stats.get('config', {})
    for key, value in config_info.items():
        print(f"   - {key}: {value}")
    
    print(f"\n📊 Processing Statistics:")
    print(f"   - Queries processed: {stats.get('queries_processed', 0)}")
    print(f"   - Queries normalized: {stats.get('queries_normalized', 0)}")
    print(f"   - Average response time: {stats.get('average_response_time', 0):.3f}s")
    print(f"   - Average retrieval count: {stats.get('average_retrieval_count', 0):.1f}")
    
    print(f"\n🏗️ Corpus State:")
    corpus_state = stats.get('corpus_state', {})
    for key, value in corpus_state.items():
        print(f"   - {key}: {value}")
    
    print(f"\n🔍 Component Statistics:")
    component_stats = ['retriever_stats', 'reranker_stats', 'reader_stats']
    for component in component_stats:
        if component in stats:
            comp_stats = stats[component]
            comp_name = component.replace('_stats', '').title()
            print(f"   {comp_name}:")
            component_prefix = component.split("_")[0]
            type_key = f'{component_prefix}_type'
            print(f"     - Type: {comp_stats.get(type_key, 'Unknown')}")
            print(f"     - Queries processed: {comp_stats.get('queries_processed', 0)}")
            print(f"     - Processing time: {comp_stats.get('processing_time', 0):.3f}s")


def demo_custom_configurations():
    """Demo custom component configurations"""
    
    print("\n" + "="*60)
    print("⚙️  CUSTOM CONFIGURATION DEMO")
    print("="*60)
    
    # Create stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Custom component configurations
    retriever_config = SimpleRetrieverConfig(
        top_k=8,
        similarity_threshold=0.1,
        embedding_model="text-embedding-3-small"
    )
    
    reranker_config = SimpleRerankerConfig(
        top_k=3,
        boost_exact_matches=True,
        length_penalty_factor=0.2
    )
    
    reader_config = SimpleReaderConfig(
        llm_model="gpt-4o-mini",
        max_context_length=1500,
        temperature=0.2,
        include_sources=True
    )
    
    # Create components
    retriever = SimpleRetriever(vector_store, config=retriever_config)
    reranker = SimpleReranker(config=reranker_config)
    reader = SimpleReader(config=reader_config)
    
    # Custom query engine configuration
    engine_config = QueryEngineConfig(
        enable_query_normalization=True,
        auto_detect_corpus_state=True,
        retriever_top_k=8,
        reranker_top_k=3,
        include_sources=True,
        include_confidence=True
    )
    
    # Create query engine
    custom_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker,
        config=engine_config
    )
    
    print(f"✅ Custom QueryEngine created with:")
    print(f"   - Retriever: top_k={retriever_config.top_k}, threshold={retriever_config.similarity_threshold}")
    print(f"   - Reranker: top_k={reranker_config.top_k}, exact_match_boost={reranker_config.boost_exact_matches}")
    print(f"   - Reader: model={reader_config.llm_model}, context_length={reader_config.max_context_length}")
    
    return custom_engine


def main():
    """Main demo function"""
    
    print("🚀 QueryEngine Complete RAG Pipeline Demo")
    print("="*60)
    print("Demonstrating:")
    print("• Corpus building with semantic normalization")
    print("• Automatic corpus state detection")
    print("• Query normalization")
    print("• Vector retrieval → Reranking → Answer generation")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create sample files
        print(f"\n📁 Creating sample files in: {temp_dir}")
        file_paths = create_sample_files(temp_dir)
        print(f"Created {len(file_paths)} sample files")
        
        # Setup corpus
        doc_store, vector_store = setup_corpus(temp_dir, file_paths)
        
        # Create QueryEngine with simple components
        retriever = SimpleRetriever(vector_store)
        reranker = SimpleReranker()
        reader = SimpleReader()
        
        query_engine = QueryEngine(
            document_store=doc_store,
            vector_store=vector_store,
            retriever=retriever,
            reader=reader,
            reranker=reranker
        )
        
        print(f"\n🤖 QueryEngine initialized:")
        print(f"   - Corpus state detected: {query_engine.corpus_state}")
        
        # Demo basic queries
        demo_basic_queries(query_engine)
        
        # Demo engine statistics
        demo_engine_stats(query_engine)
        
        # Demo custom configurations
        custom_engine = demo_custom_configurations()
        
        print("\n🎉 All QueryEngine demos completed successfully!")
        print(f"📁 Generated files: {temp_dir}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up (comment out to inspect files)
        # shutil.rmtree(temp_dir)
        pass
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)