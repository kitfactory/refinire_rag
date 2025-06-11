#!/usr/bin/env python3
"""
Manual Query Normalization Test

Test query normalization functionality by manually setting up 
the normalizer to demonstrate the feature works correctly.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.use_cases.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.document import Document
from refinire_rag.storage.vector_store import VectorEntry


def create_test_dictionary(temp_dir: Path) -> str:
    """Create a test dictionary file for normalization"""
    
    dict_file = temp_dir / "test_dictionary.md"
    dict_file.write_text("""# ドメイン用語辞書

## 技術用語

- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, RAGシステム, 検索拡張技術

- **ベクトル検索** (Vector Search): ベクトル検索  
  - 表現揺らぎ: ベクトル検索, 意味検索, セマンティック検索, 意味的検索

- **LLM** (Large Language Model): 大規模言語モデル
  - 表現揺らぎ: 大規模言語モデル, 言語モデル, LLMモデル

- **埋め込み** (Embedding): 埋め込み
  - 表現揺らぎ: 埋め込み, エンベディング, ベクトル表現
""", encoding='utf-8')
    
    return str(dict_file)


def create_sample_documents() -> list:
    """Create sample documents for testing"""
    
    docs = [
        Document(
            id="doc1",
            content="""
検索拡張生成（RAG）は革新的なAI技術です。
大規模言語モデルと外部知識ベースを統合し、
より正確で根拠のある回答を生成します。
この技術は企業の情報検索システムで広く採用されています。
""",
            metadata={"title": "RAG Technology", "type": "overview"}
        ),
        
        Document(
            id="doc2", 
            content="""
ベクトル検索は意味的類似性に基づく検索技術です。
文書を高次元ベクトル空間に埋め込み、
コサイン類似度などを用いて関連文書を発見します。
従来のキーワード検索とは異なり、文脈を理解した検索が可能です。
""",
            metadata={"title": "Vector Search", "type": "technical"}
        ),
        
        Document(
            id="doc3",
            content="""
大規模言語モデル（LLM）は自然言語処理の中核技術です。
GPT、Claude、Geminiなどの先進的なモデルが存在します。
これらのモデルは文章生成、翻訳、要約などの
幅広いタスクに対応できます。
""",
            metadata={"title": "LLM Overview", "type": "technical"}
        )
    ]
    
    return docs


def setup_manual_corpus(documents: list) -> tuple:
    """Manually setup corpus with vector embeddings"""
    
    print("📚 Setting up manual corpus...")
    
    # Create stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Setup embedder
    config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=config)
    
    # Fit embedder
    corpus_texts = [doc.content for doc in documents]
    embedder.fit(corpus_texts)
    
    # Add documents to stores
    for doc in documents:
        # Store in document store
        doc_store.store_document(doc)
        
        # Generate embedding and store in vector store
        embedding_result = embedder.embed_text(doc.content)
        vector_entry = VectorEntry(
            document_id=doc.id,
            content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
            embedding=embedding_result.vector.tolist(),
            metadata=doc.metadata
        )
        vector_store.add_vector(vector_entry)
        
        print(f"   ✅ Added {doc.id}: {doc.metadata.get('title', 'No title')}")
    
    print(f"📊 Manual corpus setup completed with {len(documents)} documents")
    return doc_store, vector_store, embedder


def test_normalizer_standalone(dict_path: str):
    """Test the Normalizer component standalone"""
    
    print("\n" + "="*60)
    print("🔧 STANDALONE NORMALIZER TEST")
    print("="*60)
    
    # Create normalizer
    normalizer_config = NormalizerConfig(
        dictionary_file_path=dict_path,
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False  # Disable for Japanese text
    )
    
    normalizer = Normalizer(normalizer_config)
    
    # Test queries
    test_queries = [
        "検索強化生成について教えて",
        "意味検索の仕組みは？", 
        "RAGシステムの利点を説明して",
        "セマンティック検索とは何ですか？",
        "LLMモデルの特徴は？"
    ]
    
    print(f"📖 Dictionary loaded: {dict_path}")
    print(f"🔧 Normalizer configured")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📌 Test {i}:")
        print(f"   Original: {query}")
        
        try:
            # Create query document
            query_doc = Document(
                id=f"query_{i}",
                content=query,
                metadata={"is_query": True}
            )
            
            # Normalize
            normalized_docs = normalizer.process(query_doc)
            
            if normalized_docs:
                normalized_query = normalized_docs[0].content
                if normalized_query != query:
                    print(f"   ✅ Normalized: {normalized_query}")
                    print(f"   🔄 Change: Yes")
                else:
                    print(f"   ⚪ Normalized: {normalized_query}")
                    print(f"   🔄 Change: No")
            else:
                print(f"   ❌ Normalization failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def test_query_engine_with_manual_normalizer(doc_store, vector_store, embedder, dict_path: str):
    """Test QueryEngine with manually configured normalizer"""
    
    print("\n" + "="*60)
    print("🤖 QUERY ENGINE WITH MANUAL NORMALIZER")
    print("="*60)
    
    # Create query engine components
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    # Create query engine
    query_config = QueryEngineConfig(
        enable_query_normalization=True,
        auto_detect_corpus_state=False,  # Disable auto-detection
        include_processing_metadata=True
    )
    
    query_engine = QueryEngine(
        document_store=doc_store,
        vector_store=vector_store,
        retriever=retriever,
        reader=reader,
        reranker=reranker,
        config=query_config
    )
    
    # Manually set up normalizer
    normalizer_config = NormalizerConfig(
        dictionary_file_path=dict_path,
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False  # Disable for Japanese text
    )
    query_engine.normalizer = Normalizer(normalizer_config)
    
    # Update corpus state to indicate normalization is available
    query_engine.corpus_state = {
        "has_normalization": True,
        "dictionary_path": dict_path,
        "manual_setup": True
    }
    
    print(f"🤖 QueryEngine configured with manual normalizer")
    print(f"   Normalizer: {'Enabled' if query_engine.normalizer else 'Disabled'}")
    print(f"   Dictionary: {dict_path}")
    
    # Test queries with variations
    test_queries = [
        {
            "query": "検索強化生成について教えて", 
            "description": "検索強化生成 → 検索拡張生成"
        },
        {
            "query": "意味検索の仕組みは？",
            "description": "意味検索 → ベクトル検索"
        },
        {
            "query": "RAGシステムの利点を説明して",
            "description": "RAGシステム → 検索拡張生成"
        },
        {
            "query": "セマンティック検索とは何ですか？",
            "description": "セマンティック検索 → ベクトル検索"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n📌 Test {i}: {description}")
        print(f"   Original: {query}")
        
        try:
            result = query_engine.answer(query)
            
            # Check normalization results
            normalized = result.metadata.get("query_normalized", False)
            normalized_query = result.normalized_query
            
            if normalized and normalized_query:
                print(f"   ✅ Normalized: {normalized_query}")
                print(f"   🔄 Applied: Yes")
            else:
                print(f"   ❌ Normalized: {normalized_query or 'None'}")
                print(f"   🔄 Applied: No")
            
            # Show results
            print(f"   🤖 Answer: {result.answer[:100]}{'...' if len(result.answer) > 100 else ''}")
            print(f"   📊 Sources: {result.metadata.get('source_count', 0)}")
            print(f"   ⏱️  Time: {result.metadata.get('processing_time', 0):.3f}s")
            
            if result.sources:
                print(f"   📄 Top source: {result.sources[0].metadata.get('title', 'Unknown')}")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Show statistics
    print(f"\n📈 Final Statistics:")
    stats = query_engine.get_engine_stats()
    print(f"   - Total queries: {stats.get('queries_processed', 0)}")
    print(f"   - Normalized queries: {stats.get('queries_normalized', 0)}")
    print(f"   - Normalization rate: {stats.get('queries_normalized', 0) / max(stats.get('queries_processed', 1), 1) * 100:.1f}%")


def main():
    """Main test function"""
    
    print("🚀 Manual Query Normalization Test")
    print("="*60)
    print("Testing query normalization with manually configured components")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test dictionary
        print("\n📖 Creating test dictionary...")
        dict_path = create_test_dictionary(temp_dir)
        print(f"Dictionary created: {dict_path}")
        
        # Test standalone normalizer first
        test_normalizer_standalone(dict_path)
        
        # Create sample documents and setup corpus
        print("\n📚 Setting up test corpus...")
        documents = create_sample_documents()
        doc_store, vector_store, embedder = setup_manual_corpus(documents)
        
        # Test query engine with manual normalizer
        test_query_engine_with_manual_normalizer(doc_store, vector_store, embedder, dict_path)
        
        print("\n🎉 Manual query normalization test completed!")
        print(f"📁 Test files: {temp_dir}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
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