#!/usr/bin/env python3
"""
Query Normalization Test

Test query normalization functionality in QueryEngine with a 
normalized corpus that has dictionary-based term standardization.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.models.document import Document


def create_sample_files_with_variations(temp_dir: Path) -> list:
    """Create sample files with term variations for normalization testing"""
    
    files = []
    
    # File 1: Uses "RAG" and various expressions
    file1 = temp_dir / "rag_doc1.txt"
    file1.write_text("""
RAG（Retrieval-Augmented Generation）は革新的な技術です。
検索拡張生成システムとして、LLMと知識ベースを統合します。
このRAGシステムは企業で広く使われています。
""", encoding='utf-8')
    files.append(str(file1))
    
    # File 2: Uses variations like "検索強化生成"
    file2 = temp_dir / "rag_doc2.txt" 
    file2.write_text("""
検索強化生成技術は最新のAI手法です。
検索拡張生成とも呼ばれ、情報検索と生成を組み合わせます。
LLM（大規模言語モデル）がコアコンポーネントです。
""", encoding='utf-8')
    files.append(str(file2))
    
    # File 3: Uses "ベクトル検索" and "意味検索"
    file3 = temp_dir / "vector_doc.txt"
    file3.write_text("""
ベクトル検索は意味的類似性を基にした検索手法です。
セマンティック検索とも呼ばれます。
文書埋め込みを使って意味検索を実現します。
""", encoding='utf-8')
    files.append(str(file3))
    
    return files


def create_test_dictionary(temp_dir: Path) -> str:
    """Create a test dictionary file for normalization"""
    
    dict_file = temp_dir / "test_dictionary.md"
    dict_file.write_text("""# ドメイン用語辞書

## 技術用語

- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, RAGシステム

- **ベクトル検索** (Vector Search): ベクトル検索  
  - 表現揺らぎ: ベクトル検索, 意味検索, セマンティック検索, 意味的検索

- **LLM** (Large Language Model): 大規模言語モデル
  - 表現揺らぎ: 大規模言語モデル, 言語モデル, LLMモデル

- **埋め込み** (Embedding): 埋め込み
  - 表現揺らぎ: 埋め込み, エンベディング, ベクトル表現
""", encoding='utf-8')
    
    return str(dict_file)


def build_normalized_corpus(temp_dir: Path, file_paths: list, dict_path: str):
    """Build a corpus with normalization using dictionary"""
    
    print("📚 Building normalized corpus...")
    
    # Initialize stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Create semantic RAG manager (includes normalization)
    corpus_manager = CorpusManager.create_semantic_rag(doc_store, vector_store)
    
    # Configure with custom dictionary
    stage_configs = {
        "dictionary_config": {
            "dictionary_file_path": dict_path,
            "focus_on_technical_terms": True
        },
        "normalizer_config": {
            "dictionary_file_path": dict_path,
            "normalize_variations": True,
            "expand_abbreviations": True
        }
    }
    
    try:
        stats = corpus_manager.build_corpus(file_paths, stage_configs=stage_configs)
        print(f"✅ Corpus built with normalization:")
        print(f"   - Documents: {stats.total_documents_created}")
        print(f"   - Processing time: {stats.total_processing_time:.3f}s")
        print(f"   - Stages executed: {stats.pipeline_stages_executed}")
    except Exception as e:
        print(f"❌ Corpus building failed: {e}")
        # Fallback: Build simple corpus
        print("🔄 Falling back to simple corpus...")
        simple_manager = CorpusManager.create_simple_rag(doc_store, vector_store)
        stats = simple_manager.build_corpus(file_paths)
    
    return doc_store, vector_store


def test_query_normalization(doc_store, vector_store, dict_path: str):
    """Test query normalization with various query expressions"""
    
    print("\n" + "="*60)
    print("🔍 QUERY NORMALIZATION TEST")
    print("="*60)
    
    # Create QueryEngine with normalization enabled
    config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
    embedder = TFIDFEmbedder(config=config)
    
    # Manually fit embedder (simplified for testing)
    sample_texts = ["RAG検索拡張生成", "ベクトル検索意味検索", "LLM大規模言語モデル"]
    embedder.fit(sample_texts)
    
    retriever = SimpleRetriever(vector_store, embedder=embedder)
    reranker = SimpleReranker()
    reader = SimpleReader()
    
    query_config = QueryEngineConfig(
        enable_query_normalization=True,
        auto_detect_corpus_state=True,
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
    
    print(f"🤖 QueryEngine initialized")
    print(f"   Corpus state: {query_engine.corpus_state}")
    print(f"   Query normalization: {'Enabled' if query_engine.normalizer else 'Disabled'}")
    
    # Test queries with variations that should be normalized
    test_queries = [
        {
            "query": "検索強化生成について教えて", 
            "expected_normalization": "検索拡張生成について教えて",
            "description": "検索強化生成 → 検索拡張生成"
        },
        {
            "query": "意味検索の仕組みは？",
            "expected_normalization": "ベクトル検索の仕組みは？", 
            "description": "意味検索 → ベクトル検索"
        },
        {
            "query": "RAGシステムの利点を説明して",
            "expected_normalization": "検索拡張生成の利点を説明して",
            "description": "RAGシステム → 検索拡張生成"
        },
        {
            "query": "セマンティック検索とは何ですか？",
            "expected_normalization": "ベクトル検索とは何ですか？",
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
            
            # Check if query was normalized
            normalized = result.metadata.get("query_normalized", False)
            normalized_query = result.normalized_query
            
            if normalized and normalized_query:
                print(f"   ✅ Normalized: {normalized_query}")
                print(f"   🔄 Normalization: {'Success' if normalized_query != query else 'No change'}")
            else:
                print(f"   ❌ No normalization applied")
            
            print(f"   🤖 Answer: {result.answer[:100]}{'...' if len(result.answer) > 100 else ''}")
            print(f"   📊 Sources: {result.metadata.get('source_count', 0)}")
            print(f"   ⏱️  Time: {result.metadata.get('processing_time', 0):.3f}s")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
    
    # Display normalization statistics
    print(f"\n📈 Normalization Statistics:")
    stats = query_engine.get_engine_stats()
    print(f"   - Total queries: {stats.get('queries_processed', 0)}")
    print(f"   - Normalized queries: {stats.get('queries_normalized', 0)}")
    print(f"   - Normalization rate: {stats.get('queries_normalized', 0) / max(stats.get('queries_processed', 1), 1) * 100:.1f}%")
    
    if query_engine.normalizer:
        normalizer_stats = stats.get('normalizer_stats', {})
        print(f"   - Normalizer processing time: {normalizer_stats.get('processing_time', 0):.3f}s")


def main():
    """Main test function"""
    
    print("🚀 Query Normalization Test")
    print("="*60)
    print("Testing automatic query normalization with dictionary-based term standardization")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test files and dictionary
        print("\n📝 Creating test files with term variations...")
        file_paths = create_sample_files_with_variations(temp_dir)
        dict_path = create_test_dictionary(temp_dir)
        
        print(f"Created {len(file_paths)} files with term variations")
        print(f"Created dictionary: {dict_path}")
        
        # Build normalized corpus
        doc_store, vector_store = build_normalized_corpus(temp_dir, file_paths, dict_path)
        
        # Test query normalization
        test_query_normalization(doc_store, vector_store, dict_path)
        
        print("\n🎉 Query normalization test completed!")
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