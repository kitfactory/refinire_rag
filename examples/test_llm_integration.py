#!/usr/bin/env python3
"""
Test script for Refinire LLM integration in refinire-rag

This script tests the LLM integration in DictionaryMaker and GraphBuilder
to verify that Refinire is properly integrated for term extraction and
relationship extraction tasks.
"""

import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.models.document import Document
from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dictionary_maker_llm():
    """Test DictionaryMaker with Refinire LLM integration"""
    print("\n=== Testing DictionaryMaker with Refinire LLM ===")
    
    # Create test document
    test_doc = Document(
        id="test_doc_001",
        content="""
        RAGシステムは検索拡張生成（Retrieval-Augmented Generation）の手法です。
        これはベクトル検索とLLMを組み合わせた技術で、セマンティック検索により
        関連文書を見つけ、その情報を基に回答を生成します。
        NLI（Natural Language Inference）推論により矛盾を検出することも可能です。
        """,
        metadata={"title": "RAGシステムの概要", "source": "test"}
    )
    
    # Configure DictionaryMaker
    config = DictionaryMakerConfig(
        dictionary_file_path="./test_dictionary.md",
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        focus_on_technical_terms=True,
        extract_abbreviations=True
    )
    
    # Initialize DictionaryMaker
    dictionary_maker = DictionaryMaker(config)
    
    # Process document
    try:
        result_docs = dictionary_maker.process(test_doc)
        
        print(f"✓ DictionaryMaker processed successfully")
        print(f"  - Input document ID: {test_doc.id}")
        print(f"  - Output documents: {len(result_docs)}")
        
        if result_docs:
            output_doc = result_docs[0]
            dict_metadata = output_doc.metadata.get("dictionary_metadata", {})
            print(f"  - Terms extracted: {dict_metadata.get('new_terms_extracted', 0)}")
            print(f"  - Variations detected: {dict_metadata.get('variations_detected', 0)}")
            
        # Get processing stats
        stats = dictionary_maker.get_extraction_stats()
        print(f"  - Processing stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ DictionaryMaker failed: {e}")
        return False

def test_graph_builder_llm():
    """Test GraphBuilder with Refinire LLM integration"""
    print("\n=== Testing GraphBuilder with Refinire LLM ===")
    
    # Create test document with some dictionary metadata
    test_doc = Document(
        id="test_doc_002",
        content="""
        RAG（Retrieval-Augmented Generation）システムは複数のコンポーネントから構成されます。
        主要な構成要素として、文書検索機能、ベクトル埋め込み、LLMによる生成機能があります。
        評価システムでは精度メトリクスを計算し、品質を測定します。
        """,
        metadata={
            "title": "RAGシステムの構成要素",
            "source": "test",
            "dictionary_metadata": {
                "dictionary_file_path": "./test_dictionary.md",
                "new_terms_extracted": 3
            }
        }
    )
    
    # Configure GraphBuilder
    config = GraphBuilderConfig(
        graph_file_path="./test_knowledge_graph.md",
        dictionary_file_path="./test_dictionary.md",
        llm_model="gpt-4o-mini",
        llm_temperature=0.3,
        focus_on_important_relationships=True,
        extract_hierarchical_relationships=True
    )
    
    # Initialize GraphBuilder
    graph_builder = GraphBuilder(config)
    
    # Process document
    try:
        result_docs = graph_builder.process(test_doc)
        
        print(f"✓ GraphBuilder processed successfully")
        print(f"  - Input document ID: {test_doc.id}")
        print(f"  - Output documents: {len(result_docs)}")
        
        if result_docs:
            output_doc = result_docs[0]
            graph_metadata = output_doc.metadata.get("graph_metadata", {})
            print(f"  - Relationships extracted: {graph_metadata.get('new_relationships_extracted', 0)}")
            print(f"  - Duplicates avoided: {graph_metadata.get('duplicates_avoided', 0)}")
            
        # Get processing stats
        stats = graph_builder.get_graph_stats()
        print(f"  - Processing stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ GraphBuilder failed: {e}")
        return False

def main():
    """Main test function"""
    print("Testing Refinire LLM Integration in refinire-rag")
    print("=" * 50)
    
    # Test availability of Refinire
    try:
        from refinire import get_llm
        print("✓ Refinire library is available")
        
        # Try to get a simple LLM instance
        try:
            llm = get_llm("gpt-4o-mini")
            print("✓ Refinire LLM initialization successful")
            
            # Test basic completion
            response = llm.complete("Hello, this is a test.")
            print(f"✓ Basic LLM completion works: {response[:50]}...")
            
        except Exception as e:
            print(f"⚠ LLM initialization or test failed: {e}")
            print("  This might be due to missing API keys or configuration")
            
    except ImportError:
        print("✗ Refinire library not available")
        print("  Please install Refinire: pip install refinire")
        return False
    
    # Run component tests
    dict_success = test_dictionary_maker_llm()
    graph_success = test_graph_builder_llm()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  DictionaryMaker: {'✓ PASS' if dict_success else '✗ FAIL'}")
    print(f"  GraphBuilder: {'✓ PASS' if graph_success else '✗ FAIL'}")
    
    if dict_success and graph_success:
        print("\n🎉 All LLM integration tests passed!")
        return True
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)