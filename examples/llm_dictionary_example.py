#!/usr/bin/env python3
"""
Example: LLM-based Dictionary and Graph Building with Refinire

This example demonstrates how to use Refinire LLM integration
for extracting domain-specific terms and building knowledge graphs.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.models.document import Document
from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig

def main():
    """Demonstrate LLM-based term extraction and knowledge graph building"""
    
    # Sample document about RAG systems
    sample_doc = Document(
        id="rag_intro_001",
        content="""
        RAG（Retrieval-Augmented Generation）は、大規模言語モデル（LLM）の能力を
        外部知識と組み合わせる革新的なアプローチです。
        
        この手法では、まずベクトル検索を使用して関連文書を検索し、
        その情報をコンテキストとしてLLMに提供します。これにより、
        より正確で根拠のある回答を生成することができます。
        
        主要な構成要素：
        - 文書埋め込み（Document Embedding）
        - ベクトルデータベース（Vector Database）
        - 検索機能（Retrieval Function）
        - 生成機能（Generation Function）
        
        評価においては、精度（Precision）、再現率（Recall）、
        F1スコアなどのメトリクスが重要です。
        """,
        metadata={
            "title": "RAGシステムの基礎",
            "author": "技術解説者",
            "domain": "AI・機械学習"
        }
    )
    
    print("🔍 LLM-based Term Extraction and Knowledge Graph Building Example")
    print("=" * 70)
    
    # Step 1: Extract domain-specific terms using LLM
    print("\n📚 Step 1: Extracting domain-specific terms...")
    
    dict_config = DictionaryMakerConfig(
        dictionary_file_path="./examples/domain_dictionary.md",
        llm_model="gpt-4o-mini",  # Using Refinire to call OpenAI
        focus_on_technical_terms=True,
        extract_abbreviations=True,
        detect_expression_variations=True
    )
    
    dictionary_maker = DictionaryMaker(dict_config)
    
    # Process document for term extraction
    enriched_docs = dictionary_maker.process(sample_doc)
    enriched_doc = enriched_docs[0]
    
    print(f"✓ Terms extracted and dictionary updated")
    dict_metadata = enriched_doc.metadata.get("dictionary_metadata", {})
    print(f"  - New terms: {dict_metadata.get('new_terms_extracted', 0)}")
    print(f"  - Variations detected: {dict_metadata.get('variations_detected', 0)}")
    
    # Show dictionary content
    dictionary_content = dictionary_maker.get_dictionary_content()
    print(f"\n📖 Dictionary Content Preview:")
    print(dictionary_content[:500] + "..." if len(dictionary_content) > 500 else dictionary_content)
    
    # Step 2: Normalize expressions using the dictionary
    print("\n🔧 Step 2: Normalizing expression variations...")
    
    norm_config = NormalizerConfig(
        dictionary_file_path="./examples/domain_dictionary.md",
        normalize_variations=True,
        expand_abbreviations=True
    )
    
    normalizer = Normalizer(norm_config)
    normalized_docs = normalizer.process(enriched_doc)
    normalized_doc = normalized_docs[0]
    
    norm_stats = normalized_doc.metadata.get("normalization_stats", {})
    print(f"✓ Text normalized")
    print(f"  - Total replacements: {norm_stats.get('total_replacements', 0)}")
    print(f"  - Variations normalized: {norm_stats.get('variations_normalized', 0)}")
    print(f"  - Abbreviations expanded: {norm_stats.get('abbreviations_expanded', 0)}")
    
    # Step 3: Build knowledge graph using LLM
    print("\n🕸️  Step 3: Building knowledge graph...")
    
    graph_config = GraphBuilderConfig(
        graph_file_path="./examples/domain_knowledge_graph.md",
        dictionary_file_path="./examples/domain_dictionary.md",
        llm_model="gpt-4o-mini",  # Using Refinire to call OpenAI
        focus_on_important_relationships=True,
        extract_hierarchical_relationships=True,
        extract_causal_relationships=True
    )
    
    graph_builder = GraphBuilder(graph_config)
    graph_enriched_docs = graph_builder.process(normalized_doc)
    graph_enriched_doc = graph_enriched_docs[0]
    
    print(f"✓ Knowledge graph updated")
    graph_metadata = graph_enriched_doc.metadata.get("graph_metadata", {})
    print(f"  - Relationships extracted: {graph_metadata.get('new_relationships_extracted', 0)}")
    print(f"  - Duplicates avoided: {graph_metadata.get('duplicates_avoided', 0)}")
    
    # Show graph content
    graph_content = graph_builder.get_graph_content()
    print(f"\n🕸️  Knowledge Graph Content Preview:")
    print(graph_content[:500] + "..." if len(graph_content) > 500 else graph_content)
    
    # Step 4: Display processing statistics
    print("\n📊 Processing Statistics:")
    print("-" * 40)
    
    dict_stats = dictionary_maker.get_extraction_stats()
    print(f"DictionaryMaker:")
    print(f"  - Documents processed: {dict_stats.get('documents_processed', 0)}")
    print(f"  - Terms extracted: {dict_stats.get('terms_extracted', 0)}")
    print(f"  - LLM API calls: {dict_stats.get('llm_api_calls', 0)}")
    
    norm_stats = normalizer.get_normalization_stats()
    print(f"\nNormalizer:")
    print(f"  - Documents processed: {norm_stats.get('documents_processed', 0)}")
    print(f"  - Total replacements: {norm_stats.get('total_replacements', 0)}")
    print(f"  - Mappings loaded: {norm_stats.get('mappings_loaded', 0)}")
    
    graph_stats = graph_builder.get_graph_stats()
    print(f"\nGraphBuilder:")
    print(f"  - Documents processed: {graph_stats.get('documents_processed', 0)}")
    print(f"  - Relationships extracted: {graph_stats.get('relationships_extracted', 0)}")
    print(f"  - LLM API calls: {graph_stats.get('llm_api_calls', 0)}")
    
    print("\n🎉 LLM integration example completed!")
    print("📁 Generated files:")
    print("   - examples/domain_dictionary.md")
    print("   - examples/domain_knowledge_graph.md")
    
    return graph_enriched_doc

if __name__ == "__main__":
    try:
        result_doc = main()
        print("\n✓ Example ran successfully!")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)