#!/usr/bin/env python3
"""
DocumentProcessor Demo - Comprehensive example of unified document processing

This example demonstrates the complete DocumentProcessor architecture,
showing how different processors work together in a unified pipeline.
"""

import sys
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.models.document import Document
from refinire_rag.processing.document_pipeline import DocumentPipeline
from refinire_rag.processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
from refinire_rag.processing.graph_builder import GraphBuilder, GraphBuilderConfig
from refinire_rag.processing.chunker import Chunker, ChunkingConfig


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration"""
    
    documents = [
        Document(
            id="rag_overview_001",
            content="""
            RAG（Retrieval-Augmented Generation）は、検索拡張生成と呼ばれる
            AI技術の革新的なアプローチです。この手法では、大規模言語モデル（LLM）の
            能力を外部知識ベースと組み合わせることで、より正確で根拠のある回答を
            生成することができます。
            
            RAGシステムの主要な構成要素は以下の通りです：
            - 文書埋め込み（Document Embedding）
            - ベクトルデータベース（Vector Database） 
            - 検索機能（Retrieval Function）
            - 生成機能（Generation Function）
            
            このアプローチにより、ハルシネーション（幻覚）の問題を大幅に軽減し、
            信頼性の高いAIシステムを構築することが可能になります。
            """,
            metadata={
                "title": "RAGシステムの概要",
                "author": "AI研究者",
                "domain": "機械学習",
                "creation_date": "2024-01-15",
                "source": "技術文書"
            }
        ),
        
        Document(
            id="vector_search_002", 
            content="""
            ベクトル検索（Vector Search）は、セマンティック検索とも呼ばれ、
            従来のキーワードベース検索とは根本的に異なるアプローチです。
            
            この技術では、文書やクエリを高次元ベクトル空間に埋め込み、
            類似度計算により関連性を判定します。主な利点として、
            - 意味的類似性の検索が可能
            - 多言語対応
            - 同義語や表現の揺らぎに対応
            
            実装においては、Chroma、Faiss、Pineconeなどのベクトルデータベースが
            広く使用されています。また、埋め込みモデルとしては、
            OpenAIのtext-embedding-ada-002や、オープンソースの
            all-MiniLM-L6-v2などが人気です。
            """,
            metadata={
                "title": "ベクトル検索の技術詳細",
                "author": "検索エンジニア", 
                "domain": "情報検索",
                "creation_date": "2024-01-20",
                "source": "技術ブログ"
            }
        ),
        
        Document(
            id="evaluation_metrics_003",
            content="""
            RAGシステムの評価は、従来の機械学習モデルとは異なる課題があります。
            主要な評価指標として以下が挙げられます：
            
            【検索性能の評価】
            - 精度（Precision）: 検索された文書の中で関連性のある文書の割合
            - 再現率（Recall）: 関連性のある全文書の中で検索された文書の割合  
            - F1スコア: 精度と再現率の調和平均
            - NDCG（Normalized Discounted Cumulative Gain）: ランキング品質の評価
            
            【生成品質の評価】
            - BLEU Score: 参照回答との類似度
            - ROUGE Score: 要約品質の評価
            - 人手評価: 流暢さ、正確性、有用性
            
            また、RAG特有の課題として、検索した文書と生成された回答の
            一貫性や矛盾の検出も重要です。NLI（Natural Language Inference）
            モデルを使用した自動矛盾検出システムの研究も活発です。
            """,
            metadata={
                "title": "RAGシステム評価指標",
                "author": "MLエンジニア",
                "domain": "機械学習評価",
                "creation_date": "2024-01-25", 
                "source": "学術論文"
            }
        )
    ]
    
    return documents


def setup_processing_pipeline() -> DocumentPipeline:
    """Set up the complete document processing pipeline"""
    
    # Configure each processor
    dictionary_config = DictionaryMakerConfig(
        dictionary_file_path="./examples/demo_dictionary.md",
        llm_model="gpt-4o-mini",
        focus_on_technical_terms=True,
        extract_abbreviations=True,
        detect_expression_variations=True,
        min_term_importance="medium"
    )
    
    normalizer_config = NormalizerConfig(
        dictionary_file_path="./examples/demo_dictionary.md",
        normalize_variations=True,
        expand_abbreviations=True,
        standardize_technical_terms=True,
        whole_word_only=True
    )
    
    graph_config = GraphBuilderConfig(
        graph_file_path="./examples/demo_knowledge_graph.md",
        dictionary_file_path="./examples/demo_dictionary.md",
        llm_model="gpt-4o-mini",
        focus_on_important_relationships=True,
        extract_hierarchical_relationships=True,
        extract_causal_relationships=True,
        min_relationship_importance="medium"
    )
    
    chunking_config = ChunkingConfig(
        chunk_size=256,
        overlap=64,
        split_by_sentence=True
    )
    
    # Create processors
    processors = [
        DictionaryMaker(dictionary_config),
        Normalizer(normalizer_config), 
        GraphBuilder(graph_config),
        Chunker(chunking_config)
    ]
    
    # Create pipeline
    pipeline = DocumentPipeline(processors)
    
    return pipeline


def display_processing_results(documents: List[Document], final_chunks: List[Document], 
                             pipeline: DocumentPipeline):
    """Display comprehensive processing results"""
    
    print("\n" + "="*80)
    print("📊 DOCUMENT PROCESSING RESULTS")
    print("="*80)
    
    # Input summary
    print(f"\n📄 Input Documents: {len(documents)}")
    for doc in documents:
        print(f"   • {doc.id}: {doc.metadata.get('title', 'Untitled')}")
    
    # Output summary
    print(f"\n📑 Output Chunks: {len(final_chunks)}")
    
    # Group chunks by original document
    chunks_by_doc = {}
    for chunk in final_chunks:
        orig_id = chunk.metadata.get("original_document_id")
        if orig_id not in chunks_by_doc:
            chunks_by_doc[orig_id] = []
        chunks_by_doc[orig_id].append(chunk)
    
    for orig_id, chunks in chunks_by_doc.items():
        print(f"   • {orig_id}: {len(chunks)} chunks")
    
    # Pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\n⚡ Pipeline Performance:")
    print(f"   • Total processing time: {stats.get('total_processing_time', 0):.3f}s")
    print(f"   • Documents processed: {stats.get('total_documents_processed', 0)}")
    print(f"   • Processors executed: {len(stats.get('processors_executed', []))}")
    
    # Processor-specific statistics
    print(f"\n🔧 Processor Statistics:")
    for i, processor in enumerate(pipeline.processors):
        processor_stats = processor.get_processing_stats()
        processor_name = processor.__class__.__name__
        print(f"   {i+1}. {processor_name}:")
        
        if processor_name == "DictionaryMaker":
            print(f"      - Terms extracted: {processor_stats.get('terms_extracted', 0)}")
            print(f"      - Variations detected: {processor_stats.get('variations_detected', 0)}")
            print(f"      - LLM API calls: {processor_stats.get('llm_api_calls', 0)}")
            
        elif processor_name == "Normalizer":
            print(f"      - Total replacements: {processor_stats.get('total_replacements', 0)}")
            print(f"      - Variations normalized: {processor_stats.get('variations_normalized', 0)}")
            print(f"      - Abbreviations expanded: {processor_stats.get('abbreviations_expanded', 0)}")
            
        elif processor_name == "GraphBuilder":
            print(f"      - Relationships extracted: {processor_stats.get('relationships_extracted', 0)}")
            print(f"      - Graph updates: {processor_stats.get('graph_updates', 0)}")
            print(f"      - LLM API calls: {processor_stats.get('llm_api_calls', 0)}")
            
        elif processor_name == "Chunker":
            print(f"      - Chunks created: {processor_stats.get('chunks_created', 0)}")
            print(f"      - Average chunk size: {processor_stats.get('average_chunk_size', 0):.1f}")
        
        print(f"      - Documents processed: {processor_stats.get('documents_processed', 0)}")


def display_generated_files():
    """Display information about generated files"""
    
    print(f"\n📁 Generated Files:")
    
    dict_path = Path("./examples/demo_dictionary.md")
    graph_path = Path("./examples/demo_knowledge_graph.md")
    
    if dict_path.exists():
        print(f"   📖 Dictionary: {dict_path}")
        print(f"      Size: {dict_path.stat().st_size:,} bytes")
        
        # Show dictionary preview
        content = dict_path.read_text(encoding='utf-8')
        lines = content.split('\n')[:10]
        print(f"      Preview:")
        for line in lines:
            if line.strip():
                print(f"        {line}")
        lines_total = len(content.split('\n'))
        if lines_total > 10:
            print(f"        ... ({lines_total - 10} more lines)")
    
    if graph_path.exists():
        print(f"\n   🕸️  Knowledge Graph: {graph_path}")
        print(f"      Size: {graph_path.stat().st_size:,} bytes")
        
        # Show graph preview
        content = graph_path.read_text(encoding='utf-8')
        lines = content.split('\n')[:10]
        print(f"      Preview:")
        for line in lines:
            if line.strip():
                print(f"        {line}")
        lines_total = len(content.split('\n'))
        if lines_total > 10:
            print(f"        ... ({lines_total - 10} more lines)")


def display_sample_chunks(final_chunks: List[Document], num_samples: int = 3):
    """Display sample processed chunks"""
    
    print(f"\n📝 Sample Processed Chunks (showing {min(num_samples, len(final_chunks))}):")
    
    for i, chunk in enumerate(final_chunks[:num_samples]):
        print(f"\n   Chunk {i+1}/{len(final_chunks)} (ID: {chunk.id}):")
        print(f"   Original Document: {chunk.metadata.get('original_document_id')}")
        print(f"   Processing Stage: {chunk.metadata.get('processing_stage')}")
        print(f"   Chunk Position: {chunk.metadata.get('chunk_position', 0)}")
        
        # Show content preview
        content_preview = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
        print(f"   Content: {content_preview}")
        
        # Show relevant metadata
        if "normalization_stats" in chunk.metadata:
            norm_stats = chunk.metadata["normalization_stats"]
            if norm_stats.get("total_replacements", 0) > 0:
                print(f"   Normalizations: {norm_stats['total_replacements']} replacements")
        
        if "graph_metadata" in chunk.metadata:
            graph_meta = chunk.metadata["graph_metadata"]
            if graph_meta.get("new_relationships_extracted", 0) > 0:
                print(f"   Graph Relations: {graph_meta['new_relationships_extracted']} extracted")


def main():
    """Main demonstration function"""
    
    print("🚀 DocumentProcessor Architecture Demo")
    print("="*50)
    print("This demo showcases the unified DocumentProcessor architecture")
    print("with LLM-powered term extraction and knowledge graph building.")
    
    # Create sample documents
    print("\n📚 Creating sample documents...")
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Set up processing pipeline
    print("\n🔧 Setting up processing pipeline...")
    pipeline = setup_processing_pipeline()
    print("Pipeline configured with:")
    for i, processor in enumerate(pipeline.processors):
        print(f"  {i+1}. {processor.__class__.__name__}")
    
    # Process documents
    print("\n⚡ Processing documents through pipeline...")
    start_time = time.time()
    
    all_chunks = []
    for doc in documents:
        print(f"  Processing: {doc.id}")
        chunks = pipeline.process_document(doc)
        all_chunks.extend(chunks)
    
    processing_time = time.time() - start_time
    print(f"✅ Processing completed in {processing_time:.3f} seconds")
    
    # Display results
    display_processing_results(documents, all_chunks, pipeline)
    display_generated_files()
    display_sample_chunks(all_chunks)
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   • {len(documents)} documents → {len(all_chunks)} chunks")
    print(f"   • Total processing time: {processing_time:.3f}s")
    print(f"   • Average time per document: {processing_time/len(documents):.3f}s")
    
    return all_chunks


if __name__ == "__main__":
    try:
        result_chunks = main()
        print("\n✅ DocumentProcessor demo completed successfully!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)