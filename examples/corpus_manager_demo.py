#!/usr/bin/env python3
"""
CorpusManager Demo - Multi-stage corpus building demonstration

This example demonstrates the new CorpusManager with support for:
- Preset configurations (simple_rag, semantic_rag, knowledge_rag)
- Stage selection approach
- Custom pipeline definitions
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.use_cases.corpus_manager_new import CorpusManager
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.models.document import Document
from refinire_rag.processing.dictionary_maker import DictionaryMakerConfig
from refinire_rag.processing.normalizer import NormalizerConfig
from refinire_rag.processing.graph_builder import GraphBuilderConfig
from refinire_rag.processing.chunker import ChunkingConfig
from refinire_rag.loaders.base import LoaderConfig
from refinire_rag.loaders.specialized import TextLoader


def create_sample_files(temp_dir: Path) -> list:
    """Create sample files for corpus building demo"""
    
    files = []
    
    # Sample file 1: RAG overview
    file1 = temp_dir / "rag_overview.txt"
    file1.write_text("""
RAG（Retrieval-Augmented Generation）システムは、検索拡張生成技術の革新的な実装です。
このシステムでは、大規模言語モデル（LLM）と外部知識ベースを統合し、
より正確で根拠のある回答生成を実現します。

主要コンポーネント：
- 文書埋め込み（Document Embedding）
- ベクトルデータベース（Vector Database）
- 検索エンジン（Retrieval Engine）
- 生成モデル（Generation Model）

RAGの利点として、ハルシネーション（幻覚）の減少、知識の更新容易性、
専門ドメインへの適応性が挙げられます。
""", encoding='utf-8')
    files.append(str(file1))
    
    # Sample file 2: Vector search
    file2 = temp_dir / "vector_search.txt"
    file2.write_text("""
ベクトル検索（Vector Search）は、セマンティック検索の中核技術です。
従来のキーワードベース検索とは異なり、意味的類似性に基づいて検索を行います。

技術的詳細：
- 高次元ベクトル空間への埋め込み
- コサイン類似度による類似性計算
- 近似最近傍探索（ANN）アルゴリズム
- インデックス構造（LSH、IVF、HNSW）

実装例：
- Faiss（Facebook AI Similarity Search）
- Chroma（オープンソースベクトルDB）
- Pinecone（マネージドサービス）
- Weaviate（GraphQLベースVectorDB）
""", encoding='utf-8')
    files.append(str(file2))
    
    # Sample file 3: Evaluation metrics
    file3 = temp_dir / "evaluation.txt"
    file3.write_text("""
RAGシステムの評価は多面的なアプローチが必要です。

検索性能指標：
- 精度（Precision）: 検索結果の正確性
- 再現率（Recall）: 関連文書の網羅性
- F1スコア: 精度と再現率の調和平均
- NDCG: ランキング品質の評価

生成品質指標：
- BLEU Score: n-gramベースの類似度
- ROUGE Score: 要約品質評価
- BERTScore: 意味的類似度評価
- 人手評価: 流暢さ、正確性、有用性

RAG固有の課題：
- 検索-生成間の一貫性
- 引用の正確性
- 知識の時効性
- バイアスの検出と軽減
""", encoding='utf-8')
    files.append(str(file3))
    
    return files


def demo_preset_configurations(temp_dir: Path, file_paths: list):
    """Demonstrate preset configurations"""
    
    print("\n" + "="*60)
    print("🎯 PRESET CONFIGURATIONS DEMO")
    print("="*60)
    
    # Initialize stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Demo 1: Simple RAG
    print("\n📌 Demo 1: Simple RAG (Load → Chunk → Vector)")
    print("-" * 40)
    
    simple_manager = CorpusManager.create_simple_rag(doc_store, vector_store)
    simple_stats = simple_manager.build_corpus(file_paths)
    
    print(f"✅ Simple RAG completed:")
    print(f"   - Files processed: {simple_stats.total_files_processed}")
    print(f"   - Documents created: {simple_stats.total_documents_created}")
    print(f"   - Chunks created: {simple_stats.total_chunks_created}")
    print(f"   - Processing time: {simple_stats.total_processing_time:.3f}s")
    print(f"   - Stages executed: {simple_stats.pipeline_stages_executed}")
    
    # Demo 2: Semantic RAG
    print("\n📌 Demo 2: Semantic RAG (Load → Dictionary → Normalize → Chunk → Vector)")
    print("-" * 40)
    
    doc_store2 = SQLiteDocumentStore(":memory:")
    vector_store2 = InMemoryVectorStore()
    
    semantic_manager = CorpusManager.create_semantic_rag(doc_store2, vector_store2)
    semantic_stats = semantic_manager.build_corpus(file_paths)
    
    print(f"✅ Semantic RAG completed:")
    print(f"   - Files processed: {semantic_stats.total_files_processed}")
    print(f"   - Documents created: {semantic_stats.total_documents_created}")
    print(f"   - Chunks created: {semantic_stats.total_chunks_created}")
    print(f"   - Processing time: {semantic_stats.total_processing_time:.3f}s")
    print(f"   - Stages executed: {semantic_stats.pipeline_stages_executed}")
    
    # Demo 3: Knowledge RAG
    print("\n📌 Demo 3: Knowledge RAG (Load → Dictionary → Graph → Normalize → Chunk → Vector)")
    print("-" * 40)
    
    doc_store3 = SQLiteDocumentStore(":memory:")
    vector_store3 = InMemoryVectorStore()
    
    knowledge_manager = CorpusManager.create_knowledge_rag(doc_store3, vector_store3)
    knowledge_stats = knowledge_manager.build_corpus(file_paths)
    
    print(f"✅ Knowledge RAG completed:")
    print(f"   - Files processed: {knowledge_stats.total_files_processed}")
    print(f"   - Documents created: {knowledge_stats.total_documents_created}")
    print(f"   - Chunks created: {knowledge_stats.total_chunks_created}")
    print(f"   - Processing time: {knowledge_stats.total_processing_time:.3f}s")
    print(f"   - Stages executed: {knowledge_stats.pipeline_stages_executed}")


def demo_stage_selection(temp_dir: Path, file_paths: list):
    """Demonstrate stage selection approach"""
    
    print("\n" + "="*60)
    print("🎛️  STAGE SELECTION DEMO")
    print("="*60)
    
    # Initialize stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    # Demo: Custom stage selection
    print("\n📌 Custom Stage Selection: Load → Dictionary → Chunk → Vector")
    print("-" * 40)
    
    corpus_manager = CorpusManager(doc_store, vector_store)
    
    # Configure each stage
    stage_configs = {
        "loader_config": LoaderConfig(),
        "dictionary_config": DictionaryMakerConfig(
            dictionary_file_path=str(temp_dir / "custom_dictionary.md"),
            focus_on_technical_terms=True,
            extract_abbreviations=True
        ),
        "chunker_config": ChunkingConfig(
            chunk_size=256,
            overlap=32,
            split_by_sentence=True
        )
    }
    
    # Execute selected stages
    selected_stages = ["load", "dictionary", "chunk", "vector"]
    stage_stats = corpus_manager.build_corpus(
        file_paths=file_paths,
        stages=selected_stages,
        stage_configs=stage_configs
    )
    
    print(f"✅ Stage selection completed:")
    print(f"   - Selected stages: {selected_stages}")
    print(f"   - Files processed: {stage_stats.total_files_processed}")
    print(f"   - Documents created: {stage_stats.total_documents_created}")
    print(f"   - Chunks created: {stage_stats.total_chunks_created}")
    print(f"   - Processing time: {stage_stats.total_processing_time:.3f}s")
    print(f"   - Documents by stage: {stage_stats.documents_by_stage}")
    
    # Check generated dictionary
    dict_file = temp_dir / "custom_dictionary.md"
    if dict_file.exists():
        print(f"\n📖 Generated dictionary:")
        content = dict_file.read_text(encoding='utf-8')
        lines = content.split('\n')[:8]
        for line in lines:
            if line.strip():
                print(f"   {line}")
        lines_total = len(content.split('\n'))
        if lines_total > 8:
            print(f"   ... ({lines_total - 8} more lines)")


def demo_custom_pipelines(temp_dir: Path, file_paths: list):
    """Demonstrate custom pipeline approach"""
    
    print("\n" + "="*60)
    print("🔧 CUSTOM PIPELINES DEMO")
    print("="*60)
    
    # Initialize stores
    doc_store = SQLiteDocumentStore(":memory:")
    vector_store = InMemoryVectorStore()
    
    print("\n📌 Custom Multi-Stage Pipeline")
    print("-" * 40)
    
    # Import required processors
    from refinire_rag.processing.document_pipeline import DocumentPipeline
    from refinire_rag.processing.document_store_processor import DocumentStoreProcessor
    from refinire_rag.processing.document_store_loader import DocumentStoreLoader, DocumentStoreLoaderConfig
    from refinire_rag.loaders.base import Loader
    from refinire_rag.processing.dictionary_maker import DictionaryMaker
    from refinire_rag.processing.normalizer import Normalizer
    from refinire_rag.processing.chunker import Chunker
    
    # Define custom pipelines
    custom_pipelines = [
        # Stage 1: Load and store original documents
        DocumentPipeline([
            TextLoader(LoaderConfig()),
            DocumentStoreProcessor(doc_store)
        ]),
        
        # Stage 2: Extract dictionary from original documents
        DocumentPipeline([
            DocumentStoreLoader(doc_store, config=DocumentStoreLoaderConfig(processing_stage="original")),
            DictionaryMaker(DictionaryMakerConfig(
                dictionary_file_path=str(temp_dir / "pipeline_dictionary.md")
            ))
        ]),
        
        # Stage 3: Normalize and store normalized documents
        DocumentPipeline([
            DocumentStoreLoader(doc_store, config=DocumentStoreLoaderConfig(processing_stage="original")),
            Normalizer(NormalizerConfig(
                dictionary_file_path=str(temp_dir / "pipeline_dictionary.md")
            )),
            DocumentStoreProcessor(doc_store)
        ]),
        
        # Stage 4: Chunk normalized documents
        DocumentPipeline([
            DocumentStoreLoader(doc_store, config=DocumentStoreLoaderConfig(processing_stage="normalized")),
            Chunker(ChunkingConfig(
                chunk_size=128,
                overlap=16
            ))
        ])
    ]
    
    # Execute custom pipelines
    corpus_manager = CorpusManager(doc_store, vector_store)
    pipeline_stats = corpus_manager.build_corpus(
        file_paths=file_paths,
        custom_pipelines=custom_pipelines
    )
    
    print(f"✅ Custom pipelines completed:")
    print(f"   - Pipeline stages: {len(custom_pipelines)}")
    print(f"   - Files processed: {pipeline_stats.total_files_processed}")
    print(f"   - Documents created: {pipeline_stats.total_documents_created}")
    print(f"   - Chunks created: {pipeline_stats.total_chunks_created}")
    print(f"   - Processing time: {pipeline_stats.total_processing_time:.3f}s")
    print(f"   - Errors encountered: {pipeline_stats.errors_encountered}")


def main():
    """Main demo function"""
    
    print("🚀 CorpusManager Multi-Stage Pipeline Demo")
    print("="*60)
    print("Demonstrating flexible corpus building with:")
    print("• Preset configurations (simple_rag, semantic_rag, knowledge_rag)")
    print("• Stage selection approach")
    print("• Custom pipeline definitions")
    
    # Create temporary directory for files
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create sample files
        print(f"\n📁 Creating sample files in: {temp_dir}")
        file_paths = create_sample_files(temp_dir)
        print(f"Created {len(file_paths)} sample files")
        
        # Demo 1: Preset configurations
        demo_preset_configurations(temp_dir, file_paths)
        
        # Demo 2: Stage selection
        demo_stage_selection(temp_dir, file_paths)
        
        # Demo 3: Custom pipelines
        demo_custom_pipelines(temp_dir, file_paths)
        
        print("\n🎉 All demos completed successfully!")
        print(f"📁 Generated files can be found in: {temp_dir}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up (optional - comment out to inspect generated files)
        # shutil.rmtree(temp_dir)
        pass
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)