# Tutorial 6: Incremental Document Loading

This tutorial demonstrates how to efficiently manage large document collections using incremental loading capabilities, building on the enterprise RAG system from Tutorial 5.

## Overview

Incremental loading allows you to:
- Process only new and updated documents
- Skip unchanged files for efficient updates
- Handle large document repositories efficiently
- Maintain document lineage and history

## Prerequisites

Complete Tutorial 5 (Enterprise RAG Usage) as we'll extend that enterprise system with incremental loading capabilities.

## Implementation

### Step 1: Enhanced Enterprise RAG with Incremental Loading

```python
#!/usr/bin/env python3
"""
Tutorial 6: Incremental Document Loading
企業環境でのRAGシステムに増分ローディング機能を追加
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.loaders.incremental_loader import IncrementalLoader
from refinire_rag.application.corpus_manager import CorpusManager, CorpusManagerConfig
from refinire_rag.application.query_engine import QueryEngine, QueryEngineConfig
from refinire_rag.storage import SQLiteDocumentStore, InMemoryVectorStore
from refinire_rag.embedding import TFIDFEmbedder, TFIDFEmbeddingConfig
from refinire_rag.retrieval import SimpleRetriever, SimpleReranker, SimpleReader
from refinire_rag.processing import Normalizer, NormalizerConfig, TokenBasedChunker, ChunkingConfig
from refinire_rag.models.document import Document


class IncrementalEnterpriseRAG:
    """
    Enterprise RAG system with incremental loading capabilities
    増分ローディング対応の企業RAGシステム
    """
    
    def __init__(self, department: str, base_dir: Path):
        self.department = department
        self.base_dir = base_dir
        self.docs_dir = base_dir / f"{department}_docs"
        self.db_path = str(base_dir / f"{department}_rag.db")
        self.cache_path = str(base_dir / f".{department}_cache.json")
        
        # Create department directory
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_components()
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "last_update": None,
            "incremental_runs": 0,
            "total_processing_time": 0.0
        }
    
    def _setup_components(self):
        """Initialize RAG components"""
        
        # Storage
        self.document_store = SQLiteDocumentStore(self.db_path)
        self.vector_store = InMemoryVectorStore(similarity_metric="cosine")
        
        # Embedding
        embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0, max_features=5000)
        self.embedder = TFIDFEmbedder(config=embedder_config)
        
        # Processing components
        normalizer_config = NormalizerConfig(
            dictionary_file_path=str(self.base_dir / f"{self.department}_dictionary.md"),
            whole_word_only=False
        )
        
        chunking_config = ChunkingConfig(
            chunk_size=300,
            overlap=30,
            split_by_sentence=True
        )
        
        # Corpus manager with processors
        corpus_config = CorpusManagerConfig(
            document_store=self.document_store,
            vector_store=self.vector_store,
            embedder=self.embedder,
            processors=[
                Normalizer(normalizer_config),
                TokenBasedChunker(chunking_config)
            ],
            enable_progress_reporting=True
        )
        
        self.corpus_manager = CorpusManager(corpus_config)
        
        # Query engine
        retriever = SimpleRetriever(self.vector_store, self.embedder)
        reranker = SimpleReranker()
        reader = SimpleReader()
        
        query_config = QueryEngineConfig(
            retriever=retriever,
            reranker=reranker,
            reader=reader,
            normalizer=Normalizer(normalizer_config)
        )
        
        self.query_engine = QueryEngine(query_config)
        
        # Incremental loader
        self.incremental_loader = IncrementalLoader(
            document_store=self.document_store,
            base_loader=self.corpus_manager._loader,
            cache_file=self.cache_path
        )
    
    def add_documents(self, documents: Dict[str, str], force_update: bool = False):
        """
        Add or update documents in the department's collection
        
        Args:
            documents: Dict of filename -> content
            force_update: Force update even if file hasn't changed
        """
        print(f"\\n📁 Adding documents to {self.department} department...")
        
        # Write documents to files
        for filename, content in documents.items():
            file_path = self.docs_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   Created: {filename}")
        
        # Process incrementally
        start_time = time.time()
        
        force_reload = set()
        if force_update:
            force_reload = {str(self.docs_dir / filename) for filename in documents.keys()}
        
        results = self.incremental_loader.process_incremental(
            sources=[self.docs_dir],
            force_reload=force_reload
        )
        
        processing_time = time.time() - start_time
        
        # Fit embedder if we have new documents
        all_docs = results['new'] + results['updated']
        if all_docs and not self.embedder.is_fitted():
            print(f"   Fitting embedder on {len(all_docs)} documents...")
            texts = [doc.content for doc in all_docs if doc.content.strip()]
            if texts:
                self.embedder.fit(texts)
        
        # Process through corpus manager pipeline
        if all_docs:
            print(f"   Processing {len(all_docs)} documents through RAG pipeline...")
            corpus_results = self.corpus_manager.process_documents(all_docs)
            
            # Generate embeddings and store
            embedded_docs = self.corpus_manager.embed_documents(corpus_results)
            print(f"   Generated embeddings for {len(embedded_docs)} documents")
        
        # Update statistics
        self.stats["total_documents"] = self.document_store.get_stats().total_documents
        self.stats["last_update"] = datetime.now().isoformat()
        self.stats["incremental_runs"] += 1
        self.stats["total_processing_time"] += processing_time
        
        # Report results
        print(f"\\n📊 Processing Results:")
        print(f"   New documents: {len(results['new'])}")
        print(f"   Updated documents: {len(results['updated'])}")
        print(f"   Skipped documents: {len(results['skipped'])}")
        print(f"   Processing time: {processing_time:.2f}s")
        
        return results
    
    def update_document(self, filename: str, new_content: str):
        """Update a specific document"""
        print(f"\\n📝 Updating document: {filename}")
        
        file_path = self.docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        # Process the update
        results = self.add_documents({filename: new_content})
        return results
    
    def delete_document(self, filename: str):
        """Delete a document and clean up"""
        print(f"\\n🗑️ Deleting document: {filename}")
        
        file_path = self.docs_dir / filename
        if file_path.exists():
            file_path.unlink()
            print(f"   Deleted file: {filename}")
            
            # Clean up from stores
            deleted_docs = self.incremental_loader.cleanup_deleted_files([self.docs_dir])
            print(f"   Cleaned up {len(deleted_docs)} documents from stores")
            
            return deleted_docs
        else:
            print(f"   File not found: {filename}")
            return []
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the department's RAG system"""
        return self.query_engine.answer(question)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        cache_stats = self.incremental_loader.get_cache_stats()
        store_stats = self.document_store.get_stats()
        
        return {
            **self.stats,
            "cache_statistics": cache_stats,
            "document_store": {
                "total_documents": store_stats.total_documents,
                "db_path": self.db_path
            },
            "vector_store": {
                "total_vectors": len(self.vector_store.vectors),
                "dimension": getattr(self.vector_store, 'dimension', 'Unknown')
            }
        }


def demo_incremental_enterprise_rag():
    """Demonstrate incremental loading in enterprise RAG"""
    
    print("🏢 Tutorial 6: Incremental Document Loading")
    print("=" * 60)
    
    # Setup demo environment
    demo_dir = Path("enterprise_incremental_demo")
    demo_dir.mkdir(exist_ok=True)
    
    try:
        # Create HR department RAG system
        hr_rag = IncrementalEnterpriseRAG("hr", demo_dir)
        
        print("\\n🎯 Step 1: Initial Document Loading")
        print("-" * 40)
        
        # Initial documents
        initial_docs = {
            "employee_handbook.md": '''
# 従業員ハンドブック v1.0

## 勤務時間
- 平日: 9:00-18:00
- 昼休み: 12:00-13:00

## 休暇制度
- 年次有給休暇: 20日
- 夏季休暇: 3日
- 年末年始: 5日

## 福利厚生
- 健康保険
- 厚生年金
- 雇用保険
            ''',
            
            "remote_work_policy.md": '''
# リモートワーク規定

## 対象者
正社員および契約社員

## 申請手続き
1. 上司に事前申請
2. 人事部承認
3. 実施報告

## 勤務環境
- 安定したインターネット接続
- 適切な作業スペース
- セキュリティ対策
            '''
        }
        
        # Add initial documents
        results1 = hr_rag.add_documents(initial_docs)
        
        print("\\n🔍 Step 2: Test Initial Query")
        print("-" * 40)
        
        result = hr_rag.query("リモートワークの申請手続きは？")
        print(f"質問: リモートワークの申請手続きは？")
        print(f"回答: {result['answer']}")
        
        print("\\n⏱️ Step 3: Incremental Update (No Changes)")
        print("-" * 40)
        
        # Run again with no changes - should skip all
        results2 = hr_rag.add_documents({})
        
        print("\\n📝 Step 4: Update Existing Document")
        print("-" * 40)
        
        # Update employee handbook
        updated_handbook = '''
# 従業員ハンドブック v2.0

## 勤務時間
- 平日: 9:00-17:30 (変更)
- 昼休み: 12:00-13:00

## 休暇制度
- 年次有給休暇: 25日 (改善)
- 夏季休暇: 5日 (改善)
- 年末年始: 5日
- リフレッシュ休暇: 3日 (新規)

## 福利厚生
- 健康保険
- 厚生年金
- 雇用保険
- 住宅手当 (新規)

最終更新: 2024年6月
        '''
        
        results3 = hr_rag.update_document("employee_handbook.md", updated_handbook)
        
        print("\\n📄 Step 5: Add New Document")
        print("-" * 40)
        
        new_docs = {
            "performance_review.md": '''
# 人事評価制度

## 評価期間
- 上半期: 4月-9月
- 下半期: 10月-3月

## 評価項目
1. 業務成果
2. プロセス
3. 行動特性

## 評価プロセス
1. 自己評価
2. 上司評価
3. フィードバック面談
4. 評価確定

## 昇進・昇格
評価結果に基づき年2回検討
            '''
        }
        
        results4 = hr_rag.add_documents(new_docs)
        
        print("\\n🔍 Step 6: Test Updated Knowledge")
        print("-" * 40)
        
        # Test updated information
        questions = [
            "勤務時間は何時までですか？",
            "年次有給休暇は何日もらえますか？",
            "人事評価はいつ行われますか？"
        ]
        
        for question in questions:
            result = hr_rag.query(question)
            print(f"\\n質問: {question}")
            print(f"回答: {result['answer']}")
        
        print("\\n🗑️ Step 7: Document Deletion")
        print("-" * 40)
        
        # Delete a document
        deleted = hr_rag.delete_document("remote_work_policy.md")
        
        print("\\n📊 Step 8: Final Statistics")
        print("-" * 40)
        
        stats = hr_rag.get_statistics()
        print(f"📈 System Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Incremental runs: {stats['incremental_runs']}")
        print(f"   Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"   Last update: {stats['last_update']}")
        print(f"   Cache files: {stats['cache_statistics']['total_files']}")
        print(f"   Vector store size: {stats['vector_store']['total_vectors']}")
        
        print("\\n🎯 Benefits of Incremental Loading")
        print("-" * 40)
        print("✅ Efficient updates - only processes changed files")
        print("✅ Fast reprocessing - skips unchanged documents")
        print("✅ Large-scale support - handles thousands of documents")
        print("✅ Automatic cleanup - removes deleted documents")
        print("✅ Change detection - multiple verification methods")
        print("✅ Enterprise ready - production deployment suitable")
        
        print("\\n💡 Production Usage Tips")
        print("-" * 40)
        print("1. Run incremental updates on schedule (nightly/hourly)")
        print("2. Monitor cache statistics for optimization")
        print("3. Use force_reload for systematic updates")
        print("4. Implement file watching for real-time updates")
        print("5. Backup cache files for disaster recovery")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup option
        cleanup = input("\\nClean up demo files? (y/N): ").lower().strip()
        if cleanup == 'y':
            import shutil
            if demo_dir.exists():
                shutil.rmtree(demo_dir)
            print("✅ Demo files cleaned up")
        else:
            print(f"📁 Demo files preserved in: {demo_dir}")
    
    print("\\n🎉 Tutorial 6 Complete!")
    print("\\nNext: Tutorial 7 will cover advanced query optimization and caching")


if __name__ == "__main__":
    demo_incremental_enterprise_rag()