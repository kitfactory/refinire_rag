"""
Base classes for document processing
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Union, Type, TypeVar, TYPE_CHECKING, Iterable, Iterator
from datetime import datetime
from dataclasses import dataclass

from refinire.rag.models.document import Document
from refinire.rag.storage.document_store import DocumentStore

if TYPE_CHECKING:
    from refinire.rag.storage.document_store import DocumentStore

print("[DEBUG] document_processor.py import start")

print("[DEBUG] document_processor.py import completed")

logger = logging.getLogger(__name__)

# Type variable for config classes
TConfig = TypeVar('TConfig')


class DocumentProcessor(ABC):
    """Base interface for document processing
    文書処理の基底インターフェース"""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize document processor
        文書プロセッサーを初期化
        
        Args:
            config: Optional configuration for the processor
        """
        self.config = config
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0,
            "last_processed": None
        }
    
    @abstractmethod
    def process(self, documents: Iterable[Document], config: Optional[Any] = None) -> Iterator[Document]:
        """Process a document and return list of resulting documents
        文書を処理して結果文書のリストを返す
        
        Args:
            documents: Input documents to process
            config: Optional configuration for processing
            
        Returns:
            Iterator of processed documents
        """
        pass
    

class DocumentPipeline:
    """Pipeline for chaining multiple document processors
    複数の文書プロセッサーをチェーンするパイプライン"""
    
    def __init__(
        self, 
        processors: List[DocumentProcessor], 
        document_store: Optional[DocumentStore] = None,
        store_intermediate_results: bool = True
    ):
        """Initialize document pipeline
        文書パイプラインを初期化
        
        Args:
            processors: List of document processors to chain
            document_store: Optional document store for persistence
            store_intermediate_results: Whether to store intermediate processing results
        """
        self.processors = processors
        self.document_store = document_store
        self.store_intermediate_results = store_intermediate_results
        self.pipeline_stats = {
            "documents_processed": 0,
            "total_pipeline_time": 0.0,
            "errors": 0,
            "last_processed": None,
            "processor_stats": {}
        }
        
        logger.info(f"Initialized DocumentPipeline with {len(processors)} processors")
    
    def process_document(self, document: Document) -> List[Document]:
        """Process document through the entire pipeline
        文書をパイプライン全体で処理
        
        Args:
            document: Input document to process
            
        Returns:
            All documents created during processing
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing document {document.id} through pipeline with {len(self.processors)} processors")
            
            current_docs = [document]
            all_results = []
            
            # Store original document if store is available
            if self.document_store and self.store_intermediate_results:
                self.document_store.store_document(document)
                all_results.append(document)
            
            # Process through each processor
            for i, processor in enumerate(self.processors):
                logger.debug(f"Running processor {i+1}/{len(self.processors)}: {processor.__class__.__name__}")
                
                next_docs = []
                processor_start_time = time.time()
                
                for doc in current_docs:
                    try:
                        processed = processor.process_with_stats(doc)
                        next_docs.extend(processed)
                        
                        # Store each processed document if store is available
                        if self.document_store:
                            for processed_doc in processed:
                                self.document_store.store_document(processed_doc)
                                all_results.append(processed_doc)
                                
                    except Exception as e:
                        logger.error(f"Error processing document {doc.id} with {processor.__class__.__name__}: {e}")
                        self.pipeline_stats["errors"] += 1
                        
                        # Continue with other documents
                        continue
                
                # Update processor stats
                processor_time = time.time() - processor_start_time
                processor_name = processor.__class__.__name__
                if processor_name not in self.pipeline_stats["processor_stats"]:
                    self.pipeline_stats["processor_stats"][processor_name] = {
                        "total_time": 0.0,
                        "documents_processed": 0,
                        "errors": 0
                    }
                
                self.pipeline_stats["processor_stats"][processor_name]["total_time"] += processor_time
                self.pipeline_stats["processor_stats"][processor_name]["documents_processed"] += len(current_docs)
                
                current_docs = next_docs
                logger.debug(f"Processor {processor.__class__.__name__} produced {len(next_docs)} documents")
            
            # Update pipeline statistics
            pipeline_time = time.time() - start_time
            self.pipeline_stats["documents_processed"] += 1
            self.pipeline_stats["total_pipeline_time"] += pipeline_time
            self.pipeline_stats["last_processed"] = datetime.now().isoformat()
            
            logger.info(f"Pipeline processing completed for document {document.id} in {pipeline_time:.3f}s, produced {len(all_results)} total documents")
            
            return all_results
            
        except Exception as e:
            self.pipeline_stats["errors"] += 1
            logger.error(f"Pipeline processing failed for document {document.id}: {e}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents through the pipeline
        複数の文書をパイプラインで処理
        
        Args:
            documents: List of documents to process
            
        Returns:
            All documents created during processing
        """
        all_results = []
        
        logger.info(f"Processing {len(documents)} documents through pipeline")
        
        for i, doc in enumerate(documents):
            logger.debug(f"Processing document {i+1}/{len(documents)}: {doc.id}")
            
            try:
                results = self.process_document(doc)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to process document {doc.id}: {e}")
                continue
        
        logger.info(f"Pipeline batch processing completed: processed {len(documents)} input documents, produced {len(all_results)} total documents")
        
        return all_results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics
        パイプライン処理統計を取得
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = self.pipeline_stats.copy()
        
        # Calculate averages
        if stats["documents_processed"] > 0:
            stats["average_pipeline_time"] = stats["total_pipeline_time"] / stats["documents_processed"]
        else:
            stats["average_pipeline_time"] = 0.0
        
        # Add individual processor stats
        for processor in self.processors:
            processor_name = processor.__class__.__name__
            stats["processor_stats"][processor_name] = {
                **stats["processor_stats"].get(processor_name, {}),
                **processor.get_processing_stats()
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics
        パイプライン統計をリセット"""
        self.pipeline_stats = {
            "documents_processed": 0,
            "total_pipeline_time": 0.0,
            "errors": 0,
            "last_processed": None,
            "processor_stats": {}
        }
        
        # Reset individual processor stats
        for processor in self.processors:
            processor.reset_stats()
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information
        パイプライン情報を取得
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "pipeline_id": id(self),
            "num_processors": len(self.processors),
            "processors": [processor.get_processor_info() for processor in self.processors],
            "store_intermediate_results": self.store_intermediate_results,
            "has_document_store": self.document_store is not None,
            "stats": self.get_pipeline_stats()
        }