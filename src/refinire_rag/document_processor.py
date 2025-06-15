"""
Base classes for document processing
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Union, Type, TypeVar, TYPE_CHECKING, Iterable, Iterator
from datetime import datetime
from dataclasses import dataclass

from refinire_rag.models.document import Document

if TYPE_CHECKING:
    from refinire_rag.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)

# Type variable for config classes
TConfig = TypeVar('TConfig')


@dataclass
class DocumentProcessorConfig:
    """Base configuration for document processors
    文書プロセッサーの基本設定
    
    This is the base configuration class that all document processor configs should inherit from.
    It provides common configuration options that apply to all processors.
    """
    
    # Processing options
    name: Optional[str] = None  # Processor name for identification
    enabled: bool = True  # Whether the processor is enabled
    log_level: str = "INFO"  # Logging level for this processor
    
    # Performance options
    batch_processing: bool = False  # Enable batch processing mode
    max_workers: int = 1  # Maximum number of workers for parallel processing
    timeout: Optional[float] = None  # Processing timeout in seconds
    
    # Error handling
    skip_on_error: bool = False  # Skip documents that cause errors
    retry_count: int = 0  # Number of retries on error
    
    # Metadata options
    preserve_metadata: bool = True  # Preserve original document metadata
    add_processor_metadata: bool = True  # Add processor-specific metadata


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
    
    def process_with_stats(self, document: Document, config: Optional[Any] = None) -> List[Document]:
        """Process a single document with statistics tracking
        統計追跡付きで単一文書を処理
        
        Args:
            document: Document to process
            config: Optional configuration for processing
            
        Returns:
            List of processed documents
        """
        start_time = time.time()
        
        try:
            results = list(self.process([document], config))
            
            # Update processing stats
            processing_time = time.time() - start_time
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics
        処理統計を取得
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.processing_stats.copy()
        
        # Calculate averages
        if stats["documents_processed"] > 0:
            stats["average_processing_time"] = stats["total_processing_time"] / stats["documents_processed"]
        else:
            stats["average_processing_time"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset processing statistics
        処理統計をリセット"""
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0,
            "last_processed": None
        }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information
        プロセッサー情報を取得
        
        Returns:
            Dictionary with processor information
        """
        return {
            "processor_id": id(self),
            "processor_class": self.__class__.__name__,
            "config": self.config,
            "stats": self.get_processing_stats()
        }
    

class DocumentPipeline:
    """Pipeline for chaining multiple document processors
    複数の文書プロセッサーをチェーンするパイプライン"""
    
    def __init__(
        self, 
        processors: List[DocumentProcessor], 
        document_store: Optional['DocumentStore'] = None,
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