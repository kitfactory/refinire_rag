"""
Base classes for document processing
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Union, Type, TypeVar
from datetime import datetime
from dataclasses import dataclass

from ..models import Document
from ..storage import DocumentStore

logger = logging.getLogger(__name__)

# Type variable for config classes
TConfig = TypeVar('TConfig')


@dataclass
class DocumentProcessorConfig:
    """Base configuration class for document processors
    文書プロセッサーの基底設定クラス"""
    
    # Common configuration fields that all processors might use
    preserve_metadata: bool = True
    add_processing_info: bool = True
    fail_on_error: bool = False
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary
        設定を辞書に変換"""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            result[field_name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "DocumentProcessorConfig":
        """Create configuration from dictionary
        辞書から設定を作成"""
        # Filter data to only include fields that exist in the dataclass
        filtered_data = {
            key: value for key, value in data.items() 
            if key in cls.__dataclass_fields__
        }
        return cls(**filtered_data)


class DocumentProcessor(ABC):
    """Base interface for document processing
    文書処理の基底インターフェース"""
    
    def __init__(self, config: Optional[DocumentProcessorConfig] = None):
        """Initialize document processor
        文書プロセッサーを初期化
        
        Args:
            config: Optional configuration for the processor
        """
        # Use default config if none provided
        if config is None:
            config = self.get_default_config()
        
        self.config = config
        self.processing_stats = {
            "documents_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0,
            "last_processed": None
        }
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[DocumentProcessorConfig]:
        """Get the configuration class for this processor
        このプロセッサーの設定クラスを取得
        
        Returns:
            Configuration class type
        """
        pass
    
    @classmethod
    def get_default_config(cls) -> DocumentProcessorConfig:
        """Get default configuration for this processor
        このプロセッサーのデフォルト設定を取得
        
        Returns:
            Default configuration instance
        """
        config_class = cls.get_config_class()
        return config_class()
    
    @abstractmethod
    def process(self, document: Document, config: Optional[Any] = None) -> List[Document]:
        """Process a document and return list of resulting documents
        文書を処理して結果文書のリストを返す
        
        Args:
            document: Input document to process
            config: Optional configuration for processing
            
        Returns:
            List of processed documents (could be 1 for normalization, many for chunking)
        """
        pass
    
    def validate_config(self, config: Any) -> bool:
        """Validate that config is of the correct type
        設定が正しい型であることを検証
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if config is valid
        """
        expected_class = self.get_config_class()
        if config is not None and not isinstance(config, expected_class):
            logger.warning(
                f"Config type mismatch: expected {expected_class.__name__}, "
                f"got {type(config).__name__}"
            )
            return False
        return True
    
    def process_with_stats(self, document: Document, config: Optional[Any] = None) -> List[Document]:
        """Process document with statistics tracking
        統計追跡付きで文書を処理
        
        Args:
            document: Input document to process
            config: Optional configuration for processing
            
        Returns:
            List of processed documents
        """
        start_time = time.time()
        
        try:
            logger.debug(f"Processing document {document.id} with {self.__class__.__name__}")
            
            # Use provided config or fall back to instance config
            processing_config = config or self.config
            
            # Validate config type
            if not self.validate_config(processing_config):
                logger.warning("Invalid config type, using default config")
                processing_config = self.get_default_config()
            
            # Process the document
            results = self.process(document, processing_config)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["last_processed"] = datetime.now().isoformat()
            
            logger.debug(f"Successfully processed document {document.id} in {processing_time:.3f}s, produced {len(results)} documents")
            
            # Add processing metadata if enabled
            if processing_config and getattr(processing_config, 'add_processing_info', True):
                for result_doc in results:
                    if 'processing_info' not in result_doc.metadata:
                        result_doc.metadata['processing_info'] = {}
                    
                    result_doc.metadata['processing_info'].update({
                        'processor': self.__class__.__name__,
                        'processed_at': datetime.now().isoformat(),
                        'processing_time_seconds': processing_time,
                        'config_used': processing_config.to_dict() if hasattr(processing_config, 'to_dict') else str(processing_config)
                    })
            
            return results
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            logger.error(f"Error processing document {document.id}: {e}")
            
            # Re-raise if fail_on_error is True
            if self.config and getattr(self.config, 'fail_on_error', False):
                raise
            
            # Return empty list on error if fail_on_error is False
            return []
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics
        処理統計を取得
        
        Returns:
            Dictionary with processing statistics
        """
        stats = self.processing_stats.copy()
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
            "processor_class": self.__class__.__name__,
            "config_class": self.get_config_class().__name__,
            "config": self.config.to_dict() if self.config and hasattr(self.config, 'to_dict') else str(self.config),
            "stats": self.get_processing_stats()
        }


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