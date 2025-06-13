"""
DocumentPipeline - Sequential processing pipeline for DocumentProcessor chain

This module provides a pipeline system for chaining multiple DocumentProcessor
instances together for sequential document processing.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .document_processor import DocumentProcessor
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics for pipeline execution"""
    total_documents_processed: int = 0
    total_processing_time: float = 0.0
    processors_executed: List[str] = None
    individual_processor_times: Dict[str, float] = None
    errors_encountered: int = 0
    
    def __post_init__(self):
        if self.processors_executed is None:
            self.processors_executed = []
        if self.individual_processor_times is None:
            self.individual_processor_times = {}


class DocumentPipeline:
    """Pipeline for sequential document processing through multiple DocumentProcessor instances
    
    This class chains multiple DocumentProcessor instances together to create
    a complete document processing workflow. Each processor in the pipeline
    receives the output from the previous processor as input.
    """
    
    def __init__(self, processors: List[DocumentProcessor]):
        """Initialize the document processing pipeline
        
        Args:
            processors: List of DocumentProcessor instances to chain together
        """
        self.processors = processors
        self.stats = PipelineStats()
        
        logger.info(f"Initialized DocumentPipeline with {len(processors)} processors")
        for i, processor in enumerate(processors):
            logger.debug(f"  {i+1}. {processor.__class__.__name__}")
    
    def process_document(self, document: Document) -> List[Document]:
        """Process a single document through the entire pipeline
        
        Args:
            document: Input document to process
            
        Returns:
            List of processed documents (typically chunks from the final processor)
        """
        start_time = time.time()
        
        try:
            # Start with a single document
            current_documents = [document]
            
            # Process through each processor in sequence
            for i, processor in enumerate(self.processors):
                processor_start_time = time.time()
                processor_name = processor.__class__.__name__
                
                logger.debug(f"Processing through {processor_name} (step {i+1}/{len(self.processors)})")
                
                # Process all current documents through this processor
                next_documents = []
                for doc in current_documents:
                    try:
                        processed_docs = processor.process(doc)
                        next_documents.extend(processed_docs)
                    except Exception as e:
                        logger.error(f"Error in {processor_name} for document {doc.id}: {e}")
                        self.stats.errors_encountered += 1
                        # Continue with original document on error
                        next_documents.append(doc)
                
                # Update current documents for next processor
                current_documents = next_documents
                
                # Record processor execution time
                processor_time = time.time() - processor_start_time
                self.stats.individual_processor_times[processor_name] = \
                    self.stats.individual_processor_times.get(processor_name, 0) + processor_time
                
                if processor_name not in self.stats.processors_executed:
                    self.stats.processors_executed.append(processor_name)
                
                logger.debug(f"Completed {processor_name} in {processor_time:.3f}s, "
                           f"produced {len(current_documents)} documents")
            
            # Update overall stats
            total_time = time.time() - start_time
            self.stats.total_documents_processed += 1
            self.stats.total_processing_time += total_time
            
            logger.info(f"Pipeline processed document {document.id} in {total_time:.3f}s, "
                       f"final output: {len(current_documents)} documents")
            
            return current_documents
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for document {document.id}: {e}")
            self.stats.errors_encountered += 1
            # Return original document on complete failure
            return [document]
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process multiple documents through the pipeline
        
        Args:
            documents: List of input documents to process
            
        Returns:
            List of all processed documents from all input documents
        """
        all_results = []
        
        logger.info(f"Processing {len(documents)} documents through pipeline")
        start_time = time.time()
        
        for i, document in enumerate(documents):
            logger.debug(f"Processing document {i+1}/{len(documents)}: {document.id}")
            
            try:
                result_docs = self.process_document(document)
                all_results.extend(result_docs)
            except Exception as e:
                logger.error(f"Failed to process document {document.id}: {e}")
                self.stats.errors_encountered += 1
                # Include original document on error
                all_results.append(document)
        
        total_time = time.time() - start_time
        logger.info(f"Completed processing {len(documents)} documents in {total_time:.3f}s, "
                   f"produced {len(all_results)} final documents")
        
        return all_results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics
        
        Returns:
            Dictionary containing pipeline execution statistics
        """
        return {
            "total_documents_processed": self.stats.total_documents_processed,
            "total_processing_time": self.stats.total_processing_time,
            "processors_executed": self.stats.processors_executed.copy(),
            "individual_processor_times": self.stats.individual_processor_times.copy(),
            "errors_encountered": self.stats.errors_encountered,
            "average_time_per_document": (
                self.stats.total_processing_time / max(1, self.stats.total_documents_processed)
            ),
            "pipeline_length": len(self.processors),
            "processor_names": [p.__class__.__name__ for p in self.processors]
        }
    
    def get_processor_stats(self, processor_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific processor or all processors
        
        Args:
            processor_name: Name of specific processor, or None for all processors
            
        Returns:
            Dictionary containing processor-specific statistics
        """
        if processor_name:
            # Find the specific processor
            for processor in self.processors:
                if processor.__class__.__name__ == processor_name:
                    stats = processor.get_processing_stats()
                    stats["pipeline_execution_time"] = self.stats.individual_processor_times.get(
                        processor_name, 0
                    )
                    return stats
            return {}
        else:
            # Return stats for all processors
            all_stats = {}
            for processor in self.processors:
                name = processor.__class__.__name__
                stats = processor.get_processing_stats()
                stats["pipeline_execution_time"] = self.stats.individual_processor_times.get(name, 0)
                all_stats[name] = stats
            return all_stats
    
    def reset_stats(self):
        """Reset all pipeline statistics"""
        self.stats = PipelineStats()
        
        # Reset individual processor stats
        for processor in self.processors:
            processor.processing_stats = {
                "documents_processed": 0,
                "processing_time": 0.0,
                "errors_encountered": 0
            }
        
        logger.info("Pipeline statistics reset")
    
    def validate_pipeline(self) -> bool:
        """Validate that the pipeline is properly configured
        
        Returns:
            True if pipeline is valid, False otherwise
        """
        if not self.processors:
            logger.error("Pipeline validation failed: No processors configured")
            return False
        
        for i, processor in enumerate(self.processors):
            if not isinstance(processor, DocumentProcessor):
                logger.error(f"Pipeline validation failed: Processor {i} is not a DocumentProcessor")
                return False
        
        logger.info(f"Pipeline validation passed: {len(self.processors)} processors configured")
        return True
    
    def get_pipeline_description(self) -> str:
        """Get a human-readable description of the pipeline
        
        Returns:
            String describing the pipeline configuration
        """
        processor_names = [p.__class__.__name__ for p in self.processors]
        return f"DocumentPipeline({' → '.join(processor_names)})"
    
    def __str__(self) -> str:
        """String representation of the pipeline"""
        return self.get_pipeline_description()
    
    def __repr__(self) -> str:
        """Detailed representation of the pipeline"""
        return f"DocumentPipeline(processors={len(self.processors)}, " \
               f"processed={self.stats.total_documents_processed})"