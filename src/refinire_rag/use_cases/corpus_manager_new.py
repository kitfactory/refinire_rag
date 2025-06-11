"""
CorpusManager - Document corpus construction and management

A Refinire Step that provides flexible corpus building with multiple pipeline execution.
Supports preset configurations, stage selection, and custom pipeline definitions.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

from ..processing.document_pipeline import DocumentPipeline
from ..processing.document_processor import DocumentProcessor
from ..processing.document_store_processor import DocumentStoreProcessor, DocumentStoreProcessorConfig
from ..processing.document_store_loader import DocumentStoreLoader, DocumentStoreLoaderConfig
from ..processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
from ..processing.normalizer import Normalizer, NormalizerConfig
from ..processing.graph_builder import GraphBuilder, GraphBuilderConfig
from ..processing.chunker import Chunker, ChunkingConfig
from ..loaders.base import Loader, LoaderConfig
from ..loaders.specialized import TextLoader
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    """Statistics for corpus building operations"""
    total_files_processed: int = 0
    total_documents_created: int = 0
    total_chunks_created: int = 0
    total_processing_time: float = 0.0
    pipeline_stages_executed: int = 0
    documents_by_stage: Dict[str, int] = None
    errors_encountered: int = 0
    
    def __post_init__(self):
        if self.documents_by_stage is None:
            self.documents_by_stage = {}


class CorpusManager:
    """Document corpus construction and management system
    
    This class provides flexible corpus building with support for:
    - Preset configurations (simple_rag, semantic_rag, knowledge_rag)
    - Stage selection (load, dictionary, graph, normalize, chunk, vector)
    - Custom pipeline definitions (complete control over processing)
    
    The system uses multiple-stage pipeline execution where each stage
    can involve DocumentStore operations for persistence and retrieval.
    """
    
    def __init__(self, document_store, vector_store, config: Optional[Dict[str, Any]] = None):
        """Initialize CorpusManager
        
        Args:
            document_store: DocumentStore for document persistence
            vector_store: VectorStore for vector persistence
            config: Optional global configuration
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.config = config or {}
        self.stats = CorpusStats()
        
        logger.info(f"Initialized CorpusManager with DocumentStore: {type(document_store).__name__}, "
                   f"VectorStore: {type(vector_store).__name__}")
    
    def build_corpus(self, 
                    file_paths: List[str],
                    stages: Optional[List[str]] = None,
                    custom_pipelines: Optional[List[DocumentPipeline]] = None,
                    stage_configs: Optional[Dict[str, Any]] = None) -> CorpusStats:
        """Build corpus using specified approach
        
        Args:
            file_paths: List of file paths to process
            stages: List of stages to execute (stage selection approach)
            custom_pipelines: List of custom pipelines (custom approach)
            stage_configs: Configuration for each stage
            
        Returns:
            CorpusStats with processing results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting corpus building for {len(file_paths)} files")
            
            if custom_pipelines:
                # Custom pipeline approach
                logger.info("Using custom pipeline approach")
                result_stats = self._execute_custom_pipelines(file_paths, custom_pipelines)
            elif stages:
                # Stage selection approach
                logger.info(f"Using stage selection approach: {stages}")
                result_stats = self._execute_stage_selection(file_paths, stages, stage_configs or {})
            else:
                # Default approach (simple_rag)
                logger.info("Using default simple_rag approach")
                result_stats = self._execute_simple_rag(file_paths, stage_configs or {})
            
            # Update timing
            total_time = time.time() - start_time
            result_stats.total_processing_time = total_time
            
            logger.info(f"Corpus building completed in {total_time:.3f}s: "
                       f"{result_stats.total_documents_created} documents, "
                       f"{result_stats.total_chunks_created} chunks")
            
            self.stats = result_stats
            return result_stats
            
        except Exception as e:
            logger.error(f"Corpus building failed: {e}")
            self.stats.errors_encountered += 1
            raise
    
    def _execute_custom_pipelines(self, file_paths: List[str], pipelines: List[DocumentPipeline]) -> CorpusStats:
        """Execute custom pipelines sequentially"""
        stats = CorpusStats()
        
        # Create trigger document for pipeline execution
        trigger_doc = Document(
            id="corpus_build_trigger",
            content="",
            metadata={
                "paths": file_paths,
                "trigger_type": "corpus_build"
            }
        )
        
        # Execute each pipeline in sequence
        for i, pipeline in enumerate(pipelines):
            logger.info(f"Executing custom pipeline {i+1}/{len(pipelines)}")
            
            try:
                # Execute pipeline
                results = pipeline.process_document(trigger_doc)
                
                # Update statistics
                pipeline_stats = pipeline.get_pipeline_stats()
                stats.pipeline_stages_executed += 1
                
                # Collect statistics from pipeline
                self._update_stats_from_pipeline(stats, pipeline_stats, results)
                
                logger.info(f"Pipeline {i+1} completed: {len(results)} documents")
                
            except Exception as e:
                logger.error(f"Pipeline {i+1} failed: {e}")
                stats.errors_encountered += 1
        
        return stats
    
    def _execute_stage_selection(self, file_paths: List[str], stages: List[str], 
                                stage_configs: Dict[str, Any]) -> CorpusStats:
        """Execute stages based on selection"""
        stats = CorpusStats()
        
        # Build pipelines from stages
        pipelines = self._build_pipelines_from_stages(stages, stage_configs)
        
        # Execute pipelines sequentially
        trigger_doc = Document(
            id="stage_build_trigger",
            content="",
            metadata={
                "paths": file_paths,
                "trigger_type": "stage_build",
                "stages": stages
            }
        )
        
        for i, (stage_name, pipeline) in enumerate(pipelines):
            logger.info(f"Executing stage: {stage_name} ({i+1}/{len(pipelines)})")
            
            try:
                # Execute pipeline
                results = pipeline.process_document(trigger_doc)
                
                # Update statistics
                pipeline_stats = pipeline.get_pipeline_stats()
                stats.pipeline_stages_executed += 1
                
                # Collect stage-specific statistics
                self._update_stats_from_stage(stats, stage_name, pipeline_stats, results)
                
                logger.info(f"Stage '{stage_name}' completed: {len(results)} documents")
                
            except Exception as e:
                logger.error(f"Stage '{stage_name}' failed: {e}")
                stats.errors_encountered += 1
        
        return stats
    
    def _execute_simple_rag(self, file_paths: List[str], stage_configs: Dict[str, Any]) -> CorpusStats:
        """Execute simple RAG approach (load → chunk → vector)"""
        logger.info("Executing simple RAG approach")
        
        # Use stage selection with simple_rag stages
        simple_stages = ["load", "chunk", "vector"]
        return self._execute_stage_selection(file_paths, simple_stages, stage_configs)
    
    def _build_pipelines_from_stages(self, stages: List[str], 
                                   stage_configs: Dict[str, Any]) -> List[tuple]:
        """Build pipelines from stage selection
        
        Returns:
            List of (stage_name, pipeline) tuples
        """
        pipelines = []
        
        # Stage 1: Loading (if requested)
        if "load" in stages:
            loader_config = stage_configs.get("loader_config", LoaderConfig())
            load_pipeline = DocumentPipeline([
                TextLoader(loader_config),
                DocumentStoreProcessor(self.document_store, 
                                     stage_configs.get("store_config", DocumentStoreProcessorConfig()))
            ])
            pipelines.append(("load", load_pipeline))
        
        # Stage 2: Knowledge extraction (dictionary/graph)
        knowledge_processors = []
        if "dictionary" in stages:
            dict_config = stage_configs.get("dictionary_config", DictionaryMakerConfig())
            knowledge_processors.append(DictionaryMaker(dict_config))
        
        if "graph" in stages:
            graph_config = stage_configs.get("graph_config", GraphBuilderConfig())
            knowledge_processors.append(GraphBuilder(graph_config))
        
        if knowledge_processors:
            loader_config = DocumentStoreLoaderConfig(processing_stage="original")
            knowledge_pipeline = DocumentPipeline([
                DocumentStoreLoader(self.document_store, config=loader_config)
            ] + knowledge_processors)
            pipelines.append(("knowledge_extraction", knowledge_pipeline))
        
        # Stage 3: Normalization (if requested)
        if "normalize" in stages:
            loader_config = DocumentStoreLoaderConfig(processing_stage="original")
            norm_config = stage_configs.get("normalizer_config", NormalizerConfig())
            store_config = stage_configs.get("store_config", DocumentStoreProcessorConfig())
            
            normalize_pipeline = DocumentPipeline([
                DocumentStoreLoader(self.document_store, config=loader_config),
                Normalizer(norm_config),
                DocumentStoreProcessor(self.document_store, store_config)
            ])
            pipelines.append(("normalize", normalize_pipeline))
        
        # Stage 4: Chunking and vectorization
        final_processors = []
        
        # Determine source stage for final processing
        source_stage = "normalized" if "normalize" in stages else "original"
        loader_config = DocumentStoreLoaderConfig(processing_stage=source_stage)
        final_processors.append(DocumentStoreLoader(self.document_store, config=loader_config))
        
        if "chunk" in stages:
            chunk_config = stage_configs.get("chunker_config", ChunkingConfig())
            final_processors.append(Chunker(chunk_config))
        
        if "vector" in stages:
            # Import VectorStoreProcessor when needed
            from ..processing.vector_store_processor import VectorStoreProcessor, VectorStoreProcessorConfig
            vector_config = stage_configs.get("vector_config", VectorStoreProcessorConfig())
            final_processors.append(VectorStoreProcessor(self.vector_store, config=vector_config))
        
        if len(final_processors) > 1:  # More than just the loader
            final_pipeline = DocumentPipeline(final_processors)
            pipelines.append(("chunk_vector", final_pipeline))
        
        return pipelines
    
    def _update_stats_from_pipeline(self, stats: CorpusStats, pipeline_stats: Dict[str, Any], 
                                   results: List[Document]):
        """Update corpus stats from pipeline statistics"""
        stats.total_documents_created += len(results)
        
        # Count chunks (documents with chunk metadata)
        chunks = [doc for doc in results if doc.metadata.get("processing_stage") == "chunked"]
        stats.total_chunks_created += len(chunks)
        
        # Update stage counts
        for doc in results:
            stage = doc.metadata.get("processing_stage", "unknown")
            stats.documents_by_stage[stage] = stats.documents_by_stage.get(stage, 0) + 1
    
    def _update_stats_from_stage(self, stats: CorpusStats, stage_name: str, 
                                pipeline_stats: Dict[str, Any], results: List[Document]):
        """Update corpus stats from stage execution"""
        self._update_stats_from_pipeline(stats, pipeline_stats, results)
        
        # Add stage-specific tracking
        if stage_name == "load":
            stats.total_files_processed = pipeline_stats.get("total_documents_processed", 0)
    
    @classmethod
    def create_simple_rag(cls, document_store, vector_store, config: Optional[Dict[str, Any]] = None):
        """Create CorpusManager configured for simple RAG
        
        Pipeline: Load → Chunk → Vector
        """
        manager = cls(document_store, vector_store, config)
        
        def build_simple_corpus(file_paths: List[str], **kwargs):
            return cls.build_corpus(
                manager,
                file_paths=file_paths,
                stages=["load", "chunk", "vector"],
                stage_configs=kwargs
            )
        
        # Replace build_corpus method
        manager.build_corpus = build_simple_corpus
        return manager
    
    @classmethod
    def create_semantic_rag(cls, document_store, vector_store, config: Optional[Dict[str, Any]] = None):
        """Create CorpusManager configured for semantic RAG
        
        Pipeline: Load → Dictionary → Normalize → Chunk → Vector
        """
        manager = cls(document_store, vector_store, config)
        
        def build_semantic_corpus(file_paths: List[str], **kwargs):
            return cls.build_corpus(
                manager,
                file_paths=file_paths,
                stages=["load", "dictionary", "normalize", "chunk", "vector"],
                stage_configs=kwargs
            )
        
        # Replace build_corpus method
        manager.build_corpus = build_semantic_corpus
        return manager
    
    @classmethod
    def create_knowledge_rag(cls, document_store, vector_store, config: Optional[Dict[str, Any]] = None):
        """Create CorpusManager configured for knowledge RAG
        
        Pipeline: Load → Dictionary → Graph → Normalize → Chunk → Vector
        """
        manager = cls(document_store, vector_store, config)
        
        def build_knowledge_corpus(file_paths: List[str], **kwargs):
            return cls.build_corpus(
                manager,
                file_paths=file_paths,
                stages=["load", "dictionary", "graph", "normalize", "chunk", "vector"],
                stage_configs=kwargs
            )
        
        # Replace build_corpus method
        manager.build_corpus = build_knowledge_corpus
        return manager
    
    def get_corpus_stats(self) -> CorpusStats:
        """Get current corpus statistics"""
        return self.stats
    
    def get_documents_by_stage(self, processing_stage: str) -> List[Document]:
        """Get documents by processing stage
        
        Args:
            processing_stage: Stage to filter by
            
        Returns:
            List of documents in the specified stage
        """
        loader = DocumentStoreLoader(self.document_store, 
                                   config=DocumentStoreLoaderConfig(processing_stage=processing_stage))
        
        # Create trigger document
        trigger = Document(id="stage_query", content="", metadata={})
        return loader.process(trigger)
    
    def rebuild_stage(self, stage_name: str, stage_config: Optional[Dict[str, Any]] = None):
        """Rebuild a specific processing stage
        
        Args:
            stage_name: Name of stage to rebuild
            stage_config: Configuration for the stage
        """
        logger.info(f"Rebuilding stage: {stage_name}")
        
        # This would implement stage-specific rebuilding logic
        # For now, just log the request
        logger.warning("Stage rebuilding not yet implemented")
    
    def clear_corpus(self):
        """Clear all documents from the corpus"""
        logger.warning("Corpus clearing not yet implemented")
        # This would implement corpus clearing logic