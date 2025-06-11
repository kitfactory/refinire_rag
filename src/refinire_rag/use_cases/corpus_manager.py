"""
CorpusManager Use Case

Orchestrates the complete document processing pipeline from loading to embedding storage.
Combines document loading, processing, chunking, embedding, and storage into a unified workflow.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import time
import logging

from ..models.document import Document
from ..models.config import LoadingConfig
from ..loaders import UniversalLoader
from ..processing import DocumentProcessor, DocumentPipeline
from ..chunking import TokenBasedChunker, ChunkingConfig
from ..embedding import Embedder, TFIDFEmbedder, TFIDFEmbeddingConfig
from ..storage import DocumentStore, SQLiteDocumentStore, VectorStore, InMemoryVectorStore, VectorEntry
from ..exceptions import RefinireRAGError


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class CorpusManagerConfig:
    """Configuration for CorpusManager operations
    CorpusManager操作の設定"""
    
    # Document loading configuration
    loading_config: LoadingConfig = field(default_factory=LoadingConfig)
    
    # Document processing pipeline
    enable_processing: bool = True
    processors: List[DocumentProcessor] = field(default_factory=list)
    
    # Chunking configuration
    enable_chunking: bool = True
    chunking_config: ChunkingConfig = field(default_factory=lambda: ChunkingConfig(
        chunk_size=512,
        overlap=50,
        split_by_sentence=True
    ))
    
    # Embedding configuration
    enable_embedding: bool = True
    embedder: Optional[Embedder] = None
    auto_fit_embedder: bool = True  # Automatically fit TF-IDF embedders
    
    # Storage configuration
    document_store: Optional[DocumentStore] = None
    vector_store: Optional[VectorStore] = None
    store_intermediate_results: bool = True
    
    # Processing options
    batch_size: int = 100
    parallel_processing: bool = False  # Future: enable parallel processing
    
    # Error handling
    fail_on_error: bool = False
    max_errors: int = 10
    
    # Progress reporting
    enable_progress_reporting: bool = True
    progress_interval: int = 10  # Report every N documents


class CorpusManager:
    """Manages document corpus processing pipeline
    
    Coordinates the complete workflow from document loading through embedding storage:
    1. Document loading from various sources
    2. Document processing (normalization, enrichment, etc.)
    3. Document chunking for optimal embedding
    4. Text embedding generation
    5. Storage in document store with full lineage tracking
    """
    
    def __init__(self, config: Optional[CorpusManagerConfig] = None):
        """Initialize CorpusManager with configuration"""
        self.config = config or CorpusManagerConfig()
        
        # Initialize components
        self._loader = UniversalLoader(self.config.loading_config)
        self._pipeline = None
        self._embedder = None
        self._document_store = None
        self._vector_store = None
        
        # Initialize pipeline components
        self._setup_pipeline()
        self._setup_embedder()
        self._setup_document_store()
        self._setup_vector_store()
        
        # Processing statistics
        self._stats = {
            "documents_loaded": 0,
            "documents_processed": 0,
            "documents_chunked": 0,
            "documents_embedded": 0,
            "documents_stored": 0,
            "total_processing_time": 0.0,
            "errors": 0,
            "last_processing_time": None
        }
    
    def _setup_pipeline(self):
        """Setup document processing pipeline"""
        if not self.config.enable_processing and not self.config.enable_chunking:
            return
        
        processors = []
        
        # Add custom processors
        if self.config.enable_processing and self.config.processors:
            processors.extend(self.config.processors)
        
        # Add chunker
        if self.config.enable_chunking:
            chunker = TokenBasedChunker(self.config.chunking_config)
            processors.append(chunker)
        
        if processors:
            self._pipeline = DocumentPipeline(
                processors=processors,
                document_store=None,  # We'll handle storage separately
                store_intermediate_results=False  # We control storage
            )
    
    def _setup_embedder(self):
        """Setup embedding component"""
        if not self.config.enable_embedding:
            return
        
        if self.config.embedder:
            self._embedder = self.config.embedder
        else:
            # Default to TF-IDF embedder
            self._embedder = TFIDFEmbedder(TFIDFEmbeddingConfig(
                max_features=10000,
                min_df=2,
                ngram_range=(1, 2)
            ))
    
    def _setup_document_store(self):
        """Setup document storage"""
        if self.config.document_store:
            self._document_store = self.config.document_store
        else:
            # Default to in-memory SQLite
            self._document_store = SQLiteDocumentStore(":memory:")
    
    def _setup_vector_store(self):
        """Setup vector storage for embeddings"""
        if self.config.vector_store:
            self._vector_store = self.config.vector_store
        else:
            # Default to in-memory vector store
            self._vector_store = InMemoryVectorStore(similarity_metric="cosine")
    
    def load_documents(self, sources: Union[str, Path, List[Union[str, Path]]]) -> List[Document]:
        """Load documents from various sources
        
        Args:
            sources: File paths, directory paths, or list of paths to load
            
        Returns:
            List of loaded documents
        """
        start_time = time.time()
        
        try:
            # Ensure sources is a list
            if not isinstance(sources, list):
                sources = [sources]
            
            # Convert to Path objects
            source_paths = [Path(source) for source in sources]
            
            # Load documents
            all_documents = []
            
            for source_path in source_paths:
                if source_path.is_file():
                    # Load single file
                    document = self._loader.load_single(source_path)
                    all_documents.append(document)
                elif source_path.is_dir():
                    # Load all files in directory
                    for file_path in source_path.glob("*"):
                        if file_path.is_file():
                            try:
                                document = self._loader.load_single(file_path)
                                all_documents.append(document)
                            except Exception as e:
                                logger.warning(f"Failed to load file {file_path}: {e}")
                else:
                    logger.warning(f"Source path does not exist: {source_path}")
            
            # Update statistics
            self._stats["documents_loaded"] += len(all_documents)
            loading_time = time.time() - start_time
            
            if self.config.enable_progress_reporting:
                logger.info(f"Loaded {len(all_documents)} documents in {loading_time:.2f}s")
            
            return all_documents
            
        except Exception as e:
            self._stats["errors"] += 1
            if self.config.fail_on_error:
                raise RefinireRAGError(f"Failed to load documents: {e}")
            
            logger.error(f"Error loading documents: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents through the pipeline
        
        Args:
            documents: Documents to process
            
        Returns:
            Processed documents (may include chunks)
        """
        if not self._pipeline:
            return documents
        
        start_time = time.time()
        processed_documents = []
        
        try:
            # Process documents in batches
            batch_size = self.config.batch_size
            total_docs = len(documents)
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                # Process each document in the batch
                for doc in batch:
                    try:
                        results = self._pipeline.process_document(doc)
                        processed_documents.extend(results)
                        
                        # Store intermediate results if requested
                        if self.config.store_intermediate_results:
                            for result_doc in results:
                                self._document_store.add_document(result_doc)
                        
                    except Exception as e:
                        self._stats["errors"] += 1
                        
                        if self.config.fail_on_error:
                            raise RefinireRAGError(f"Failed to process document {doc.id}: {e}")
                        
                        logger.error(f"Error processing document {doc.id}: {e}")
                
                # Progress reporting
                if self.config.enable_progress_reporting and (i + batch_size) % (self.config.progress_interval * batch_size) == 0:
                    processed_count = min(i + batch_size, total_docs)
                    logger.info(f"Processed {processed_count}/{total_docs} documents")
            
            # Update statistics
            self._stats["documents_processed"] += len(documents)
            
            # Count chunks separately
            chunk_count = sum(1 for doc in processed_documents 
                            if doc.metadata.get("processing_stage") == "chunked")
            self._stats["documents_chunked"] += chunk_count
            
            processing_time = time.time() - start_time
            self._stats["total_processing_time"] += processing_time
            
            if self.config.enable_progress_reporting:
                logger.info(f"Processing completed: {len(processed_documents)} documents in {processing_time:.2f}s")
            
            return processed_documents
            
        except Exception as e:
            if self.config.fail_on_error:
                raise
            
            logger.error(f"Error in document processing: {e}")
            return processed_documents
    
    def embed_documents(self, documents: List[Document]) -> List[Tuple[Document, Any]]:
        """Generate embeddings for documents
        
        Args:
            documents: Documents to embed
            
        Returns:
            List of (document, embedding_result) tuples
        """
        if not self.config.enable_embedding or not self._embedder:
            return [(doc, None) for doc in documents]
        
        start_time = time.time()
        embedded_docs = []
        
        try:
            # Auto-fit TF-IDF embedder if needed
            if (self.config.auto_fit_embedder and 
                isinstance(self._embedder, TFIDFEmbedder) and 
                not self._embedder.is_fitted()):
                
                logger.info("Auto-fitting TF-IDF embedder on document corpus...")
                fit_texts = [doc.content for doc in documents if doc.content.strip()]
                if fit_texts:
                    self._embedder.fit(fit_texts)
                else:
                    logger.warning("No valid texts found for TF-IDF fitting")
                    return [(doc, None) for doc in documents]
            
            # Generate embeddings in batches
            batch_size = self.config.batch_size
            total_docs = len(documents)
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                try:
                    # Generate embeddings for batch
                    results = self._embedder.embed_documents(batch)
                    
                    # Store results
                    for doc, result in zip(batch, results):
                        embedded_docs.append((doc, result))
                        
                        # Store embedding in VectorStore
                        if result.success and self._vector_store:
                            try:
                                # Create vector entry for storage
                                vector_entry = VectorEntry(
                                    document_id=doc.id,
                                    content=doc.content,
                                    embedding=result.vector,  # Use 'vector' not 'embedding'
                                    metadata={
                                        **doc.metadata,
                                        "embedding_model": result.model_name,
                                        "embedding_dimension": result.dimension,
                                        "embedding_processing_time": result.processing_time,
                                        "processing_stage": "embedded"
                                    }
                                )
                                self._vector_store.add_vector(vector_entry)
                                
                            except Exception as ve:
                                logger.error(f"Failed to store embedding for {doc.id}: {ve}")
                        
                        # Store document metadata in DocumentStore if requested
                        if result.success and self.config.store_intermediate_results:
                            # Create enriched document with embedding metadata (no embedding data)
                            enriched_doc = Document(
                                id=f"{doc.id}_embedded",
                                content=doc.content,
                                metadata={
                                    **doc.metadata,
                                    "parent_document_id": doc.id,
                                    "processing_stage": "embedded",
                                    "embedding_model": result.model_name,
                                    "embedding_dimension": result.dimension,
                                    "embedding_processing_time": result.processing_time
                                }
                            )
                            self._document_store.add_document(enriched_doc)
                    
                except Exception as e:
                    self._stats["errors"] += 1
                    
                    if self.config.fail_on_error:
                        raise RefinireRAGError(f"Failed to embed batch starting at {i}: {e}")
                    
                    logger.error(f"Error embedding batch: {e}")
                    # Add failed results
                    for doc in batch:
                        embedded_docs.append((doc, None))
                
                # Progress reporting
                if self.config.enable_progress_reporting and (i + batch_size) % (self.config.progress_interval * batch_size) == 0:
                    processed_count = min(i + batch_size, total_docs)
                    logger.info(f"Embedded {processed_count}/{total_docs} documents")
            
            # Update statistics
            successful_embeddings = sum(1 for _, result in embedded_docs if result and result.success)
            self._stats["documents_embedded"] += successful_embeddings
            
            embedding_time = time.time() - start_time
            
            if self.config.enable_progress_reporting:
                logger.info(f"Embedding completed: {successful_embeddings}/{len(documents)} successful in {embedding_time:.2f}s")
            
            return embedded_docs
            
        except Exception as e:
            if self.config.fail_on_error:
                raise
            
            logger.error(f"Error in document embedding: {e}")
            return embedded_docs
    
    def store_documents(self, documents: List[Document]) -> int:
        """Store documents in the document store
        
        Args:
            documents: Documents to store
            
        Returns:
            Number of documents successfully stored
        """
        if not self._document_store:
            return 0
        
        start_time = time.time()
        stored_count = 0
        
        try:
            # Store documents in batches
            batch_size = self.config.batch_size
            total_docs = len(documents)
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                
                for doc in batch:
                    try:
                        self._document_store.add_document(doc)
                        stored_count += 1
                        
                    except Exception as e:
                        self._stats["errors"] += 1
                        
                        if self.config.fail_on_error:
                            raise RefinireRAGError(f"Failed to store document {doc.id}: {e}")
                        
                        logger.error(f"Error storing document {doc.id}: {e}")
                
                # Progress reporting
                if self.config.enable_progress_reporting and (i + batch_size) % (self.config.progress_interval * batch_size) == 0:
                    processed_count = min(i + batch_size, total_docs)
                    logger.info(f"Stored {processed_count}/{total_docs} documents")
            
            # Update statistics
            self._stats["documents_stored"] += stored_count
            
            storage_time = time.time() - start_time
            
            if self.config.enable_progress_reporting:
                logger.info(f"Storage completed: {stored_count}/{len(documents)} documents in {storage_time:.2f}s")
            
            return stored_count
            
        except Exception as e:
            if self.config.fail_on_error:
                raise
            
            logger.error(f"Error in document storage: {e}")
            return stored_count
    
    def process_corpus(self, sources: Union[str, Path, List[Union[str, Path]]]) -> Dict[str, Any]:
        """Complete corpus processing pipeline
        
        Args:
            sources: Document sources to process
            
        Returns:
            Processing results and statistics
        """
        start_time = time.time()
        
        try:
            # Step 1: Load documents
            logger.info("Starting corpus processing pipeline...")
            documents = self.load_documents(sources)
            
            if not documents:
                logger.warning("No documents loaded, stopping pipeline")
                return self._get_processing_results(start_time, documents, [], [], [])
            
            # Step 2: Process documents (includes chunking)
            processed_documents = self.process_documents(documents)
            
            # Step 3: Generate embeddings
            embedded_docs = self.embed_documents(processed_documents)
            
            # Step 4: Store final results
            if not self.config.store_intermediate_results:
                # Store all processed documents
                stored_count = self.store_documents(processed_documents)
            else:
                # Documents already stored during processing
                stored_count = len(processed_documents)
            
            # Update final statistics
            total_time = time.time() - start_time
            self._stats["last_processing_time"] = time.time()
            
            # Prepare results
            results = self._get_processing_results(
                start_time, documents, processed_documents, embedded_docs, embedded_docs
            )
            
            logger.info(f"Corpus processing completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Corpus processing failed: {e}")
            
            if self.config.fail_on_error:
                raise RefinireRAGError(f"Corpus processing failed: {e}")
            
            return self._get_processing_results(start_time, [], [], [], [])
    
    def _get_processing_results(self, start_time: float, loaded_docs: List[Document], 
                              processed_docs: List[Document], embedded_docs: List[Tuple], 
                              final_docs: List) -> Dict[str, Any]:
        """Compile processing results and statistics"""
        total_time = time.time() - start_time
        
        successful_embeddings = sum(1 for _, result in embedded_docs if result and result.success)
        
        return {
            "success": True,
            "total_processing_time": total_time,
            "documents_loaded": len(loaded_docs),
            "documents_processed": len(processed_docs),
            "documents_embedded": successful_embeddings,
            "documents_stored": self._stats["documents_stored"],
            "total_errors": self._stats["errors"],
            "pipeline_stats": self._pipeline.get_pipeline_stats() if self._pipeline else {},
            "embedder_stats": self._embedder.get_embedding_stats() if self._embedder else {},
            "storage_stats": getattr(self._document_store, 'get_stats', lambda: {})() if self._document_store else {},
            "final_document_count": len(final_docs),
            "processing_stages": self._analyze_processing_stages(processed_docs)
        }
    
    def _analyze_processing_stages(self, documents: List[Document]) -> Dict[str, int]:
        """Analyze documents by processing stage"""
        stages = {}
        
        for doc in documents:
            stage = doc.metadata.get("processing_stage", "original")
            stages[stage] = stages.get(stage, 0) + 1
        
        return stages
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get comprehensive corpus processing statistics"""
        base_stats = self._stats.copy()
        
        # Add component statistics
        if self._pipeline:
            base_stats["pipeline_stats"] = self._pipeline.get_pipeline_stats()
        
        if self._embedder:
            base_stats["embedder_stats"] = self._embedder.get_embedding_stats()
        
        if self._document_store:
            base_stats["storage_stats"] = getattr(self._document_store, 'get_stats', lambda: {})()
        
        if self._vector_store:
            base_stats["vector_stats"] = self._vector_store.get_stats()
        
        return base_stats
    
    def search_documents(self, query: str, limit: int = 10, use_semantic: bool = True) -> List[Any]:
        """Search documents in the corpus
        
        Args:
            query: Search query
            limit: Maximum number of results
            use_semantic: Whether to use semantic search (embeddings) or text search
            
        Returns:
            Search results
        """
        # Try semantic search first if available and requested
        if use_semantic and self._vector_store and self._embedder:
            try:
                return self._semantic_search(query, limit)
            except Exception as e:
                logger.warning(f"Semantic search failed, falling back to text search: {e}")
        
        # Fallback to text search in DocumentStore
        if self._document_store:
            try:
                return self._document_store.search_by_content(query, limit=limit)
            except AttributeError:
                logger.warning("Document store doesn't support search functionality")
        
        return []
    
    def _semantic_search(self, query: str, limit: int = 10) -> List[Any]:
        """Perform semantic search using embeddings
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            Semantic search results
        """
        try:
            # Generate embedding for the query
            from datetime import datetime
            query_doc = Document(
                id="query", 
                content=query, 
                metadata={
                    "path": "/query",
                    "created_at": datetime.now().isoformat(),
                    "file_type": "query",
                    "size_bytes": len(query)
                }
            )
            query_results = self._embedder.embed_documents([query_doc])
            
            if not query_results or not query_results[0].success:
                logger.warning("Failed to generate query embedding")
                return []
            
            query_embedding = query_results[0].vector
            
            # Search for similar vectors
            similar_results = self._vector_store.search_similar(
                query_vector=query_embedding,
                limit=limit
            )
            
            logger.debug(f"Found {len(similar_results)} similar documents for query: '{query}'")
            return similar_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_document_lineage(self, document_id: str) -> List[Document]:
        """Get the complete lineage of a document"""
        if not self._document_store:
            return []
        
        return self._document_store.get_documents_by_lineage(document_id)
    
    def cleanup(self):
        """Clean up resources"""
        if self._document_store and hasattr(self._document_store, 'close'):
            self._document_store.close()
        
        if self._vector_store and hasattr(self._vector_store, 'close'):
            self._vector_store.close()
        
        if self._embedder and hasattr(self._embedder, 'clear_cache'):
            self._embedder.clear_cache()