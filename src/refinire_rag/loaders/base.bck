"""
Base loader interfaces and abstract classes
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Callable
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import hashlib
import logging

from ..models.document import Document
from ..models.config import LoadingConfig, LoadingResult
from ..exceptions import LoaderError
from ..processing.document_processor import DocumentProcessor, DocumentProcessorConfig
from dataclasses import dataclass
from typing import Type

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig(DocumentProcessorConfig):
    """Configuration for Loader processors"""
    
    # Loading configuration
    loading_config: LoadingConfig = None
    
    # Source configuration 
    source_paths: List[str] = None  # Paths to load from
    auto_detect_sources: bool = True  # Auto-detect from input document metadata
    
    # Output configuration
    preserve_trigger: bool = False  # Include the trigger document in output
    add_loading_metadata: bool = True  # Add metadata about loading process
    
    def __post_init__(self):
        """Initialize default values"""
        if self.loading_config is None:
            self.loading_config = LoadingConfig()
        if self.source_paths is None:
            self.source_paths = []


class MetadataGenerator(ABC):
    """Interface for generating additional metadata"""
    
    @abstractmethod
    def generate_metadata(self, required_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate additional metadata from required fields
        
        Args:
            required_metadata: Required metadata fields (path, created_at, file_type, size_bytes)
            
        Returns:
            Additional metadata to be merged with required fields
        """
        pass


class Loader(DocumentProcessor, ABC):
    """Base interface for document loading"""
    
    def __init__(
        self, 
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoaderConfig] = None,
        loading_config: Optional[LoadingConfig] = None  # Backwards compatibility
    ):
        """Initialize loader with optional metadata generator and config"""
        # Handle backwards compatibility with loading_config parameter
        if config is None and loading_config is not None:
            config = LoaderConfig(loading_config=loading_config)
        elif config is None:
            config = LoaderConfig()
        
        # Initialize DocumentProcessor
        super().__init__(config)
        
        self.metadata_generator = metadata_generator
        # Keep loading_config for backwards compatibility
        self.loading_config = self.config.loading_config
    
    @classmethod
    def get_config_class(cls) -> Type[LoaderConfig]:
        """Get the configuration class for this processor"""
        return LoaderConfig
    
    def process(self, document: Document, config: Optional[LoaderConfig] = None) -> List[Document]:
        """Process document by loading related documents from file system
        
        Args:
            document: Input document (trigger) - may contain source paths in metadata
            config: Optional loader configuration override
            
        Returns:
            List of loaded documents (optionally including trigger)
        """
        try:
            # Use provided config or fall back to instance config
            loader_config = config or self.config
            
            logger.debug(f"Processing document {document.id} with {self.__class__.__name__}")
            
            # Collect source paths
            source_paths = self._collect_source_paths(document, loader_config)
            
            if not source_paths:
                logger.warning(f"No source paths found for {self.__class__.__name__}")
                return [document] if loader_config.preserve_trigger else []
            
            # Load documents from sources
            loaded_documents = self._load_documents_from_sources(source_paths)
            
            # Add loading metadata if configured
            if loader_config.add_loading_metadata:
                loaded_documents = self._add_loading_metadata(document, loaded_documents)
            
            # Prepare output
            output_documents = []
            if loader_config.preserve_trigger:
                output_documents.append(document)
            output_documents.extend(loaded_documents)
            
            logger.info(f"{self.__class__.__name__}: Loaded {len(loaded_documents)} documents for {document.id}")
            return output_documents
            
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} for document {document.id}: {e}")
            # Return trigger document on error if configured
            return [document] if self.config.preserve_trigger else []
    
    @abstractmethod
    def load_single(self, path: Union[str, Path]) -> Document:
        """Load a single document from path (must be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        pass
    
    def load(self, path: Union[str, Path]) -> Document:
        """Load a single document from path"""
        return self.load_single(path)
    
    def load_batch(
        self, 
        paths: List[Union[str, Path]], 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Load multiple documents with optional parallel processing
        
        Args:
            paths: List of file paths to load
            progress_callback: Optional callback function(completed, total)
        
        Returns:
            LoadingResult with documents and statistics
        """
        start_time = time.time()
        
        if self.loading_config.parallel and len(paths) > 1:
            result = self._load_parallel(paths, progress_callback)
        else:
            result = self._load_sequential(paths, progress_callback)
        
        result.total_time_seconds = time.time() - start_time
        return result
    
    async def load_batch_async(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Async version of batch loading"""
        start_time = time.time()
        
        if self.loading_config.parallel and len(paths) > 1:
            result = await self._load_async_parallel(paths, progress_callback)
        else:
            result = self._load_sequential(paths, progress_callback)
        
        result.total_time_seconds = time.time() - start_time
        return result
    
    def _load_sequential(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Sequential loading implementation"""
        documents = []
        failed_paths = []
        errors = []
        
        for i, path in enumerate(paths):
            try:
                doc = self.load_single(path)
                documents.append(doc)
                if progress_callback:
                    progress_callback(i + 1, len(paths))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                failed_paths.append(str(path))
                errors.append(e)
                if not self.loading_config.skip_errors:
                    raise LoaderError(f"Failed to load {path}") from e
        
        return LoadingResult(
            documents=documents,
            failed_paths=failed_paths,
            errors=errors,
            total_time_seconds=0,  # Will be set by caller
            successful_count=len(documents),
            failed_count=len(failed_paths)
        )
    
    def _load_parallel(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Parallel loading using threads or processes"""
        documents = []
        failed_paths = []
        errors = []
        completed = 0
        
        ExecutorClass = ProcessPoolExecutor if self.loading_config.use_multiprocessing else ThreadPoolExecutor
        
        with ExecutorClass(max_workers=self.loading_config.max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self._safe_load_single, path): path 
                for path in paths
            }
            
            # Collect results
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    result = future.result(timeout=self.loading_config.timeout_per_file)
                    if result:
                        documents.append(result)
                    else:
                        failed_paths.append(str(path))
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
                    failed_paths.append(str(path))
                    errors.append(e)
                    if not self.loading_config.skip_errors:
                        # Cancel remaining futures
                        for f in future_to_path:
                            f.cancel()
                        raise LoaderError(f"Failed to load {path}") from e
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(paths))
        
        return LoadingResult(
            documents=documents,
            failed_paths=failed_paths,
            errors=errors,
            total_time_seconds=0,  # Will be set by caller
            successful_count=len(documents),
            failed_count=len(failed_paths)
        )
    
    async def _load_async_parallel(
        self,
        paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> LoadingResult:
        """Async parallel loading"""
        documents = []
        failed_paths = []
        errors = []
        
        semaphore = asyncio.Semaphore(self.loading_config.max_workers or 10)
        
        async def load_with_semaphore(path):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self._safe_load_single, path)
        
        # Create tasks
        tasks = [load_with_semaphore(path) for path in paths]
        
        # Execute with progress tracking
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await coro
                if result:
                    documents.append(result)
                else:
                    failed_paths.append(str(paths[i]))
            except Exception as e:
                logger.error(f"Failed to load {paths[i]}: {e}")
                failed_paths.append(str(paths[i]))
                errors.append(e)
                if not self.loading_config.skip_errors:
                    # Cancel remaining tasks
                    for task in tasks:
                        task.cancel()
                    raise LoaderError(f"Failed to load {paths[i]}") from e
            
            if progress_callback:
                progress_callback(i + 1, len(paths))
        
        return LoadingResult(
            documents=documents,
            failed_paths=failed_paths,
            errors=errors,
            total_time_seconds=0,  # Will be set by caller
            successful_count=len(documents),
            failed_count=len(failed_paths)
        )
    
    def _safe_load_single(self, path: Union[str, Path]) -> Optional[Document]:
        """Safe wrapper for load_single that handles errors"""
        try:
            return self.load_single(path)
        except Exception as e:
            if self.loading_config.skip_errors:
                logger.warning(f"Skipping {path} due to error: {e}")
                return None
            raise
    
    def _generate_base_metadata(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Generate required metadata fields"""
        path_obj = Path(path)
        
        try:
            stat = path_obj.stat()
        except OSError as e:
            raise LoaderError(f"Cannot access file {path}: {e}") from e
        
        base_metadata = {
            "path": str(path),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "file_type": path_obj.suffix,
            "size_bytes": stat.st_size
        }
        
        # Generate additional metadata if generator is provided
        if self.metadata_generator:
            try:
                additional_metadata = self.metadata_generator.generate_metadata(base_metadata)
                base_metadata.update(additional_metadata)
            except Exception as e:
                logger.warning(f"Failed to generate metadata for {path}: {e}")
                if not self.loading_config.skip_errors:
                    raise
        
        return base_metadata
    
    def _generate_document_id(self, path: Union[str, Path]) -> str:
        """Generate unique document ID from path"""
        return hashlib.md5(str(path).encode()).hexdigest()
    
    def _collect_source_paths(self, document: Document, config: LoaderConfig) -> List[str]:
        """Collect source paths from configuration and document metadata"""
        
        source_paths = []
        
        # Add configured source paths
        source_paths.extend(config.source_paths)
        
        # Auto-detect sources from document metadata if enabled
        if config.auto_detect_sources:
            # Check for common metadata fields that might contain file paths
            metadata_fields = [
                "source_paths",
                "related_files", 
                "additional_sources",
                "file_references",
                "source_directory"
            ]
            
            for field in metadata_fields:
                if field in document.metadata:
                    field_value = document.metadata[field]
                    
                    if isinstance(field_value, str):
                        source_paths.append(field_value)
                    elif isinstance(field_value, list):
                        source_paths.extend(str(item) for item in field_value)
            
            # Check if document path suggests a directory to scan
            if "path" in document.metadata:
                doc_path = Path(document.metadata["path"])
                if doc_path.is_file():
                    # Add parent directory for scanning
                    parent_dir = doc_path.parent
                    if parent_dir.exists():
                        source_paths.append(str(parent_dir))
        
        # Remove duplicates and validate paths
        unique_paths = []
        for path in source_paths:
            if path not in unique_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    unique_paths.append(path)
                else:
                    logger.warning(f"Source path does not exist: {path}")
        
        return unique_paths
    
    def _load_documents_from_sources(self, source_paths: List[str]) -> List[Document]:
        """Load documents from the collected source paths"""
        
        loaded_documents = []
        
        try:
            for source_path in source_paths:
                path_obj = Path(source_path)
                
                if path_obj.is_file():
                    # Load single file
                    try:
                        document = self.load_single(path_obj)
                        loaded_documents.append(document)
                        logger.debug(f"Loaded file: {source_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load file {source_path}: {e}")
                
                elif path_obj.is_dir():
                    # Load all files in directory (if supported)
                    try:
                        # Try to use load_directory if available, otherwise skip
                        if hasattr(self, 'load_directory'):
                            result = self.load_directory(path_obj)
                            loaded_documents.extend(result.documents)
                            logger.debug(f"Loaded {len(result.documents)} files from directory: {source_path}")
                        else:
                            logger.warning(f"Directory loading not supported by {self.__class__.__name__}: {source_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load directory {source_path}: {e}")
                
                else:
                    logger.warning(f"Source path is neither file nor directory: {source_path}")
            
            return loaded_documents
            
        except Exception as e:
            logger.error(f"Error loading documents from sources: {e}")
            return []
    
    def _add_loading_metadata(self, trigger_doc: Document, loaded_docs: List[Document]) -> List[Document]:
        """Add metadata about the loading process to loaded documents"""
        
        enriched_docs = []
        
        for doc in loaded_docs:
            # Create enriched document with loading metadata
            enriched_metadata = {
                **doc.metadata,
                "loaded_by_processor": self.__class__.__name__,
                "trigger_document_id": trigger_doc.id,
                "loaded_in_pipeline": True
            }
            
            enriched_doc = Document(
                id=doc.id,  # Keep original ID
                content=doc.content,
                metadata=enriched_metadata
            )
            
            enriched_docs.append(enriched_doc)
        
        return enriched_docs


class PathBasedMetadataGenerator(MetadataGenerator):
    """Generate metadata based on file path patterns"""
    
    def __init__(self, path_rules: Dict[str, Dict[str, Any]]):
        """Initialize with path-based rules
        
        Args:
            path_rules: Dict mapping path patterns to metadata
                例: {
                    "/docs/public/*": {"access_group": "public", "classification": "open"},
                    "/docs/internal/*": {"access_group": "employees", "classification": "internal"},
                    "/docs/confidential/*": {"access_group": "managers", "classification": "confidential"}
                }
        """
        self.path_rules = path_rules
    
    def generate_metadata(self, required_metadata: Dict[str, Any]) -> Dict[str, Any]:
        path = required_metadata["path"]
        additional_metadata = {}
        
        # Apply path-based rules
        for pattern, metadata in self.path_rules.items():
            if self._matches_pattern(path, pattern):
                additional_metadata.update(metadata)
                break
        
        # Extract folder-based information
        path_obj = Path(path)
        additional_metadata.update({
            "filename": path_obj.name,
            "directory": str(path_obj.parent),
            "folder_name": path_obj.parent.name
        })
        
        # Add file type specific metadata
        file_type = required_metadata["file_type"]
        if file_type == ".pdf":
            additional_metadata["document_type"] = "pdf_document"
        elif file_type in [".md", ".txt"]:
            additional_metadata["document_type"] = "text_document"
        elif file_type in [".docx", ".doc"]:
            additional_metadata["document_type"] = "word_document"
        
        return additional_metadata
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching (can be enhanced with regex)"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)