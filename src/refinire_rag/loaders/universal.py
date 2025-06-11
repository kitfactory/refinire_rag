"""
Universal loader that delegates to specialized loaders based on file extension
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

from .base import Loader, MetadataGenerator
from .specialized import (
    TextLoader,
    MarkdownLoader,
    PDFLoader,
    HTMLLoader,
    CSVLoader,
    JSONLoader
)
from ..models.document import Document
from ..models.config import LoadingConfig
from ..exceptions import LoaderError

logger = logging.getLogger(__name__)


class UniversalLoader(Loader):
    """Universal loader that delegates to specialized loaders based on file extension"""
    
    def __init__(
        self,
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None,
        custom_loaders: Optional[Dict[str, Loader]] = None
    ):
        """Initialize universal loader with custom loader mappings
        
        Args:
            metadata_generator: Optional metadata generator
            config: Loading configuration
            custom_loaders: Dict mapping file extensions to loader instances
        """
        super().__init__(metadata_generator, config)
        
        # Default loaders for common formats
        self._default_loaders = {
            '.txt': TextLoader(metadata_generator, config),
            '.md': MarkdownLoader(metadata_generator, config),
            '.markdown': MarkdownLoader(metadata_generator, config),
            '.pdf': PDFLoader(metadata_generator, config),
            '.html': HTMLLoader(metadata_generator, config),
            '.htm': HTMLLoader(metadata_generator, config),
            '.json': JSONLoader(metadata_generator, config),
            '.csv': CSVLoader(metadata_generator, config),
        }
        
        # Custom loaders override defaults
        if custom_loaders:
            self._default_loaders.update(custom_loaders)
        
        # Available loaders (can be extended by packages)
        self._available_loaders = {}
        self._register_available_loaders()
    
    def register_loader(self, extension: str, loader_class: type):
        """Register a new loader for a file extension
        
        Args:
            extension: File extension (e.g., '.xml')
            loader_class: Loader class to use for this extension
        """
        self._available_loaders[extension] = loader_class
        
        # Create instance if not already present
        if extension not in self._default_loaders:
            self._default_loaders[extension] = loader_class(
                self.metadata_generator, 
                self.config
            )
    
    def _register_available_loaders(self):
        """Register loaders from available packages"""
        
        # Try to import and register docling loader
        try:
            from .extensions.docling import DoclingLoader
            self.register_loader('.pdf', DoclingLoader)
            self.register_loader('.docx', DoclingLoader)
            self.register_loader('.pptx', DoclingLoader)
            logger.info("Registered DoclingLoader for advanced document processing")
        except ImportError:
            logger.debug("DoclingLoader not available")
        
        # Try to import and register unstructured loader
        try:
            from .extensions.unstructured import UnstructuredLoader
            # Only register if docling is not available
            if '.pdf' not in self._available_loaders:
                self.register_loader('.pdf', UnstructuredLoader)
            if '.docx' not in self._available_loaders:
                self.register_loader('.docx', UnstructuredLoader)
            logger.info("Registered UnstructuredLoader for document processing")
        except ImportError:
            logger.debug("UnstructuredLoader not available")
        
        # Try to import other specialized loaders
        try:
            from .extensions.excel import ExcelLoader
            self.register_loader('.xlsx', ExcelLoader)
            self.register_loader('.xls', ExcelLoader)
            logger.info("Registered ExcelLoader for Excel files")
        except ImportError:
            logger.debug("ExcelLoader not available")
    
    def load_single(self, path: Union[str, Path]) -> Document:
        """Load single document using appropriate loader"""
        path_obj = Path(path)
        extension = path_obj.suffix.lower()
        
        # Find appropriate loader
        loader = self._default_loaders.get(extension)
        if not loader:
            supported_formats = list(self._default_loaders.keys())
            raise LoaderError(
                f"No loader available for file type: {extension}. "
                f"Supported formats: {supported_formats}"
            )
        
        # Load document
        try:
            document = loader.load_single(path)
            
            # Add universal loader metadata
            document.metadata.update({
                "loader_used": loader.__class__.__name__,
                "loader_type": "universal",
                "universal_loader_version": "1.0"
            })
            
            return document
            
        except Exception as e:
            raise LoaderError(
                f"Failed to load {path} using {loader.__class__.__name__}: {e}"
            ) from e
    
    def supported_formats(self) -> List[str]:
        """Get all supported formats from registered loaders"""
        return list(self._default_loaders.keys())
    
    def get_loader_for_extension(self, extension: str) -> Optional[Loader]:
        """Get the loader that will be used for a given extension"""
        return self._default_loaders.get(extension.lower())
    
    def list_available_loaders(self) -> Dict[str, str]:
        """List all available loaders and their extensions"""
        return {
            ext: loader.__class__.__name__ 
            for ext, loader in self._default_loaders.items()
        }
    
    def can_load(self, path: Union[str, Path]) -> bool:
        """Check if the loader can handle the given file"""
        path_obj = Path(path)
        extension = path_obj.suffix.lower()
        return extension in self._default_loaders
    
    def get_loader_info(self, extension: str) -> Dict[str, str]:
        """Get information about the loader for a given extension"""
        loader = self.get_loader_for_extension(extension)
        if not loader:
            return {"error": f"No loader available for {extension}"}
        
        return {
            "loader_class": loader.__class__.__name__,
            "supported_formats": loader.supported_formats(),
            "module": loader.__class__.__module__
        }
    
    def validate_paths(self, paths: List[Union[str, Path]]) -> Dict[str, List[str]]:
        """Validate that all paths can be loaded
        
        Returns:
            Dict with 'supported' and 'unsupported' lists
        """
        supported = []
        unsupported = []
        
        for path in paths:
            if self.can_load(path):
                supported.append(str(path))
            else:
                unsupported.append(str(path))
        
        return {
            "supported": supported,
            "unsupported": unsupported
        }