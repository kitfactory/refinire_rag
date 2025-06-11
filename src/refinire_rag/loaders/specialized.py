"""
Specialized loader implementations for different file formats
"""

from typing import List, Union, Optional
from pathlib import Path
import logging

from .base import Loader, MetadataGenerator
from ..models.document import Document
from ..models.config import LoadingConfig
from ..exceptions import LoaderError

logger = logging.getLogger(__name__)


class SpecializedLoader(Loader):
    """Base class for specialized loaders"""
    
    def __init__(
        self,
        metadata_generator: Optional[MetadataGenerator] = None,
        config: Optional[LoadingConfig] = None
    ):
        super().__init__(metadata_generator, config)
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract text content from file (must be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _extract_content")
    
    def load_single(self, path: Union[str, Path]) -> Document:
        """Standard implementation using _extract_content"""
        
        try:
            # Extract content using specialized method
            content = self._extract_content(path)
            
            # Generate metadata
            metadata = self._generate_base_metadata(path)
            
            # Generate document ID
            document_id = self._generate_document_id(path)
            
            # Add loader-specific metadata
            metadata.update({
                "loader_used": self.__class__.__name__,
                "content_length": len(content)
            })
            
            return Document(
                id=document_id,
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            raise LoaderError(f"Failed to load {path}: {e}") from e


class TextLoader(SpecializedLoader):
    """Loader for plain text files"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content from text file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                        logger.info(f"Successfully read {path} with {encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            raise LoaderError(f"Could not decode {path} with any supported encoding")
    
    def supported_formats(self) -> List[str]:
        return ['.txt']


class MarkdownLoader(SpecializedLoader):
    """Loader for Markdown files"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content from Markdown file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Could add markdown-specific processing here
            # For now, just return raw content
            return content
            
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                        logger.info(f"Successfully read {path} with {encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            raise LoaderError(f"Could not decode {path} with any supported encoding")
    
    def supported_formats(self) -> List[str]:
        return ['.md', '.markdown']


class PDFLoader(SpecializedLoader):
    """Basic PDF loader using PyPDF2"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
                
        except ImportError:
            raise LoaderError(
                "PyPDF2 required for PDF loading. Install with: pip install PyPDF2"
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract text from PDF {path}: {e}") from e
    
    def supported_formats(self) -> List[str]:
        return ['.pdf']


class HTMLLoader(SpecializedLoader):
    """Loader for HTML files"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract text content from HTML file"""
        try:
            from bs4 import BeautifulSoup
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            raise LoaderError(
                "BeautifulSoup4 required for HTML loading. Install with: pip install beautifulsoup4"
            )
        except Exception as e:
            raise LoaderError(f"Failed to extract text from HTML {path}: {e}") from e
    
    def supported_formats(self) -> List[str]:
        return ['.html', '.htm']


class CSVLoader(SpecializedLoader):
    """Loader for CSV files"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content from CSV file"""
        try:
            import csv
            
            content_lines = []
            
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Join row values with spaces
                    content_lines.append(' '.join(str(cell) for cell in row))
            
            return '\n'.join(content_lines)
            
        except Exception as e:
            raise LoaderError(f"Failed to read CSV {path}: {e}") from e
    
    def supported_formats(self) -> List[str]:
        return ['.csv']


class JSONLoader(SpecializedLoader):
    """Loader for JSON files"""
    
    def _extract_content(self, path: Union[str, Path]) -> str:
        """Extract content from JSON file"""
        try:
            import json
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text format
            def extract_text_from_json(obj, prefix=""):
                texts = []
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        texts.extend(extract_text_from_json(value, new_prefix))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                        texts.extend(extract_text_from_json(item, new_prefix))
                else:
                    texts.append(f"{prefix}: {str(obj)}")
                
                return texts
            
            text_parts = extract_text_from_json(data)
            return '\n'.join(text_parts)
            
        except Exception as e:
            raise LoaderError(f"Failed to read JSON {path}: {e}") from e
    
    def supported_formats(self) -> List[str]:
        return ['.json']