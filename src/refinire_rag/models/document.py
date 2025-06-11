"""
Document data model for refinire-rag
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    """Document data model
    
    A document consists of an ID, content, and flexible metadata.
    Only 4 metadata fields are required, others are completely flexible.
    
    Required metadata fields:
    - path: str - File path
    - created_at: str - ISO 8601 timestamp  
    - file_type: str - File extension
    - size_bytes: int - File size in bytes
    """
    
    id: str
    content: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate required metadata fields and set defaults for missing ones"""
        # Set default values for missing required fields
        defaults = {
            "path": f"virtual://{self.id}",
            "created_at": "2024-01-01T00:00:00Z",
            "file_type": "txt",
            "size_bytes": len(self.content.encode('utf-8'))
        }
        
        for field, default_value in defaults.items():
            if field not in self.metadata:
                self.metadata[field] = default_value
    
    @property
    def path(self) -> str:
        """Get file path from metadata"""
        return self.metadata["path"]
    
    @property
    def created_at(self) -> str:
        """Get creation timestamp from metadata"""
        return self.metadata["created_at"]
    
    @property
    def file_type(self) -> str:
        """Get file type from metadata"""
        return self.metadata["file_type"]
    
    @property
    def size_bytes(self) -> int:
        """Get file size from metadata"""
        return self.metadata["size_bytes"]
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with optional default"""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value"""
        self.metadata[key] = value
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """Update multiple metadata values"""
        self.metadata.update(updates)