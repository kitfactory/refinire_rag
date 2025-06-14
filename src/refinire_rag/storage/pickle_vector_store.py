"""
Pickle-based Vector Store Implementation

Persistent vector storage using pickle serialization.
Good for development and small to medium datasets that need persistence.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import numpy as np

from .in_memory_vector_store import InMemoryVectorStore
from .vector_store import VectorEntry, VectorSearchResult, VectorStoreStats
from ..exceptions import StorageError

logger = logging.getLogger(__name__)


class PickleVectorStore(InMemoryVectorStore):
    """Persistent vector store using pickle serialization
    
    Extends InMemoryVectorStore with automatic persistence to disk.
    Data is kept in memory for fast access and periodically saved to disk.
    """
    
    def __init__(
        self, 
        file_path: str = "./data/vectors.pkl",
        similarity_metric: str = "cosine",
        auto_save: bool = True,
        save_interval: int = 10,
        config: Optional[Dict] = None
    ):
        """Initialize pickle-based vector store
        
        Args:
            file_path: Path to pickle file for persistence
            similarity_metric: Similarity metric to use
            auto_save: Whether to automatically save changes
            save_interval: Number of operations before auto-save (if auto_save=True)
            config: Optional configuration for DocumentProcessor
        """
        pickle_config = config or {
            "file_path": file_path,
            "similarity_metric": similarity_metric,
            "auto_save": auto_save,
            "save_interval": save_interval
        }
        super().__init__(similarity_metric=similarity_metric, config=pickle_config)
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.save_interval = save_interval
        self._operations_since_save = 0
        
        # Load existing data
        self._load_from_disk()
        
        logger.info(f"Initialized PickleVectorStore at {self.file_path}")
    
    def add_vector(self, entry: VectorEntry) -> str:
        """Add a vector entry and optionally save to disk"""
        
        result = super().add_vector(entry)
        self._maybe_save()
        return result
    
    def add_vectors(self, entries: List[VectorEntry]) -> List[str]:
        """Add multiple vector entries and optionally save to disk"""
        
        result = super().add_vectors(entries)
        self._maybe_save()
        return result
    
    def update_vector(self, entry: VectorEntry) -> bool:
        """Update vector entry and optionally save to disk"""
        
        result = super().update_vector(entry)
        if result:
            self._maybe_save()
        return result
    
    def delete_vector(self, document_id: str) -> bool:
        """Delete vector entry and optionally save to disk"""
        
        result = super().delete_vector(document_id)
        if result:
            self._maybe_save()
        return result
    
    def clear(self) -> bool:
        """Clear all vectors and save to disk"""
        
        result = super().clear()
        if result:
            self.save_to_disk()
        return result
    
    def save_to_disk(self) -> bool:
        """Explicitly save current state to disk"""
        
        try:
            # Prepare data for serialization
            data = {
                'vectors': self._vectors,
                'similarity_metric': self.similarity_metric,
                'metadata': {
                    'total_vectors': len(self._vectors),
                    'vector_dimension': self.get_vector_dimension(),
                    'save_timestamp': np.datetime64('now')
                }
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_path = self.file_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_path.replace(self.file_path)
            
            self._operations_since_save = 0
            logger.info(f"Saved {len(self._vectors)} vectors to {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vectors to disk: {e}")
            return False
    
    def load_from_disk(self) -> bool:
        """Explicitly load state from disk"""
        return self._load_from_disk()
    
    def _load_from_disk(self) -> bool:
        """Load vectors from pickle file"""
        
        try:
            if not self.file_path.exists():
                logger.info(f"No existing vector file found at {self.file_path}")
                return True
            
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Restore vectors
            self._vectors = data.get('vectors', {})
            
            # Restore similarity metric if compatible
            saved_metric = data.get('similarity_metric', 'cosine')
            if saved_metric != self.similarity_metric:
                logger.warning(f"Similarity metric mismatch: loaded '{saved_metric}', expected '{self.similarity_metric}'")
            
            # Force matrix rebuild
            self._needs_rebuild = True
            
            # Log metadata if available
            metadata = data.get('metadata', {})
            if metadata:
                logger.info(f"Loaded {metadata.get('total_vectors', 0)} vectors from {self.file_path}")
                if 'save_timestamp' in metadata:
                    logger.debug(f"Data saved at: {metadata['save_timestamp']}")
            else:
                logger.info(f"Loaded {len(self._vectors)} vectors from {self.file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vectors from disk: {e}")
            # Continue with empty store
            self._vectors = {}
            return False
    
    def _maybe_save(self):
        """Save to disk if auto_save is enabled and save_interval is reached"""
        
        if not self.auto_save:
            return
        
        self._operations_since_save += 1
        
        if self._operations_since_save >= self.save_interval:
            self.save_to_disk()
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics including file size"""
        
        stats = super().get_stats()
        
        # Add file size information
        if self.file_path.exists():
            file_size = self.file_path.stat().st_size
            stats.storage_size_bytes = file_size
        
        # Update index type
        stats.index_type = "exact_pickle"
        
        return stats
    
    def backup_to_file(self, backup_path: str) -> bool:
        """Create a backup of the vector store"""
        
        try:
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save current state first
            self.save_to_disk()
            
            # Copy to backup location
            import shutil
            shutil.copy2(self.file_path, backup_file)
            
            logger.info(f"Created backup at {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore vector store from a backup"""
        
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file {backup_path} does not exist")
                return False
            
            # Copy backup to main file
            import shutil
            shutil.copy2(backup_file, self.file_path)
            
            # Reload from disk
            return self._load_from_disk()
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False
    
    def optimize_storage(self) -> bool:
        """Optimize storage (defragment, remove unused space)"""
        
        try:
            # For pickle store, this is simply saving to disk
            # which rewrites the entire file
            return self.save_to_disk()
            
        except Exception as e:
            logger.error(f"Failed to optimize storage: {e}")
            return False
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the storage file"""
        
        try:
            if not self.file_path.exists():
                return {
                    "exists": False,
                    "path": str(self.file_path)
                }
            
            stat = self.file_path.stat()
            
            return {
                "exists": True,
                "path": str(self.file_path),
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "operations_since_save": self._operations_since_save,
                "auto_save": self.auto_save,
                "save_interval": self.save_interval
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Save and close the vector store"""
        if self.auto_save and self._operations_since_save > 0:
            self.save_to_disk()
        logger.info("PickleVectorStore closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass