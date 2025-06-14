"""
Tests for incremental directory loader functionality
インクリメンタルディレクトリローダー機能のテスト
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock

from refinire_rag.loader.incremental_directory_loader import IncrementalDirectoryLoader
from refinire_rag.loader.file_tracker import FileTracker
from refinire_rag.loader.models.filter_config import FilterConfig
from refinire_rag.loader.filters.extension_filter import ExtensionFilter
from refinire_rag.loader.filters.size_filter import SizeFilter
from refinire_rag.loader.models.file_info import FileInfo
from refinire_rag.loader.models.change_set import ChangeSet
from refinire_rag.storage.document_store import DocumentStore, SearchResult
from refinire_rag.models.document import Document


class TestExtensionFilter:
    """Test ExtensionFilter functionality"""
    
    def test_include_extensions(self):
        """Test filtering by included extensions"""
        filter_instance = ExtensionFilter(include_extensions=['.txt', '.md'])
        
        # Test included extensions
        assert filter_instance.should_include(Path('test.txt'))
        assert filter_instance.should_include(Path('test.md'))
        
        # Test excluded extensions
        assert not filter_instance.should_include(Path('test.py'))
        assert not filter_instance.should_include(Path('test.log'))
    
    def test_exclude_extensions(self):
        """Test filtering by excluded extensions"""
        filter_instance = ExtensionFilter(exclude_extensions=['.tmp', '.log'])
        
        # Test non-excluded extensions
        assert filter_instance.should_include(Path('test.txt'))
        assert filter_instance.should_include(Path('test.py'))
        
        # Test excluded extensions
        assert not filter_instance.should_include(Path('test.tmp'))
        assert not filter_instance.should_include(Path('test.log'))
    
    def test_case_insensitive(self):
        """Test case insensitive extension matching"""
        filter_instance = ExtensionFilter(include_extensions=['.TXT', '.MD'])
        
        assert filter_instance.should_include(Path('test.txt'))
        assert filter_instance.should_include(Path('test.TXT'))
        assert filter_instance.should_include(Path('test.md'))
        assert filter_instance.should_include(Path('test.MD'))


class TestSizeFilter:
    """Test SizeFilter functionality"""
    
    def test_min_size_filter(self):
        """Test minimum size filtering"""
        filter_instance = SizeFilter.min_size_filter(100)
        
        # Create mock FileInfo objects
        small_file = FileInfo(path="small.txt", size=50, modified_at=datetime.now(), hash_md5="hash1", file_type="text")
        large_file = FileInfo(path="large.txt", size=150, modified_at=datetime.now(), hash_md5="hash2", file_type="text")
        
        assert not filter_instance.should_include(Path("small.txt"), small_file)
        assert filter_instance.should_include(Path("large.txt"), large_file)
    
    def test_max_size_filter(self):
        """Test maximum size filtering"""
        filter_instance = SizeFilter.max_size_filter(100)
        
        # Create mock FileInfo objects
        small_file = FileInfo(path="small.txt", size=50, modified_at=datetime.now(), hash_md5="hash1", file_type="text")
        large_file = FileInfo(path="large.txt", size=150, modified_at=datetime.now(), hash_md5="hash2", file_type="text")
        
        assert filter_instance.should_include(Path("small.txt"), small_file)
        assert not filter_instance.should_include(Path("large.txt"), large_file)
    
    def test_size_range_filter(self):
        """Test size range filtering"""
        filter_instance = SizeFilter.size_range_filter(50, 100)
        
        # Create mock FileInfo objects
        too_small = FileInfo(path="tiny.txt", size=25, modified_at=datetime.now(), hash_md5="hash1", file_type="text")
        just_right = FileInfo(path="medium.txt", size=75, modified_at=datetime.now(), hash_md5="hash2", file_type="text")
        too_large = FileInfo(path="huge.txt", size=150, modified_at=datetime.now(), hash_md5="hash3", file_type="text")
        
        assert not filter_instance.should_include(Path("tiny.txt"), too_small)
        assert filter_instance.should_include(Path("medium.txt"), just_right)
        assert not filter_instance.should_include(Path("huge.txt"), too_large)


class TestFileInfo:
    """Test FileInfo functionality"""
    
    def test_file_info_equality(self):
        """Test FileInfo equality comparison"""
        now = datetime.now()
        
        file1 = FileInfo(path="test.txt", size=100, modified_at=now, hash_md5="abc123", file_type="text")
        file2 = FileInfo(path="test.txt", size=100, modified_at=now, hash_md5="abc123", file_type="text")
        file3 = FileInfo(path="test.txt", size=200, modified_at=now, hash_md5="abc123", file_type="text")
        
        assert file1 == file2
        assert file1 != file3
    
    def test_determine_file_type(self):
        """Test file type determination"""
        assert FileInfo._determine_file_type(Path("test.txt")) == "text"
        assert FileInfo._determine_file_type(Path("test.md")) == "markdown"
        assert FileInfo._determine_file_type(Path("test.py")) == "code"
        assert FileInfo._determine_file_type(Path("test.unknown")) == "unknown"


class TestChangeSet:
    """Test ChangeSet functionality"""
    
    def test_change_set_has_changes(self):
        """Test change detection"""
        # Empty change set
        empty_changes = ChangeSet()
        assert not empty_changes.has_changes()
        
        # Changes present
        changes_with_added = ChangeSet(added=["file1.txt"])
        assert changes_with_added.has_changes()
        
        changes_with_modified = ChangeSet(modified=["file2.txt"])
        assert changes_with_modified.has_changes()
        
        changes_with_deleted = ChangeSet(deleted=["file3.txt"])
        assert changes_with_deleted.has_changes()
    
    def test_change_set_summary(self):
        """Test change set summary"""
        changes = ChangeSet(
            added=["file1.txt", "file2.txt"],
            modified=["file3.txt"],
            deleted=["file4.txt", "file5.txt", "file6.txt"],
            unchanged=["file7.txt"]
        )
        
        summary = changes.get_summary()
        assert summary['added'] == 2
        assert summary['modified'] == 1
        assert summary['deleted'] == 3
        assert summary['unchanged'] == 1
        assert summary['total_changes'] == 6
        assert summary['total_files'] == 7
        assert summary['has_changes'] is True


class TestFileTracker:
    """Test FileTracker functionality"""
    
    def test_file_tracker_initialization(self):
        """Test FileTracker initialization"""
        tracker = FileTracker()
        assert tracker.get_file_count() == 0
        assert tracker.get_tracked_files() == []
    
    def test_file_tracking_operations(self):
        """Test basic file tracking operations"""
        tracker = FileTracker()
        
        # Test is_file_tracked
        assert not tracker.is_file_tracked("nonexistent.txt")
        
        # Test get_file_info
        assert tracker.get_file_info("nonexistent.txt") is None


class MockDocumentStore(DocumentStore):
    """Mock DocumentStore for testing"""
    
    def __init__(self):
        self.documents = {}
        
    def store_document(self, document: Document) -> str:
        self.documents[document.id] = document
        return document.id
    
    def get_document(self, document_id: str):
        return self.documents.get(document_id)
    
    def update_document(self, document: Document) -> bool:
        if document.id in self.documents:
            self.documents[document.id] = document
            return True
        return False
    
    def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False
    
    def search_by_metadata(self, filters, limit=100, offset=0):
        results = []
        for doc in self.documents.values():
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                results.append(SearchResult(document=doc))
        return results[offset:offset+limit]
    
    def search_by_content(self, query: str, limit=100, offset=0):
        results = []
        for doc in self.documents.values():
            if query.lower() in doc.content.lower():
                results.append(SearchResult(document=doc))
        return results[offset:offset+limit]
    
    def list_documents(self, limit=100, offset=0, sort_by="created_at", sort_order="desc"):
        docs = list(self.documents.values())
        return docs[offset:offset+limit]
    
    def count_documents(self, filters=None):
        if filters is None:
            return len(self.documents)
        count = 0
        for doc in self.documents.values():
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            if match:
                count += 1
        return count
    
    def get_storage_stats(self):
        from refinire_rag.storage.document_store import StorageStats
        return StorageStats(
            total_documents=len(self.documents),
            total_chunks=len(self.documents),
            storage_size_bytes=sum(len(doc.content.encode('utf-8')) for doc in self.documents.values()),
            oldest_document=None,
            newest_document=None
        )
    
    def get_documents_by_lineage(self, original_document_id: str):
        # Simple implementation for testing
        return [doc for doc in self.documents.values() 
                if doc.metadata.get('origin_id') == original_document_id]
    
    def cleanup_orphaned_documents(self):
        # Mock implementation - no orphans in our simple store
        return 0
    
    def backup_to_file(self, backup_path: str):
        # Mock implementation
        return True
    
    def restore_from_file(self, backup_path: str):
        # Mock implementation  
        return True


class TestIncrementalDirectoryLoader:
    """Test IncrementalDirectoryLoader functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.mock_store = MockDocumentStore()
        
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_loader_initialization(self):
        """Test loader initialization"""
        loader = IncrementalDirectoryLoader(
            directory_path=self.temp_path,
            document_store=self.mock_store
        )
        
        assert loader.directory_path == self.temp_path
        assert loader.document_store == self.mock_store
        assert loader.recursive is True
        assert loader.additional_metadata == {}
    
    def test_loader_with_filters(self):
        """Test loader with filter configuration"""
        filter_config = FilterConfig(
            extension_filter=ExtensionFilter(include_extensions=['.txt'])
        )
        
        loader = IncrementalDirectoryLoader(
            directory_path=self.temp_path,
            document_store=self.mock_store,
            filter_config=filter_config
        )
        
        assert loader.filter_config == filter_config
    
    def test_loader_invalid_directory(self):
        """Test loader with invalid directory"""
        invalid_path = Path("/nonexistent/directory")
        
        with pytest.raises(FileNotFoundError):
            IncrementalDirectoryLoader(
                directory_path=invalid_path,
                document_store=self.mock_store
            )
    
    def test_get_change_summary(self):
        """Test getting change summary"""
        # Create a test file
        test_file = self.temp_path / "test.txt"
        test_file.write_text("Test content")
        
        loader = IncrementalDirectoryLoader(
            directory_path=self.temp_path,
            document_store=self.mock_store
        )
        
        summary = loader.get_change_summary()
        
        assert 'directory' in summary
        assert 'recursive' in summary
        assert 'tracked_files' in summary
        assert 'changes' in summary
        assert 'filters_configured' in summary
        
        assert summary['directory'] == str(self.temp_path)
        assert summary['recursive'] is True
        assert summary['filters_configured'] is False


if __name__ == "__main__":
    pytest.main([__file__])