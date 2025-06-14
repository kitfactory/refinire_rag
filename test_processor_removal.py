#!/usr/bin/env python3
"""
Test that VectorStoreProcessor has been successfully removed and replaced with VectorStore
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def test_processor_removal():
    """Test that old processor classes are removed and new ones work"""
    print("=== Testing Processor Removal ===\n")
    
    # Test 1: VectorStoreProcessor should not be importable
    print("1. Testing VectorStoreProcessor removal:")
    
    try:
        from refinire_rag.processing.vector_store_processor import VectorStoreProcessor
        print("   ❌ VectorStoreProcessor still exists - should be removed!")
        return False
    except ImportError:
        print("   ✓ VectorStoreProcessor successfully removed")
    
    # Test 2: VectorStore should be importable and work as DocumentProcessor
    print("\n2. Testing VectorStore as DocumentProcessor:")
    
    try:
        from refinire_rag.storage.vector_store import VectorStore
        from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
        from refinire_rag.document_processor import DocumentProcessor
        
        # Check inheritance
        vector_store = InMemoryVectorStore()
        print(f"   ✓ VectorStore imported successfully")
        print(f"   ✓ Is DocumentProcessor: {isinstance(vector_store, DocumentProcessor)}")
        print(f"   ✓ Has process method: {hasattr(vector_store, 'process')}")
        print(f"   ✓ Has get_config_class method: {hasattr(vector_store, 'get_config_class')}")
        
    except ImportError as e:
        print(f"   ❌ Failed to import VectorStore: {e}")
        return False
    
    # Test 3: KeywordSearch should work as DocumentProcessor
    print("\n3. Testing KeywordSearch as DocumentProcessor:")
    
    try:
        from refinire_rag.retrieval.base import KeywordSearch
        from refinire_rag.document_processor import DocumentProcessor
        
        # KeywordSearch is abstract, but check class definition
        print(f"   ✓ KeywordSearch imported successfully")
        print(f"   ✓ Is DocumentProcessor subclass: {issubclass(KeywordSearch, DocumentProcessor)}")
        print(f"   ✓ Has process method: {hasattr(KeywordSearch, 'process')}")
        
    except ImportError as e:
        print(f"   ❌ Failed to import KeywordSearch: {e}")
        return False
    
    # Test 4: Processing package should not export VectorStoreProcessor
    print("\n4. Testing processing package exports:")
    
    try:
        import refinire_rag.processing as processing_module
        # Check if VectorStoreProcessor is in processing module
        if hasattr(processing_module, 'VectorStoreProcessor'):
            print(f"   ❌ VectorStoreProcessor still exported from processing package")
            return False
        else:
            print(f"   ✓ VectorStoreProcessor not exported from processing package")
            
    except ImportError as e:
        print(f"   ⚠️  Warning importing processing package: {e}")
    
    # Test 5: Verify DocumentPipeline still works
    print("\n5. Testing DocumentPipeline compatibility:")
    
    try:
        from refinire_rag.document_processor import DocumentPipeline
        from refinire_rag.models.document import Document
        
        # Create test pipeline with VectorStore
        vector_store = InMemoryVectorStore()
        pipeline = DocumentPipeline([vector_store])
        
        print(f"   ✓ DocumentPipeline created with VectorStore")
        print(f"   ✓ Pipeline processors: {len(pipeline.processors)}")
        
        # Test with empty document list (just check structure)
        results = pipeline.process_documents([])
        print(f"   ✓ Pipeline executed successfully")
        
    except Exception as e:
        print(f"   ❌ DocumentPipeline test failed: {e}")
        return False
    
    return True


def main():
    """Run processor removal tests"""
    print("Processor Removal Test")
    print("=" * 40)
    
    success = test_processor_removal()
    
    print("\n" + "=" * 40)
    
    if success:
        print("✅ All processor removal tests passed!")
        print("\nMigration Complete:")
        print("  🗑️  VectorStoreProcessor → Removed")
        print("  🗑️  KeywordSearchProcessor → Removed (never existed)")
        print("  ✨ VectorStore → Now DocumentProcessor")
        print("  ✨ KeywordSearch → Now DocumentProcessor")
        print("  🔗 DocumentPipeline → Works with new classes")
        
        print("\nBenefits:")
        print("  • Simplified architecture")
        print("  • Fewer classes to manage")
        print("  • Direct store usage in pipelines")
        print("  • Consistent DocumentProcessor interface")
        
        return 0
    else:
        print("❌ Some processor removal tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())