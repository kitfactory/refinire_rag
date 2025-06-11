"""
Simple test script for the Loader implementation
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    UniversalLoader,
    PathBasedMetadataGenerator,
    LoadingConfig,
    Document
)


def test_document_creation():
    """Test Document creation and validation"""
    print("Testing Document creation...")
    
    # Valid document
    metadata = {
        "path": "/test/file.txt",
        "created_at": "2024-01-15T10:30:00Z",
        "file_type": ".txt",
        "size_bytes": 1024
    }
    
    doc = Document(
        id="test_001",
        content="Test content",
        metadata=metadata
    )
    
    print(f"✓ Document created: {doc.id}")
    print(f"✓ Path property: {doc.path}")
    print(f"✓ File type: {doc.file_type}")
    
    # Test missing required field
    try:
        invalid_metadata = {"path": "/test/file.txt"}  # missing required fields
        Document(id="test_002", content="Test", metadata=invalid_metadata)
        print("✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Validation works: {e}")


def test_metadata_generator():
    """Test PathBasedMetadataGenerator"""
    print("\nTesting MetadataGenerator...")
    
    path_rules = {
        "/docs/public/*": {"access_group": "public"},
        "/docs/internal/*": {"access_group": "internal"}
    }
    
    generator = PathBasedMetadataGenerator(path_rules)
    
    required_metadata = {
        "path": "/docs/public/readme.txt",
        "created_at": "2024-01-15T10:30:00Z",
        "file_type": ".txt",
        "size_bytes": 1024
    }
    
    additional = generator.generate_metadata(required_metadata)
    print(f"✓ Generated metadata: {additional}")
    assert additional.get("access_group") == "public"
    assert additional.get("filename") == "readme.txt"


def test_universal_loader():
    """Test UniversalLoader basic functionality"""
    print("\nTesting UniversalLoader...")
    
    loader = UniversalLoader()
    
    # Test supported formats
    formats = loader.supported_formats()
    print(f"✓ Supported formats: {formats}")
    assert '.txt' in formats
    assert '.md' in formats
    
    # Test loader info
    info = loader.get_loader_info('.txt')
    print(f"✓ Loader info for .txt: {info}")
    assert 'loader_class' in info
    
    # Test path validation
    paths = ["file.txt", "file.pdf", "file.xyz"]
    validation = loader.validate_paths(paths)
    print(f"✓ Path validation: {validation}")


def test_loading_config():
    """Test LoadingConfig and LoadingResult"""
    print("\nTesting LoadingConfig...")
    
    config = LoadingConfig(
        parallel=True,
        max_workers=2,
        skip_errors=True
    )
    
    print(f"✓ Config created: parallel={config.parallel}, workers={config.max_workers}")
    
    # Test LoadingResult
    from refinire_rag.models.config import LoadingResult
    
    result = LoadingResult(
        documents=[],
        failed_paths=["file1.txt"],
        errors=[Exception("test error")],
        total_time_seconds=1.5,
        successful_count=2,
        failed_count=1
    )
    
    print(f"✓ Success rate: {result.success_rate:.1f}%")
    print(f"✓ Summary: {result.summary()}")
    assert abs(result.success_rate - 66.7) < 0.1  # 2/(2+1) * 100, with tolerance


def main():
    """Run all tests"""
    print("Running Loader implementation tests...\n")
    
    try:
        test_document_creation()
        test_metadata_generator()
        test_universal_loader()
        test_loading_config()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())