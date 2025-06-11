"""
Detailed tests for all Loader subclasses
"""

import sys
import tempfile
import json
import csv
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag import (
    UniversalLoader,
    TextLoader,
    MarkdownLoader,
    JSONLoader,
    CSVLoader,
    PathBasedMetadataGenerator,
    LoadingConfig
)


def create_test_files():
    """Create test files for each loader type"""
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Text file
    (test_dir / "sample.txt").write_text(
        "This is a sample text file.\nIt has multiple lines.\nWith some content.",
        encoding="utf-8"
    )
    
    # Markdown file
    (test_dir / "sample.md").write_text(
        "# Sample Markdown\n\nThis is a **markdown** file with *formatting*.\n\n- List item 1\n- List item 2",
        encoding="utf-8"
    )
    
    # JSON file
    sample_data = {
        "title": "Sample JSON",
        "description": "This is test data",
        "items": ["item1", "item2", "item3"],
        "metadata": {
            "version": "1.0",
            "author": "test"
        }
    }
    (test_dir / "sample.json").write_text(
        json.dumps(sample_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    # CSV file
    with open(test_dir / "sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age", "City"])
        writer.writerow(["Alice", "25", "Tokyo"])
        writer.writerow(["Bob", "30", "Osaka"])
        writer.writerow(["Charlie", "35", "Kyoto"])
    
    # HTML file
    (test_dir / "sample.html").write_text(
        """<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML</title>
    <style>body { font-family: Arial; }</style>
</head>
<body>
    <h1>Sample HTML Document</h1>
    <p>This is a paragraph with <strong>bold text</strong>.</p>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
    </ul>
    <script>console.log('This should be removed');</script>
</body>
</html>""",
        encoding="utf-8"
    )
    
    print(f"✓ Created test files in {test_dir}")
    return test_dir


def test_text_loader():
    """Test TextLoader specifically"""
    print("\n=== Testing TextLoader ===")
    
    loader = TextLoader()
    
    # Test supported formats
    formats = loader.supported_formats()
    print(f"✓ Supported formats: {formats}")
    assert ".txt" in formats
    
    # Test loading
    doc = loader.load("test_files/sample.txt")
    print(f"✓ Loaded document ID: {doc.id}")
    print(f"✓ Content length: {len(doc.content)}")
    print(f"✓ File type: {doc.file_type}")
    print(f"✓ Loader used: {doc.metadata.get('loader_used')}")
    
    # Check content
    assert "sample text file" in doc.content
    assert "multiple lines" in doc.content
    assert doc.metadata["loader_used"] == "TextLoader"
    
    print("✓ TextLoader tests passed")


def test_markdown_loader():
    """Test MarkdownLoader specifically"""
    print("\n=== Testing MarkdownLoader ===")
    
    loader = MarkdownLoader()
    
    # Test supported formats
    formats = loader.supported_formats()
    print(f"✓ Supported formats: {formats}")
    assert ".md" in formats
    assert ".markdown" in formats
    
    # Test loading
    doc = loader.load("test_files/sample.md")
    print(f"✓ Loaded document ID: {doc.id}")
    print(f"✓ Content length: {len(doc.content)}")
    print(f"✓ Content preview: {doc.content[:50]}...")
    
    # Check content
    assert "# Sample Markdown" in doc.content
    assert "**markdown**" in doc.content
    assert "- List item" in doc.content
    assert doc.metadata["loader_used"] == "MarkdownLoader"
    
    print("✓ MarkdownLoader tests passed")


def test_json_loader():
    """Test JSONLoader specifically"""
    print("\n=== Testing JSONLoader ===")
    
    loader = JSONLoader()
    
    # Test supported formats
    formats = loader.supported_formats()
    print(f"✓ Supported formats: {formats}")
    assert ".json" in formats
    
    # Test loading
    doc = loader.load("test_files/sample.json")
    print(f"✓ Loaded document ID: {doc.id}")
    print(f"✓ Content length: {len(doc.content)}")
    print(f"✓ Content preview: {doc.content[:100]}...")
    
    # Check content structure
    assert "title: Sample JSON" in doc.content
    assert "description: This is test data" in doc.content
    assert "items[0]: item1" in doc.content
    assert "metadata.version: 1.0" in doc.content
    assert doc.metadata["loader_used"] == "JSONLoader"
    
    print("✓ JSONLoader tests passed")


def test_csv_loader():
    """Test CSVLoader specifically"""
    print("\n=== Testing CSVLoader ===")
    
    loader = CSVLoader()
    
    # Test supported formats
    formats = loader.supported_formats()
    print(f"✓ Supported formats: {formats}")
    assert ".csv" in formats
    
    # Test loading
    doc = loader.load("test_files/sample.csv")
    print(f"✓ Loaded document ID: {doc.id}")
    print(f"✓ Content length: {len(doc.content)}")
    print(f"✓ Content preview: {doc.content}")
    
    # Check content
    lines = doc.content.split('\n')
    assert "Name Age City" in lines[0]
    assert "Alice 25 Tokyo" in lines[1]
    assert "Bob 30 Osaka" in lines[2]
    assert doc.metadata["loader_used"] == "CSVLoader"
    
    print("✓ CSVLoader tests passed")


def test_html_loader():
    """Test HTMLLoader (if dependencies available)"""
    print("\n=== Testing HTMLLoader ===")
    
    try:
        from refinire_rag.loaders.specialized import HTMLLoader
        
        loader = HTMLLoader()
        
        # Test supported formats
        formats = loader.supported_formats()
        print(f"✓ Supported formats: {formats}")
        assert ".html" in formats
        assert ".htm" in formats
        
        # Test loading
        doc = loader.load("test_files/sample.html")
        print(f"✓ Loaded document ID: {doc.id}")
        print(f"✓ Content length: {len(doc.content)}")
        print(f"✓ Content preview: {doc.content[:100]}...")
        
        # Check that HTML tags are removed and scripts are excluded
        assert "Sample HTML Document" in doc.content
        assert "This is a paragraph with bold text" in doc.content
        assert "List item 1" in doc.content
        assert "<h1>" not in doc.content  # HTML tags should be removed
        assert "console.log" not in doc.content  # Script content should be removed
        assert doc.metadata["loader_used"] == "HTMLLoader"
        
        print("✓ HTMLLoader tests passed")
        return True
        
    except (ImportError, Exception) as e:
        if "BeautifulSoup4 required" in str(e) or "bs4" in str(e):
            print("⚠ HTMLLoader skipped (BeautifulSoup4 not available)")
            return False
        else:
            raise


def test_universal_loader_integration():
    """Test UniversalLoader with different file types"""
    print("\n=== Testing UniversalLoader Integration ===")
    
    # Create loader with metadata generation
    path_rules = {
        "*test_files*": {
            "test_dataset": True,
            "category": "test_data"
        }
    }
    metadata_gen = PathBasedMetadataGenerator(path_rules)
    loader = UniversalLoader(metadata_generator=metadata_gen)
    
    # Test all file types
    test_files = [
        ("test_files/sample.txt", "TextLoader"),
        ("test_files/sample.md", "MarkdownLoader"), 
        ("test_files/sample.json", "JSONLoader"),
        ("test_files/sample.csv", "CSVLoader"),
    ]
    
    # Add HTML if available
    try:
        from refinire_rag.loaders.specialized import HTMLLoader
        # Quick test to see if HTML loading actually works
        test_loader = HTMLLoader()
        test_loader.load("test_files/sample.html")
        test_files.append(("test_files/sample.html", "HTMLLoader"))
    except (ImportError, Exception):
        pass  # Skip HTML if dependencies not available
    
    for file_path, expected_loader in test_files:
        print(f"\nTesting {file_path}...")
        
        # Check if loader can handle the file
        can_load = loader.can_load(file_path)
        print(f"✓ Can load: {can_load}")
        assert can_load
        
        # Get loader info
        extension = Path(file_path).suffix
        info = loader.get_loader_info(extension)
        print(f"✓ Loader info: {info}")
        assert info["loader_class"] == expected_loader
        
        # Load the document
        doc = loader.load(file_path)
        print(f"✓ Loaded with {doc.metadata.get('loader_used')}")
        print(f"✓ Universal loader metadata: {doc.metadata.get('loader_type')}")
        print(f"✓ Test dataset flag: {doc.metadata.get('test_dataset')}")
        
        # Verify metadata
        assert doc.metadata["loader_used"] == expected_loader
        assert doc.metadata["loader_type"] == "universal"
        assert doc.metadata["test_dataset"] == True
        assert doc.metadata["category"] == "test_data"
        
        print(f"✓ {file_path} successfully loaded and validated")


def test_batch_loading():
    """Test batch loading with mixed file types"""
    print("\n=== Testing Batch Loading ===")
    
    config = LoadingConfig(
        parallel=True,
        max_workers=2,
        skip_errors=True
    )
    
    loader = UniversalLoader(config=config)
    
    # Prepare file list
    file_paths = [
        "test_files/sample.txt",
        "test_files/sample.md",
        "test_files/sample.json",
        "test_files/sample.csv",
        "non_existent_file.txt"  # This should fail but be skipped
    ]
    
    # Add HTML if available
    try:
        from refinire_rag.loaders.specialized import HTMLLoader
        # Quick test to see if HTML loading actually works
        test_loader = HTMLLoader()
        test_loader.load("test_files/sample.html")
        file_paths.append("test_files/sample.html")
    except (ImportError, Exception):
        pass  # Skip HTML if dependencies not available
    
    # Progress tracking
    progress_updates = []
    def progress_callback(completed: int, total: int):
        progress_updates.append((completed, total))
        print(f"Progress: {completed}/{total}")
    
    # Load batch
    result = loader.load_batch(file_paths, progress_callback=progress_callback)
    
    print(f"\n✓ Batch loading completed: {result.summary()}")
    print(f"✓ Progress updates: {len(progress_updates)}")
    print(f"✓ Successfully loaded: {result.successful_count}")
    print(f"✓ Failed: {result.failed_count}")
    print(f"✓ Failed paths: {result.failed_paths}")
    
    # Verify results
    assert result.successful_count >= 4  # At least txt, md, json, csv
    assert result.failed_count >= 1  # The non-existent file
    assert "non_existent_file.txt" in result.failed_paths
    assert len(progress_updates) > 0
    
    # Check loaded documents
    for doc in result.documents:
        print(f"  - {doc.path}: {doc.metadata.get('loader_used')} ({len(doc.content)} chars)")
        assert len(doc.content) > 0
        assert "loader_used" in doc.metadata
    
    print("✓ Batch loading tests passed")


def test_error_handling():
    """Test error handling scenarios"""
    print("\n=== Testing Error Handling ===")
    
    loader = UniversalLoader()
    
    # Test unsupported file type
    try:
        doc = loader.load("test_files/unsupported.xyz")
        print("✗ Should have failed for unsupported file type")
        assert False
    except Exception as e:
        print(f"✓ Expected error for unsupported file: {type(e).__name__}")
        assert "No loader available" in str(e)
    
    # Test non-existent file
    try:
        doc = loader.load("non_existent_file.txt")
        print("✗ Should have failed for non-existent file")
        assert False
    except Exception as e:
        print(f"✓ Expected error for non-existent file: {type(e).__name__}")
    
    # Test batch loading with skip_errors=False
    config = LoadingConfig(skip_errors=False)
    loader_strict = UniversalLoader(config=config)
    
    file_paths = ["test_files/sample.txt", "non_existent_file.txt"]
    
    try:
        result = loader_strict.load_batch(file_paths)
        print("✗ Should have failed with skip_errors=False")
        assert False
    except Exception as e:
        print(f"✓ Expected error with skip_errors=False: {type(e).__name__}")
    
    print("✓ Error handling tests passed")


def cleanup_test_files():
    """Clean up test files"""
    import shutil
    test_dir = Path("test_files")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print("✓ Cleaned up test files")


def main():
    """Run all detailed loader tests"""
    print("Running detailed Loader subclass tests...\n")
    
    try:
        # Setup
        test_dir = create_test_files()
        
        # Test individual loaders
        test_text_loader()
        test_markdown_loader()
        test_json_loader()
        test_csv_loader()
        html_available = test_html_loader()  # May be skipped if dependencies missing
        
        # Test integration
        test_universal_loader_integration()
        test_batch_loading()
        test_error_handling()
        
        print("\n✅ All detailed loader tests passed!")
        
        # Show summary
        print("\n=== Test Summary ===")
        print("✅ TextLoader - Basic text file loading")
        print("✅ MarkdownLoader - Markdown file processing")
        print("✅ JSONLoader - JSON structure extraction")
        print("✅ CSVLoader - CSV data parsing")
        print("✅ HTMLLoader - HTML content extraction (if BeautifulSoup4 available)")
        print("✅ UniversalLoader - Extension-based delegation")
        print("✅ Batch Loading - Parallel processing with progress tracking")
        print("✅ Error Handling - Graceful failure management")
        print("✅ Metadata Generation - Path-based enrichment")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        cleanup_test_files()
    
    return 0


if __name__ == "__main__":
    exit(main())