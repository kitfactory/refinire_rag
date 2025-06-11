"""
Example usage of the Loader system in refinire-rag
"""

from pathlib import Path
import asyncio
from typing import List

from src.refinire_rag import (
    UniversalLoader,
    PathBasedMetadataGenerator,
    LoadingConfig,
    TextLoader,
    MarkdownLoader
)


def create_sample_files():
    """Create sample files for demonstration"""
    # Create directories
    Path("examples/data/docs/public").mkdir(parents=True, exist_ok=True)
    Path("examples/data/docs/internal").mkdir(parents=True, exist_ok=True)
    Path("examples/data/docs/technical").mkdir(parents=True, exist_ok=True)
    
    # Create sample text files
    sample_files = {
        "examples/data/docs/public/readme.txt": "This is a public readme file.\nIt contains general information about the project.",
        "examples/data/docs/internal/guide.md": "# Internal Guide\n\nThis is an internal guide with sensitive information.",
        "examples/data/docs/technical/api_spec.txt": "API Specification\n\nThis document describes the API endpoints and their usage.",
        "examples/data/docs/technical/architecture.md": "# System Architecture\n\n## Overview\nThis document describes the system architecture."
    }
    
    for file_path, content in sample_files.items():
        Path(file_path).write_text(content, encoding="utf-8")
        print(f"Created: {file_path}")


def basic_loader_example():
    """Basic example using TextLoader directly"""
    print("\n=== Basic Loader Example ===")
    
    # Create a simple text loader
    loader = TextLoader()
    
    # Load a single document
    doc = loader.load("examples/data/docs/public/readme.txt")
    
    print(f"Document ID: {doc.id}")
    print(f"Content length: {len(doc.content)}")
    print(f"File type: {doc.file_type}")
    print(f"Path: {doc.path}")
    print(f"Content preview: {doc.content[:100]}...")


def metadata_generator_example():
    """Example using PathBasedMetadataGenerator"""
    print("\n=== Metadata Generator Example ===")
    
    # Define path-based rules
    path_rules = {
        "*/docs/public/*": {
            "access_group": "public",
            "classification": "open",
            "department": "general"
        },
        "*/docs/internal/*": {
            "access_group": "employees", 
            "classification": "internal",
            "department": "company"
        },
        "*/docs/technical/*": {
            "access_group": "engineers",
            "classification": "technical",
            "department": "engineering",
            "tags": ["technical", "engineering"]
        }
    }
    
    # Create metadata generator
    metadata_gen = PathBasedMetadataGenerator(path_rules)
    
    # Create loader with metadata generator
    loader = MarkdownLoader(metadata_generator=metadata_gen)
    
    # Load document
    doc = loader.load("examples/data/docs/technical/architecture.md")
    
    print(f"Document ID: {doc.id}")
    print(f"Generated metadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")


def universal_loader_example():
    """Example using UniversalLoader"""
    print("\n=== Universal Loader Example ===")
    
    # Create metadata generator
    path_rules = {
        "*/docs/public/*": {"access_group": "public"},
        "*/docs/internal/*": {"access_group": "internal"},
        "*/docs/technical/*": {"access_group": "technical"}
    }
    metadata_gen = PathBasedMetadataGenerator(path_rules)
    
    # Create universal loader
    loader = UniversalLoader(metadata_generator=metadata_gen)
    
    print(f"Supported formats: {loader.supported_formats()}")
    print(f"Available loaders: {loader.list_available_loaders()}")
    
    # Load a single document
    doc = loader.load("examples/data/docs/internal/guide.md")
    print(f"\nLoaded document: {doc.path}")
    print(f"Loader used: {doc.metadata.get('loader_used')}")
    print(f"Access group: {doc.metadata.get('access_group')}")


def batch_loading_example():
    """Example of batch loading with progress tracking"""
    print("\n=== Batch Loading Example ===")
    
    # Create configuration for parallel loading
    config = LoadingConfig(
        parallel=True,
        max_workers=2,
        skip_errors=True
    )
    
    # Create loader
    loader = UniversalLoader(config=config)
    
    # Define progress callback
    def progress_callback(completed: int, total: int):
        percentage = (completed / total) * 100
        print(f"Progress: {completed}/{total} ({percentage:.1f}%)")
    
    # Get list of files to load
    file_paths = [
        "examples/data/docs/public/readme.txt",
        "examples/data/docs/internal/guide.md",
        "examples/data/docs/technical/api_spec.txt",
        "examples/data/docs/technical/architecture.md"
    ]
    
    # Validate paths first
    validation = loader.validate_paths(file_paths)
    print(f"Supported files: {len(validation['supported'])}")
    print(f"Unsupported files: {len(validation['unsupported'])}")
    
    # Load documents in batch
    result = loader.load_batch(file_paths, progress_callback=progress_callback)
    
    print(f"\nLoading completed:")
    print(f"  {result.summary()}")
    print(f"  Documents loaded: {len(result.documents)}")
    
    # Show first document
    if result.documents:
        doc = result.documents[0]
        print(f"\nFirst document:")
        print(f"  Path: {doc.path}")
        print(f"  Content length: {len(doc.content)}")
        print(f"  Metadata keys: {list(doc.metadata.keys())}")


async def async_loading_example():
    """Example of async batch loading"""
    print("\n=== Async Loading Example ===")
    
    # Create loader
    loader = UniversalLoader()
    
    # Get file paths
    file_paths = [
        "examples/data/docs/public/readme.txt",
        "examples/data/docs/internal/guide.md",
        "examples/data/docs/technical/api_spec.txt",
        "examples/data/docs/technical/architecture.md"
    ]
    
    # Async loading with progress
    async def async_progress(completed: int, total: int):
        print(f"Async progress: {completed}/{total}")
    
    result = await loader.load_batch_async(file_paths, progress_callback=async_progress)
    
    print(f"Async loading completed: {result.summary()}")


def error_handling_example():
    """Example of error handling"""
    print("\n=== Error Handling Example ===")
    
    # Create loader with error skipping disabled
    config = LoadingConfig(skip_errors=False)
    loader = UniversalLoader(config=config)
    
    # Try to load non-existent file
    try:
        doc = loader.load("non_existent_file.txt")
    except Exception as e:
        print(f"Expected error: {type(e).__name__}: {e}")
    
    # Try with error skipping enabled
    config.skip_errors = True
    loader = UniversalLoader(config=config)
    
    file_paths = [
        "examples/data/docs/public/readme.txt",  # exists
        "non_existent_file.txt",  # doesn't exist
        "examples/data/docs/internal/guide.md"  # exists
    ]
    
    result = loader.load_batch(file_paths)
    print(f"Batch loading with errors: {result.summary()}")
    print(f"Failed paths: {result.failed_paths}")


def main():
    """Run all examples"""
    print("Creating sample files...")
    create_sample_files()
    
    # Run synchronous examples
    basic_loader_example()
    metadata_generator_example()
    universal_loader_example()
    batch_loading_example()
    error_handling_example()
    
    # Run async example
    print("\nRunning async example...")
    asyncio.run(async_loading_example())
    
    print("\n=== All examples completed ===")


if __name__ == "__main__":
    main()