"""
Basic DocumentProcessor Example

This example demonstrates how to use the DocumentProcessor system with
the new configuration pattern where each processor defines its own config class.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Type, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag import (
    Document,
    DocumentProcessor,
    DocumentProcessorConfig,
    DocumentPipeline,
    TokenBasedChunker,
    ChunkingConfig,
    SQLiteDocumentStore
)


def example_basic_usage():
    """Basic usage of DocumentProcessor with default configuration"""
    print("=== Basic DocumentProcessor Usage ===\n")
    
    # Create a simple document
    document = Document(
        id="example_001",
        content="This is a sample document for demonstrating the DocumentProcessor system. "
                "It will be processed through various stages to show the capabilities.",
        metadata={
            "path": "/examples/sample.txt",
            "created_at": "2024-01-15T10:00:00Z",
            "file_type": ".txt",
            "size_bytes": 150
        }
    )
    
    print(f"Original document: {document.id}")
    print(f"Content: {document.content}")
    print(f"Metadata: {document.metadata}")
    
    # Use TokenBasedChunker with default config
    print(f"\n1. Using TokenBasedChunker with default configuration:")
    chunker = TokenBasedChunker()
    
    # Get processor information
    info = chunker.get_processor_info()
    print(f"   Processor: {info['processor_class']}")
    print(f"   Config class: {info['config_class']}")
    print(f"   Default config: {info['config']}")
    
    # Process document
    chunks = chunker.process_with_stats(document)
    print(f"\n   Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"     Chunk {i}: {chunk.id}")
        print(f"       Tokens: {chunk.metadata.get('token_count')}")
        print(f"       Content: {chunk.content[:50]}...")
    
    # Show processing stats
    stats = chunker.get_processing_stats()
    print(f"\n   Processing stats: {stats}")
    
    return chunks


def example_custom_configuration():
    """Using custom configuration for processors"""
    print("\n=== Custom Configuration Example ===\n")
    
    # Create document
    document = Document(
        id="custom_example_001",
        content="This document will be processed with custom configuration settings. "
                "We can control chunk size, overlap, and other parameters precisely. "
                "This demonstrates the flexibility of the configuration system.",
        metadata={
            "path": "/examples/custom.txt",
            "created_at": "2024-01-15T11:00:00Z",
            "file_type": ".txt",
            "size_bytes": 200
        }
    )
    
    # Create custom chunking configuration
    custom_config = ChunkingConfig(
        chunk_size=20,        # Smaller chunks
        overlap=8,            # More overlap
        split_by_sentence=True,
        min_chunk_size=5,     # Allow smaller chunks
        preserve_metadata=True,
        add_processing_info=True
    )
    
    print(f"Custom configuration: {custom_config}")
    
    # Create chunker with custom config
    chunker = TokenBasedChunker(custom_config)
    
    # Process document
    chunks = chunker.process_with_stats(document)
    
    print(f"\nProcessed with custom config - created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        position = chunk.metadata.get('chunk_position', '?')
        total = chunk.metadata.get('chunk_total', '?')
        overlap = chunk.metadata.get('overlap_previous', 0)
        print(f"  Chunk {position}/{total-1 if isinstance(total, int) else total}: "
              f"{chunk.metadata.get('token_count')} tokens, overlap={overlap}")
        print(f"    {chunk.content}")
    
    return chunks


def example_config_serialization():
    """Demonstrate configuration serialization and deserialization"""
    print("\n=== Configuration Serialization Example ===\n")
    
    # Create configuration
    config = ChunkingConfig(
        chunk_size=100,
        overlap=25,
        split_by_sentence=True,
        preserve_formatting=False,
        min_chunk_size=20,
        max_chunk_size=200
    )
    
    print(f"1. Original config: {config}")
    
    # Serialize to dictionary
    config_dict = config.to_dict()
    print(f"\n2. Serialized to dict:")
    for key, value in config_dict.items():
        print(f"     {key}: {value}")
    
    # Deserialize from dictionary
    restored_config = ChunkingConfig.from_dict(config_dict)
    print(f"\n3. Restored config: {restored_config}")
    
    # Verify they match
    print(f"\n4. Configs match: {config == restored_config}")
    
    # Test with extra fields (should be ignored)
    config_with_extra = {
        **config_dict,
        "extra_field": "ignored",
        "unknown_setting": 42
    }
    
    safe_config = ChunkingConfig.from_dict(config_with_extra)
    print(f"\n5. Config from dict with extra fields: {safe_config}")
    print(f"   Still matches original: {safe_config == config}")


def example_dynamic_config_discovery():
    """Demonstrate dynamic discovery of processor configurations"""
    print("\n=== Dynamic Config Discovery Example ===\n")
    
    # List of processor classes to examine
    processor_classes = [TokenBasedChunker]
    
    print("Discovered processor configurations:")
    
    for processor_class in processor_classes:
        # Get config class dynamically
        config_class = processor_class.get_config_class()
        default_config = processor_class.get_default_config()
        
        print(f"\n{processor_class.__name__}:")
        print(f"  Config class: {config_class.__name__}")
        print(f"  Configuration fields:")
        
        # Show all fields with their types and default values
        for field_name, field_info in config_class.__dataclass_fields__.items():
            default_value = getattr(default_config, field_name)
            field_type = field_info.type
            
            # Handle generic types
            type_name = getattr(field_type, '__name__', str(field_type))
            
            print(f"    - {field_name}: {type_name} = {default_value}")


def example_error_handling():
    """Demonstrate error handling in processors"""
    print("\n=== Error Handling Example ===\n")
    
    # Create a problematic document (empty content)
    problematic_doc = Document(
        id="problematic_001",
        content="",  # Empty content might cause issues
        metadata={
            "path": "/examples/empty.txt",
            "created_at": "2024-01-15T12:00:00Z",
            "file_type": ".txt",
            "size_bytes": 0
        }
    )
    
    # Test with fail_on_error=False (default)
    print("1. Testing with fail_on_error=False (default):")
    chunker = TokenBasedChunker()
    
    try:
        chunks = chunker.process_with_stats(problematic_doc)
        print(f"   Result: {len(chunks)} chunks")
        
        stats = chunker.get_processing_stats()
        if stats['errors'] > 0:
            print(f"   Errors handled gracefully: {stats['errors']}")
        else:
            print(f"   No errors occurred")
            
    except Exception as e:
        print(f"   Unexpected error: {e}")
    
    # Test with fail_on_error=True
    print("\n2. Testing with fail_on_error=True:")
    strict_config = ChunkingConfig(fail_on_error=True)
    strict_chunker = TokenBasedChunker(strict_config)
    
    try:
        chunks = strict_chunker.process_with_stats(problematic_doc)
        print(f"   Result: {len(chunks)} chunks (unexpected success)")
    except Exception as e:
        print(f"   ✓ Correctly raised exception: {type(e).__name__}: {e}")


def main():
    """Run all DocumentProcessor examples"""
    print("DocumentProcessor System Examples")
    print("=" * 50)
    
    try:
        # Run examples
        basic_chunks = example_basic_usage()
        custom_chunks = example_custom_configuration()
        example_config_serialization()
        example_dynamic_config_discovery()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
        print(f"\nExample Summary:")
        print(f"  ✅ Basic usage: {len(basic_chunks)} chunks created")
        print(f"  ✅ Custom configuration: {len(custom_chunks)} chunks created")
        print(f"  ✅ Config serialization")
        print(f"  ✅ Dynamic config discovery")
        print(f"  ✅ Error handling")
        
        print(f"\nNext steps:")
        print(f"  - Try custom_processor_example.py for creating your own processors")
        print(f"  - Try pipeline_example.py for complex processing pipelines")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())