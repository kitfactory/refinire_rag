"""
Test script for PluginRegistry functionality

This script demonstrates how built-in components and external plugins
are discovered and registered through the unified registry system.
"""

import logging
from .plugin_registry import PluginRegistry

# Set up logging to see the registration process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_builtin_components():
    """Test built-in component discovery and registration"""
    print("=== Testing Built-in Components ===")
    
    # Test retrievers
    print("\n--- Retrievers ---")
    retrievers = PluginRegistry.list_available_plugins('retrievers')
    print(f"Available retrievers: {retrievers}")
    
    # Test if we can create a built-in retriever
    if 'simple' in retrievers:
        try:
            simple_retriever = PluginRegistry.create_plugin('retrievers', 'simple')
            print(f"Created SimpleRetriever: {type(simple_retriever).__name__}")
        except Exception as e:
            print(f"Failed to create SimpleRetriever: {e}")
    
    # Test vector stores
    print("\n--- Vector Stores ---")
    vector_stores = PluginRegistry.list_available_plugins('vector_stores')
    print(f"Available vector stores: {vector_stores}")
    
    if 'inmemory_vector' in vector_stores:
        try:
            inmemory_vs = PluginRegistry.create_plugin('vector_stores', 'inmemory_vector')
            print(f"Created InMemoryVectorStore: {type(inmemory_vs).__name__}")
        except Exception as e:
            print(f"Failed to create InMemoryVectorStore: {e}")
    
    # Test document stores
    print("\n--- Document Stores ---")
    doc_stores = PluginRegistry.list_available_plugins('document_stores')
    print(f"Available document stores: {doc_stores}")
    
    if 'sqlite' in doc_stores:
        try:
            sqlite_store = PluginRegistry.create_plugin('document_stores', 'sqlite')
            print(f"Created SQLiteStore: {type(sqlite_store).__name__}")
        except Exception as e:
            print(f"Failed to create SQLiteStore: {e}")

def test_plugin_info():
    """Test plugin information retrieval"""
    print("\n=== Testing Plugin Information ===")
    
    # Get all plugin info
    all_info = PluginRegistry.get_all_plugins_info()
    
    for group, plugins in all_info.items():
        if plugins:  # Only show groups that have plugins
            print(f"\n--- {group.title()} ---")
            for name, info in plugins.items():
                builtin_flag = "(built-in)" if info['builtin'] else "(external)"
                print(f"  {name}: {info['class']} {builtin_flag}")

def test_builtin_listing():
    """Test built-in component listing"""
    print("\n=== Testing Built-in Component Listing ===")
    
    builtin_components = PluginRegistry.list_builtin_components()
    for group, components in builtin_components.items():
        if components:
            print(f"{group}: {components}")

if __name__ == "__main__":
    try:
        test_builtin_components()
        test_plugin_info()
        test_builtin_listing()
        print("\n=== Test completed successfully ===")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()