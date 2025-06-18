"""
Example setup.py for refinire-rag plugins
refinire-ragプラグイン用のsetup.pyの例

This file shows how to properly configure a plugin package to be
automatically discovered by refinire-rag's unified import system.

このファイルは、refinire-ragの統一インポートシステムによって
自動発見されるようにプラグインパッケージを適切に設定する方法を示します。
"""

from setuptools import setup, find_packages

# Example for a vector store plugin
setup(
    name="refinire-rag-chroma",
    version="1.0.0",
    description="ChromaDB vector store plugin for refinire-rag",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/refinire-rag-chroma",
    
    packages=find_packages(),
    
    # Dependencies
    install_requires=[
        "refinire-rag>=0.1.0",
        "chromadb>=0.4.0",
    ],
    
    # Entry points for automatic discovery
    entry_points={
        # Vector store plugins
        "refinire_rag.vectorstore": [
            "ChromaVectorStore = refinire_rag_chroma:ChromaVectorStore",
            "ChromaDBStore = refinire_rag_chroma:ChromaVectorStore",  # Alias
        ],
        
        # If your plugin provides multiple types:
        # "refinire_rag.keywordstore": [
        #     "ChromaKeywordStore = refinire_rag_chroma:ChromaKeywordStore",
        # ],
        # "refinire_rag.loaders": [
        #     "ChromaLoader = refinire_rag_chroma:ChromaLoader",
        # ],
    },
    
    # Package metadata
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="rag, vector-store, chroma, retrieval, ai",
)

"""
Entry Point Groups for refinire-rag plugins:

1. refinire_rag.vectorstore
   - For vector store implementations
   - Entry point name will be used as the import name
   - Example: ChromaVectorStore = refinire_rag_chroma:ChromaVectorStore

2. refinire_rag.keywordstore  
   - For keyword store implementations
   - Example: BM25Store = refinire_rag_bm25:BM25Store

3. refinire_rag.loaders
   - For document loader implementations
   - Example: DoclingLoader = refinire_rag_docling:DoclingLoader

4. refinire_rag.rerankers
   - For reranker implementations
   - Example: CrossEncoderReranker = refinire_rag_rerankers:CrossEncoderReranker

5. refinire_rag.readers
   - For reader implementations  
   - Example: LlamaReader = refinire_rag_llama:LlamaReader

Usage after installation:
    pip install refinire-rag-chroma
    
    from refinire.rag.vectorstore import ChromaVectorStore
    store = ChromaVectorStore(config)
"""

# Alternative setup.py for a multi-component plugin
MULTI_COMPONENT_SETUP = """
setup(
    name="refinire-rag-comprehensive",
    version="1.0.0",
    # ... other setup parameters ...
    
    entry_points={
        "refinire_rag.vectorstore": [
            "MyVectorStore = refinire_rag_comprehensive.vectorstore:MyVectorStore",
        ],
        "refinire_rag.keywordstore": [
            "MyKeywordStore = refinire_rag_comprehensive.keywordstore:MyKeywordStore",
        ],
        "refinire_rag.loaders": [
            "MyLoader = refinire_rag_comprehensive.loaders:MyLoader",
            "MySpecialLoader = refinire_rag_comprehensive.loaders:SpecialLoader",
        ],
        "refinire_rag.rerankers": [
            "MyReranker = refinire_rag_comprehensive.rerankers:MyReranker",
        ],
        "refinire_rag.readers": [
            "MyReader = refinire_rag_comprehensive.readers:MyReader",
        ],
    },
)
"""

# Minimal setup.py example
MINIMAL_SETUP = """
from setuptools import setup, find_packages

setup(
    name="refinire-rag-myplugin",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["refinire-rag>=0.1.0"],
    
    entry_points={
        "refinire_rag.vectorstore": [
            "MyVectorStore = refinire_rag_myplugin:MyVectorStore",
        ],
    },
)
"""