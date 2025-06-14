# Tutorial 1: Basic RAG Pipeline

In this tutorial, we'll build the simplest RAG system using refinire-rag.

## Learning Objectives

- Understand the basic concepts of RAG
- Build a simple document corpus
- Execute basic query processing and answer generation

## Basic Components of RAG

A RAG (Retrieval-Augmented Generation) system consists of the following components:

```
Documents → [Embedding] → Vector Store
                           ↓
Query → [Embedding] → [Search] → [Re-ranking] → [Answer Generation] → Answer
```

## Step 1: Basic Setup

First, import the necessary modules:

```python
from refinire_rag.application.corpus_manager import CorpusManager
from refinire_rag.application.query_engine import QueryEngine
from refinire_rag.storage.sqlite_store import SQLiteDocumentStore
from refinire_rag.storage.in_memory_vector_store import InMemoryVectorStore
from refinire_rag.retrieval import SimpleReranker, SimpleAnswerSynthesizer
from refinire_rag.embedding import OpenAIEmbedder, OpenAIEmbeddingConfig
from refinire_rag.models.document import Document
```

## Step 2: Creating Sample Documents

Create sample documents for the RAG system:

```python
def create_sample_documents():
    """Create sample documents"""
    
    documents = [
        Document(
            id="doc1",
            content="""
            RAG (Retrieval-Augmented Generation) is a retrieval-augmented generation technology.
            It combines large language models (LLMs) with external knowledge bases
            to generate more accurate and evidence-based answers.
            Main advantages include reduced hallucination, easy knowledge updates,
            and adaptability to specialized domains.
            """,
            metadata={"title": "RAG Overview", "category": "Technology"}
        ),
        
        Document(
            id="doc2",
            content="""
            Vector search is a search technology based on semantic similarity.
            It embeds documents and queries in high-dimensional vector space
            and calculates relevance using cosine similarity.
            It can find contextually related information
            that traditional keyword search might miss.
            """,
            metadata={"title": "Vector Search", "category": "Technology"}
        ),
        
        Document(
            id="doc3",
            content="""
            Large Language Models (LLMs) are core technologies in natural language processing.
            Advanced models like GPT, Claude, and Gemini exist,
            capable of handling various tasks including text generation,
            translation, summarization, and question answering.
            Enterprises use them for customer support, content generation,
            and document analysis.
            """,
            metadata={"title": "Large Language Models", "category": "Technology"}
        )
    ]
    
    return documents
```

## Step 3: Initializing Storage

Initialize the document store and vector store:

```python
def setup_storage():
    """Initialize storage"""
    
    # Document store (saves metadata and original text)
    document_store = SQLiteDocumentStore(":memory:")
    
    # Vector store (saves embedding vectors)
    vector_store = InMemoryVectorStore()
    
    return document_store, vector_store
```

## Step 4: Building a Simple Corpus

Build a corpus with the simplest RAG pipeline (Load → Chunk → Vector):

```python
def build_simple_corpus(documents, document_store, vector_store):
    """Build a simple corpus"""
    
    print("📚 Building corpus...")
    
    # Create Simple RAG manager
    corpus_manager = CorpusManager.create_simple_rag(
        document_store, 
        vector_store
    )
    
    # Manually add documents to store (instead of actual file paths)
    for doc in documents:
        document_store.store_document(doc)
    
    # Configure embedder and set to vector store (統合アーキテクチャ)
    embedder_config = OpenAIEmbeddingConfig(model="text-embedding-ada-002")
    embedder = OpenAIEmbedder(config=embedder_config)
    vector_store.set_embedder(embedder)  # VectorStoreに直接設定
    
    # Index documents using VectorStore integrated functionality
    for doc in documents:
        vector_store.index_document(doc)  # 統合されたインデックス機能を使用
    
    print(f"✅ Built corpus with {len(documents)} documents")
    return embedder
```

## Step 5: Creating a Query Engine

Create a query engine for search and answer generation:

```python
def create_query_engine(document_store, vector_store, embedder):
    """Create query engine"""
    
    print("🤖 Creating query engine...")
    
    # Create search and answer generation components (統合アーキテクチャ)
    # vector_store is already a Retriever, so use it directly
    reranker = SimpleReranker()
    reader = SimpleAnswerSynthesizer()
    
    # Create query engine - vector_store serves as both storage and retriever
    query_engine = QueryEngine(
        document_store=document_store,
        vector_store=vector_store,
        retriever=vector_store,  # VectorStore自体がRetrieverインターフェースを実装
        reader=reader,
        reranker=reranker
    )
    
    print("✅ Query engine created")
    return query_engine
```

## Step 6: Testing Question Answering

Test some questions with the created RAG system:

```python
def test_questions(query_engine):
    """Test questions"""
    
    questions = [
        "What is RAG?",
        "How does vector search work?",
        "What are the main uses of LLMs?",
        "Explain the advantages of RAG"
    ]
    
    print("\n" + "="*60)
    print("🔍 Question Answering Test")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📌 Question {i}: {question}")
        print("-" * 40)
        
        try:
            result = query_engine.answer(question)
            
            print(f"🤖 Answer:")
            print(f"   {result.answer}")
            
            print(f"\n📊 Details:")
            print(f"   - Processing time: {result.metadata.get('processing_time', 0):.3f} seconds")
            print(f"   - Number of sources: {result.metadata.get('source_count', 0)}")
            print(f"   - Confidence: {result.confidence:.3f}")
            
            if result.sources:
                print(f"   - Main source: {result.sources[0].metadata.get('title', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
```

## Step 7: Complete Sample Program

Here's the complete sample program:

```python
#!/usr/bin/env python3
"""
Tutorial 1: Basic RAG Pipeline
"""

def main():
    """Main function"""
    
    print("🚀 Basic RAG Pipeline Tutorial")
    print("="*60)
    
    try:
        # Step 1: Create sample documents
        documents = create_sample_documents()
        print(f"📝 Created {len(documents)} sample documents")
        
        # Step 2: Initialize storage
        document_store, vector_store = setup_storage()
        print("💾 Storage initialized")
        
        # Step 3: Build corpus
        embedder = build_simple_corpus(documents, document_store, vector_store)
        
        # Step 4: Create query engine
        query_engine = create_query_engine(document_store, vector_store, embedder)
        
        # Step 5: Test question answering
        test_questions(query_engine)
        
        print("\n🎉 Tutorial 1 completed!")
        print("\nNext: [Tutorial 2: Corpus Management and Document Processing]")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
```

## Running the Tutorial

To run this sample program:

```bash
# Navigate to tutorials directory
cd tutorials

# Run the program
python tutorial_01_basic_rag.py
```

## Expected Output

When you run the program, you should see output like this:

```
🚀 Basic RAG Pipeline Tutorial
============================================================
📝 Created 3 sample documents
💾 Storage initialized
📚 Building corpus...
✅ Built corpus with 3 documents
🤖 Creating query engine...
✅ Query engine created

============================================================
🔍 Question Answering Test
============================================================

📌 Question 1: What is RAG?
----------------------------------------
🤖 Answer:
   RAG (Retrieval-Augmented Generation) is a retrieval-augmented generation technology.
   It combines large language models with external knowledge bases to generate more accurate and evidence-based answers.

📊 Details:
   - Processing time: 0.002 seconds
   - Number of sources: 3
   - Confidence: 0.250
   - Main source: RAG Overview

...
```

## Understanding Check

Let's review what you learned in this tutorial:

1. **What are the basic components of RAG?**
   - Document store, vector store, search, answer generation

2. **What is the processing order of the Simple RAG pipeline?**
   - Load → Chunk → Vector

3. **What is the main role of QueryEngine?**
   - Receives queries and manages integrated search and answer generation

## Next Steps

Now that you understand the basic RAG pipeline, proceed to [Tutorial 2: Corpus Management and Document Processing](tutorial_02_corpus_management.md) to learn about more advanced document processing features.

## Troubleshooting

### Common Issues

1. **ImportError**: Module not found
   ```bash
   pip install -e .
   ```

2. **TF-IDF Error**: Corpus too small
   ```python
   # Set min_df=1
   embedder_config = TFIDFEmbeddingConfig(min_df=1, max_df=1.0)
   ```

3. **Memory Error**: Large documents
   ```python
   # Reduce chunk size
   chunk_config = ChunkingConfig(chunk_size=200)
   ```