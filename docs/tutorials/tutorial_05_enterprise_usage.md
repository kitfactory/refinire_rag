# Tutorial 5: Enterprise Department-based RAG System

## Overview

This tutorial demonstrates how to implement department-specific RAG systems in enterprise environments. We'll create separate RAG instances for HR and Sales departments, ensuring data isolation and department-specific knowledge access.

## Scenario

**Company**: TechCorp Inc.
- **HR Department**: Needs access to employee policies, benefits, training materials
- **Sales Department**: Needs access to product information, sales processes, customer data

**Requirements**:
- Data isolation between departments
- Department-specific document access
- Independent RAG systems per department
- Shared infrastructure but isolated data

## System Architecture

```
TechCorp RAG System
├── HR Department RAG
│   ├── HR Document Store
│   ├── HR Vector Store  
│   └── HR Query Engine
└── Sales Department RAG
    ├── Sales Document Store
    ├── Sales Vector Store
    └── Sales Query Engine
```

## Implementation Steps

### Step 1: Setup Department-specific Document Stores

Each department gets its own isolated document store and vector store:

```python
# HR Department Setup
hr_document_store = SQLiteDocumentStore("data/hr/hr_documents.db")
hr_vector_store = InMemoryVectorStore()

# Sales Department Setup  
sales_document_store = SQLiteDocumentStore("data/sales/sales_documents.db")
sales_vector_store = InMemoryVectorStore()
```

### Step 2: Configure Department-specific Corpus Managers

```python
# HR Corpus Manager
hr_config = CorpusManagerConfig(
    enable_processing=True,
    enable_chunking=True,
    enable_embedding=True,
    document_store=hr_document_store,
    vector_store=hr_vector_store
)
hr_corpus_manager = CorpusManager(config=hr_config)

# Sales Corpus Manager
sales_config = CorpusManagerConfig(
    enable_processing=True,
    enable_chunking=True, 
    enable_embedding=True,
    document_store=sales_document_store,
    vector_store=sales_vector_store
)
sales_corpus_manager = CorpusManager(config=sales_config)
```

### Step 3: Setup Department-specific Query Engines

```python
# HR Query Engine
hr_retriever = SimpleRetriever(vector_store=hr_vector_store, embedder=hr_embedder)
hr_query_engine = QueryEngine(
    document_store=hr_document_store,
    vector_store=hr_vector_store,
    retriever=hr_retriever,
    reader=SimpleReader(),
    reranker=SimpleReranker()
)

# Sales Query Engine
sales_retriever = SimpleRetriever(vector_store=sales_vector_store, embedder=sales_embedder)
sales_query_engine = QueryEngine(
    document_store=sales_document_store,
    vector_store=sales_vector_store,
    retriever=sales_retriever,
    reader=SimpleReader(),
    reranker=SimpleReranker()
)
```

## Usage Examples

### HR Department Queries

```python
# Employee asking about vacation policy
hr_response = hr_query_engine.answer("What is our vacation policy?")
print(f"HR Response: {hr_response.answer}")

# Manager asking about performance review process
hr_response = hr_query_engine.answer("How do I conduct performance reviews?")
print(f"HR Response: {hr_response.answer}")
```

### Sales Department Queries  

```python
# Sales rep asking about product features
sales_response = sales_query_engine.answer("What are the key features of our enterprise product?")
print(f"Sales Response: {sales_response.answer}")

# Sales manager asking about pricing strategy
sales_response = sales_query_engine.answer("What is our pricing strategy for new customers?")
print(f"Sales Response: {sales_response.answer}")
```

## Data Isolation Verification

```python
# Verify HR cannot access Sales data
hr_response = hr_query_engine.answer("What is our product pricing?")
# Should return: "No relevant information found"

# Verify Sales cannot access HR data  
sales_response = sales_query_engine.answer("What is our vacation policy?")
# Should return: "No relevant information found"
```

## Quality Monitoring per Department

```python
# Setup department-specific quality monitoring
hr_quality_lab = QualityLab(
    test_suite=TestSuite(),
    evaluator=Evaluator(),
    contradiction_detector=ContradictionDetector(),
    insight_reporter=InsightReporter()
)

sales_quality_lab = QualityLab(
    test_suite=TestSuite(),
    evaluator=Evaluator(), 
    contradiction_detector=ContradictionDetector(),
    insight_reporter=InsightReporter()
)
```

## Best Practices

### 1. Access Control
```python
class DepartmentRAGManager:
    def __init__(self, department: str, user_department: str):
        self.department = department
        self.user_department = user_department
        
    def query(self, question: str):
        if self.user_department != self.department:
            raise PermissionError(f"User from {self.user_department} cannot access {self.department} data")
        return self.query_engine.answer(question)
```

### 2. Audit Logging
```python
def log_query(user_id: str, department: str, query: str, response: str):
    logging.info({
        "timestamp": datetime.now(),
        "user_id": user_id,
        "department": department,
        "query": query,
        "response_confidence": response.confidence,
        "sources_used": len(response.sources)
    })
```

### 3. Regular Quality Checks
```python
# Weekly quality assessment per department
def weekly_quality_check(department_rag: QueryEngine, department_name: str):
    test_queries = load_department_test_queries(department_name)
    results = []
    
    for query in test_queries:
        result = department_rag.answer(query)
        results.append(result)
    
    # Generate quality report
    quality_report = generate_quality_report(results, department_name)
    send_to_department_admin(quality_report, department_name)
```

## Scaling Considerations

### Multi-tenant Architecture
```python
class EnterpriseRAGSystem:
    def __init__(self):
        self.departments = {}
    
    def add_department(self, dept_name: str, config: CorpusManagerConfig):
        self.departments[dept_name] = {
            'corpus_manager': CorpusManager(config),
            'query_engine': self._create_query_engine(config),
            'quality_lab': self._create_quality_lab()
        }
    
    def get_department_rag(self, dept_name: str):
        return self.departments.get(dept_name)
```

### Resource Management
```python
# Shared embedder for efficiency, isolated data stores
shared_embedder = TFIDFEmbedder()

hr_config = CorpusManagerConfig(embedder=shared_embedder, ...)
sales_config = CorpusManagerConfig(embedder=shared_embedder, ...)
```

## Security Considerations

1. **Data Encryption**: Encrypt document stores at rest
2. **Access Logs**: Maintain detailed access logs per department
3. **Network Isolation**: Use VPNs or network segments for sensitive departments
4. **Regular Audits**: Perform regular security audits of department data access

## Monitoring & Alerting

```python
# Department-specific monitoring
def monitor_department_usage(dept_name: str):
    rag_system = get_department_rag(dept_name)
    
    # Monitor query volume
    daily_queries = get_daily_query_count(dept_name)
    if daily_queries > threshold:
        alert_department_admin(f"High query volume in {dept_name}")
    
    # Monitor response quality
    avg_confidence = get_average_confidence(dept_name)
    if avg_confidence < 0.5:
        alert_department_admin(f"Low response quality in {dept_name}")
```

## Next Steps

- **Tutorial 6**: Advanced Multi-modal RAG (Images, Documents, Video)
- **Tutorial 7**: Real-time Learning and Adaptation
- **Tutorial 8**: Performance Optimization and Scaling
- **Tutorial 9**: Production Deployment and DevOps

This enterprise setup ensures that each department maintains data privacy while benefiting from shared infrastructure and consistent RAG capabilities.