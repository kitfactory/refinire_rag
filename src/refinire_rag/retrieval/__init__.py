# Retrieval components package

from .base import QueryComponent, Retriever, Reranker, AnswerSynthesizer, Indexer, KeywordSearch
from .base import QueryResult, SearchResult
from .base import RetrieverConfig, RerankerConfig, AnswerSynthesizerConfig

# Simple implementations (import only what exists)
try:
    from .simple_retriever import SimpleRetriever, SimpleRetrieverConfig
except ImportError:
    SimpleRetriever = None
    SimpleRetrieverConfig = None

try:
    from .simple_reranker import SimpleReranker, SimpleRerankerConfig  
except ImportError:
    SimpleReranker = None
    SimpleRerankerConfig = None

try:
    from .simple_answer_synthesizer import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
except ImportError:
    SimpleAnswerSynthesizer = None
    SimpleAnswerSynthesizerConfig = None

__all__ = [
    # Base classes
    "QueryComponent", "Retriever", "Reranker", "AnswerSynthesizer", "Indexer", "KeywordSearch",
    "QueryResult", "SearchResult",
    "RetrieverConfig", "RerankerConfig", "AnswerSynthesizerConfig",
]

# Add simple implementations if they exist
if SimpleRetriever:
    __all__.extend(["SimpleRetriever", "SimpleRetrieverConfig"])
if SimpleReranker:
    __all__.extend(["SimpleReranker", "SimpleRerankerConfig"])
if SimpleAnswerSynthesizer:
    __all__.extend(["SimpleAnswerSynthesizer", "SimpleAnswerSynthesizerConfig"])