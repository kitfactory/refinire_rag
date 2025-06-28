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
    from .heuristic_reranker import HeuristicReranker, HeuristicRerankerConfig  
except ImportError:
    HeuristicReranker = None
    HeuristicRerankerConfig = None

try:
    from .rrf_reranker import RRFReranker, RRFRerankerConfig
except ImportError:
    RRFReranker = None
    RRFRerankerConfig = None

try:
    from .llm_reranker import LLMReranker, LLMRerankerConfig
except ImportError:
    LLMReranker = None
    LLMRerankerConfig = None

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
if HeuristicReranker:
    __all__.extend(["HeuristicReranker", "HeuristicRerankerConfig"])
if RRFReranker:
    __all__.extend(["RRFReranker", "RRFRerankerConfig"])
if LLMReranker:
    __all__.extend(["LLMReranker", "LLMRerankerConfig"])
if SimpleAnswerSynthesizer:
    __all__.extend(["SimpleAnswerSynthesizer", "SimpleAnswerSynthesizerConfig"])