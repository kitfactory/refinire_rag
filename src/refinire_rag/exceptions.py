"""
Exception classes for refinire-rag
"""


class RefinireRAGError(Exception):
    """Base exception for refinire-rag"""
    pass


class LoaderError(RefinireRAGError):
    """Error in document loading"""
    pass


class EmbeddingError(RefinireRAGError):
    """Error in embedding generation"""
    pass


class StorageError(RefinireRAGError):
    """Error in vector storage operations"""
    pass


class MetadataError(RefinireRAGError):
    """Error in metadata generation"""
    pass


class ValidationError(RefinireRAGError):
    """Error in data validation"""
    pass