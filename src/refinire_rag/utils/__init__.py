"""
Utility modules for refinire-rag
"""

# Import from the specific utils module to avoid circular imports
import uuid

def generate_document_id() -> str:
    """Generate a unique document ID using UUID"""
    return str(uuid.uuid4())

def generate_chunk_id() -> str:
    """Generate a unique chunk ID using UUID"""
    return str(uuid.uuid4())

__all__ = ["generate_document_id", "generate_chunk_id"]