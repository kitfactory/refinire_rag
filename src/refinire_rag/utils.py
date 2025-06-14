"""
Utility functions for refinire_rag
refinire_ragのユーティリティ関数
"""

import uuid

def generate_document_id() -> str:
    """
    Generate a unique document ID using UUID
    UUIDを使用して一意の文書IDを生成
    
    Returns:
        Unique document ID (UUID string)
        一意の文書ID（UUID文字列）
    """
    return str(uuid.uuid4())

def generate_chunk_id() -> str:
    """
    Generate a unique chunk ID using UUID
    UUIDを使用して一意のチャンクIDを生成
    
    Returns:
        Unique chunk ID (UUID string)
        一意のチャンクID（UUID文字列）
    """
    return str(uuid.uuid4())