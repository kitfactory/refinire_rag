"""
Model configuration utilities for refinire-rag

Provides utilities for getting LLM model names from environment variables
with fallback priorities.
"""

import os
from typing import Optional


def get_default_llm_model(override_model: Optional[str] = None) -> str:
    """Get default LLM model name from environment variables with fallback priority
    
    環境変数から優先順位に従ってデフォルトLLMモデル名を取得
    
    Priority order:
    1. override_model parameter (if provided)
    2. REFINIRE_RAG_LLM_MODEL environment variable
    3. REFINIRE_DEFAULT_LLM_MODEL environment variable  
    4. Default fallback: "gpt-4o-mini"
    
    Args:
        override_model: Optional model name to override environment variables
                       環境変数を上書きするオプションのモデル名
        
    Returns:
        str: LLM model name to use
             使用するLLMモデル名
        
    Example:
        # Use environment variable priority
        model = get_default_llm_model()
        
        # Override with specific model
        model = get_default_llm_model("gpt-4")
        
        # With environment variables set:
        # REFINIRE_RAG_LLM_MODEL=claude-3-opus → returns "claude-3-opus"
        # REFINIRE_DEFAULT_LLM_MODEL=gpt-4 → returns "gpt-4" (if RAG model not set)
        # Neither set → returns "gpt-4o-mini"
    """
    # 1. Use override parameter if provided
    if override_model:
        return override_model
    
    # 2. Check REFINIRE_RAG_LLM_MODEL first
    rag_model = os.getenv("REFINIRE_RAG_LLM_MODEL")
    if rag_model:
        return rag_model
    
    # 3. Check REFINIRE_DEFAULT_LLM_MODEL
    default_model = os.getenv("REFINIRE_DEFAULT_LLM_MODEL")
    if default_model:
        return default_model
    
    # 4. Default fallback
    return "gpt-4o-mini"