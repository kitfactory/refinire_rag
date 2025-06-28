"""
Plugin factory for creating components from environment variables

Provides unified factory methods for creating both built-in components
and external plugins based on environment variable configuration.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from ..registry.plugin_registry import PluginRegistry

logger = logging.getLogger(__name__)

class PluginFactory:
    """Universal factory for creating plugins from environment variables
    
    環境変数からプラグインを作成する統合ファクトリー
    """
    
    @staticmethod
    def create_plugins_from_env(group: str, env_var: str) -> List[Any]:
        """Create plugins based on environment variable
        
        環境変数に基づいてプラグインを作成
        
        Args:
            group: Plugin group (e.g., 'retrievers', 'evaluators')
            env_var: Environment variable name (e.g., 'REFINIRE_RAG_RETRIEVERS')
        
        Returns:
            List of created plugin instances
        """
        plugins_config = os.getenv(env_var, "").strip()
        if not plugins_config:
            logger.warning(f"{env_var} not set, no {group} will be created")
            return []
        
        plugin_names = [name.strip() for name in plugins_config.split(",")]
        plugins = []
        
        for name in plugin_names:
            if not name:
                continue
            
            try:
                plugin = PluginRegistry.create_plugin(group, name)
                plugins.append(plugin)
                
                # Log whether it's built-in or external
                is_builtin = PluginRegistry.is_builtin(group, name)
                plugin_type = "built-in" if is_builtin else "external"
                logger.info(f"Created {plugin_type} {group} plugin: {name}")
                
            except Exception as e:
                logger.error(f"Failed to create {group} plugin '{name}': {e}")
        
        return plugins
    
    @staticmethod
    def create_retrievers_from_env() -> List[Any]:
        """Create retrievers from REFINIRE_RAG_RETRIEVERS"""
        return PluginFactory.create_plugins_from_env('retrievers', 'REFINIRE_RAG_RETRIEVERS')
    
    @staticmethod
    def create_vector_stores_from_env() -> List[Any]:
        """Create vector stores from REFINIRE_RAG_VECTOR_STORES"""
        return PluginFactory.create_plugins_from_env('vector_stores', 'REFINIRE_RAG_VECTOR_STORES')
    
    @staticmethod
    def create_keyword_stores_from_env() -> List[Any]:
        """Create keyword stores from REFINIRE_RAG_KEYWORD_STORES"""
        return PluginFactory.create_plugins_from_env('keyword_stores', 'REFINIRE_RAG_KEYWORD_STORES')
    
    @staticmethod
    def create_document_stores_from_env() -> List[Any]:
        """Create document stores from REFINIRE_RAG_DOCUMENT_STORES"""
        return PluginFactory.create_plugins_from_env('document_stores', 'REFINIRE_RAG_DOCUMENT_STORES')
    
    @staticmethod
    def create_evaluators_from_env() -> List[Any]:
        """Create evaluators from REFINIRE_RAG_EVALUATORS"""
        return PluginFactory.create_plugins_from_env('evaluators', 'REFINIRE_RAG_EVALUATORS')
    
    @staticmethod
    def create_rerankers_from_env() -> List[Any]:
        """Create rerankers from REFINIRE_RAG_RERANKERS"""
        rerankers = PluginFactory.create_plugins_from_env('rerankers', 'REFINIRE_RAG_RERANKERS')
        return rerankers[0] if rerankers else None  # Usually single reranker
    
    @staticmethod
    def create_embedders_from_env() -> List[Any]:
        """Create embedders from REFINIRE_RAG_EMBEDDERS"""
        return PluginFactory.create_plugins_from_env('embedders', 'REFINIRE_RAG_EMBEDDERS')
    
    @staticmethod
    def create_synthesizers_from_env() -> Any:
        """Create synthesizer from REFINIRE_RAG_SYNTHESIZERS"""
        synthesizers = PluginFactory.create_plugins_from_env('synthesizers', 'REFINIRE_RAG_SYNTHESIZERS')
        return synthesizers[0] if synthesizers else None  # Usually single synthesizer
    
    # QualityLab plugin creation methods
    @staticmethod
    def create_test_suites_from_env() -> Any:
        """Create test suite from REFINIRE_RAG_TEST_SUITES"""
        test_suites = PluginFactory.create_plugins_from_env('test_suites', 'REFINIRE_RAG_TEST_SUITES')
        return test_suites[0] if test_suites else None  # Usually single test suite
    
    @staticmethod
    def create_evaluators_from_env() -> Any:
        """Create evaluator from REFINIRE_RAG_EVALUATORS"""
        evaluators = PluginFactory.create_plugins_from_env('evaluators', 'REFINIRE_RAG_EVALUATORS')
        return evaluators[0] if evaluators else None  # Usually single evaluator
    
    @staticmethod
    def create_contradiction_detectors_from_env() -> Any:
        """Create contradiction detector from REFINIRE_RAG_CONTRADICTION_DETECTORS"""
        detectors = PluginFactory.create_plugins_from_env('contradiction_detectors', 'REFINIRE_RAG_CONTRADICTION_DETECTORS')
        return detectors[0] if detectors else None  # Usually single detector
    
    @staticmethod
    def create_insight_reporters_from_env() -> Any:
        """Create insight reporter from REFINIRE_RAG_INSIGHT_REPORTERS"""
        reporters = PluginFactory.create_plugins_from_env('insight_reporters', 'REFINIRE_RAG_INSIGHT_REPORTERS')
        return reporters[0] if reporters else None  # Usually single reporter
    
    @staticmethod
    def get_available_plugins(group: str) -> Dict[str, Any]:
        """Get information about available plugins for a group"""
        return PluginRegistry.get_all_plugins_info().get(group, {})
    
    @staticmethod
    def list_builtin_components(group: str = None) -> Dict[str, List[str]]:
        """List built-in components by group"""
        return PluginRegistry.list_builtin_components(group)
    
    @staticmethod
    def get_plugin_info(group: str, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plugin
        
        特定のプラグインの詳細情報を取得
        """
        all_info = PluginRegistry.get_all_plugins_info()
        return all_info.get(group, {}).get(name)