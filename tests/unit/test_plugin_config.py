"""
Tests for plugin configuration system
プラグイン設定システムのテスト
"""

import pytest
from refinire_rag.plugins.plugin_config import PluginConfig, ConfigManager


class TestPluginConfig:
    """Test cases for PluginConfig class
    PluginConfigクラスのテストケース"""
    
    def test_initialization(self):
        """Test configuration initialization
        設定初期化のテスト"""
        name = "test_plugin"
        config_data = {"param1": "value1", "param2": 42}
        config = PluginConfig(name, config_data)
        
        assert config.name == name
        assert config.config == config_data
    
    def test_empty_config(self):
        """Test initialization with empty config
        空の設定での初期化テスト"""
        name = "empty_plugin"
        config = PluginConfig(name, {})
        
        assert config.name == name
        assert config.config == {}
    
    def test_config_access(self):
        """Test configuration data access
        設定データアクセステスト"""
        name = "access_test"
        config_data = {
            "string_param": "test_value",
            "int_param": 100,
            "bool_param": True,
            "list_param": [1, 2, 3],
            "dict_param": {"nested": "value"}
        }
        config = PluginConfig(name, config_data)
        
        assert config.config["string_param"] == "test_value"
        assert config.config["int_param"] == 100
        assert config.config["bool_param"] is True
        assert config.config["list_param"] == [1, 2, 3]
        assert config.config["dict_param"]["nested"] == "value"


class TestConfigManager:
    """Test cases for ConfigManager class
    ConfigManagerクラスのテストケース"""
    
    def test_initialization(self):
        """Test ConfigManager initialization
        ConfigManager初期化のテスト"""
        manager = ConfigManager()
        assert manager.configs == {}
    
    def test_add_config(self):
        """Test adding configuration
        設定追加のテスト"""
        manager = ConfigManager()
        config = PluginConfig("test_plugin", {"param": "value"})
        
        manager.add_config(config)
        
        assert "test_plugin" in manager.configs
        assert manager.configs["test_plugin"] == config
    
    def test_get_config(self):
        """Test getting configuration
        設定取得のテスト"""
        manager = ConfigManager()
        config = PluginConfig("test_plugin", {"param": "value"})
        manager.add_config(config)
        
        retrieved_config = manager.get_config("test_plugin")
        
        assert retrieved_config == config
        assert retrieved_config.name == "test_plugin"
        assert retrieved_config.config["param"] == "value"
    
    def test_get_nonexistent_config(self):
        """Test getting non-existent configuration
        存在しない設定取得のテスト"""
        manager = ConfigManager()
        
        result = manager.get_config("nonexistent")
        
        assert result is None
    
    def test_remove_config(self):
        """Test removing configuration
        設定削除のテスト"""
        manager = ConfigManager()
        config = PluginConfig("test_plugin", {"param": "value"})
        manager.add_config(config)
        
        # Verify config exists
        assert manager.get_config("test_plugin") is not None
        
        # Remove config
        manager.remove_config("test_plugin")
        
        # Verify config is removed
        assert manager.get_config("test_plugin") is None
    
    def test_remove_nonexistent_config(self):
        """Test removing non-existent configuration
        存在しない設定削除のテスト"""
        manager = ConfigManager()
        
        # Should not raise error
        manager.remove_config("nonexistent")
        
        # Manager should still be empty
        assert manager.configs == {}
    
    def test_multiple_configs(self):
        """Test managing multiple configurations
        複数設定管理のテスト"""
        manager = ConfigManager()
        
        configs = [
            PluginConfig("plugin1", {"type": "vector_store"}),
            PluginConfig("plugin2", {"type": "keyword_store"}),
            PluginConfig("plugin3", {"type": "reranker"})
        ]
        
        # Add all configs
        for config in configs:
            manager.add_config(config)
        
        # Verify all configs exist
        for config in configs:
            retrieved = manager.get_config(config.name)
            assert retrieved is not None
            assert retrieved.name == config.name
            assert retrieved.config == config.config
        
        # Remove one config
        manager.remove_config("plugin2")
        
        # Verify only plugin2 is removed
        assert manager.get_config("plugin1") is not None
        assert manager.get_config("plugin2") is None
        assert manager.get_config("plugin3") is not None


class TestIntegration:
    """Integration tests for plugin configuration system
    プラグイン設定システムの統合テスト"""
    
    def test_realistic_usage_scenario(self):
        """Test realistic usage scenario
        現実的な使用シナリオのテスト"""
        manager = ConfigManager()
        
        # Create configurations for different plugin types
        vector_config = PluginConfig("openai_vector", {
            "type": "vector_store",
            "embedding_model": "text-embedding-3-small",
            "top_k": 10,
            "similarity_threshold": 0.7
        })
        
        keyword_config = PluginConfig("tfidf_keyword", {
            "type": "keyword_store",
            "algorithm": "tfidf",
            "max_features": 10000,
            "min_df": 2
        })
        
        reranker_config = PluginConfig("cross_encoder", {
            "type": "reranker",
            "model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "top_k": 5
        })
        
        # Add all configurations
        manager.add_config(vector_config)
        manager.add_config(keyword_config)
        manager.add_config(reranker_config)
        
        # Verify all configurations are accessible
        assert manager.get_config("openai_vector").config["embedding_model"] == "text-embedding-3-small"
        assert manager.get_config("tfidf_keyword").config["algorithm"] == "tfidf"
        assert manager.get_config("cross_encoder").config["top_k"] == 5
        
        # Update configuration
        updated_vector_config = PluginConfig("openai_vector", {
            "type": "vector_store",
            "embedding_model": "text-embedding-3-large",  # Updated
            "top_k": 15,  # Updated
            "similarity_threshold": 0.8  # Updated
        })
        manager.add_config(updated_vector_config)  # This will replace the old one
        
        # Verify update
        retrieved = manager.get_config("openai_vector")
        assert retrieved.config["embedding_model"] == "text-embedding-3-large"
        assert retrieved.config["top_k"] == 15
        assert retrieved.config["similarity_threshold"] == 0.8
    
    def test_edge_cases(self):
        """Test edge cases and error conditions
        エッジケースとエラー条件のテスト"""
        manager = ConfigManager()
        
        # Empty name
        empty_name_config = PluginConfig("", {"param": "value"})
        manager.add_config(empty_name_config)
        assert manager.get_config("") == empty_name_config
        
        # None in config values
        none_config = PluginConfig("none_test", {
            "string_param": None,
            "normal_param": "value"
        })
        manager.add_config(none_config)
        retrieved = manager.get_config("none_test")
        assert retrieved.config["string_param"] is None
        assert retrieved.config["normal_param"] == "value"
        
        # Very large config
        large_config_data = {f"param_{i}": f"value_{i}" for i in range(1000)}
        large_config = PluginConfig("large_config", large_config_data)
        manager.add_config(large_config)
        retrieved = manager.get_config("large_config")
        assert len(retrieved.config) == 1000
        assert retrieved.config["param_500"] == "value_500"