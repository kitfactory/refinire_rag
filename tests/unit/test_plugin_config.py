"""
Tests for unified plugin configuration system
統一プラグイン設定システムのテスト
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from refinire_rag.plugins.plugin_config import PluginConfig, ConfigManager


class TestPluginConfig:
    """Test cases for PluginConfig class
    PluginConfigクラスのテストケース"""
    
    def test_default_initialization(self):
        """Test default configuration initialization
        デフォルト設定初期化のテスト"""
        config = PluginConfig()
        
        assert config.plugin_name == "default"
        assert config.plugin_type == "unknown"
        assert config.top_k == 10
        assert config.similarity_threshold == 0.0
        assert config.embedding_model == "text-embedding-3-small"
        assert config.llm_model == "gpt-4o-mini"
        assert config.custom_settings == {}
    
    def test_custom_initialization(self):
        """Test custom configuration initialization
        カスタム設定初期化のテスト"""
        config = PluginConfig(
            plugin_name="test_plugin",
            plugin_type="vector_store",
            top_k=20,
            embedding_model="custom-model"
        )
        
        assert config.plugin_name == "test_plugin"
        assert config.plugin_type == "vector_store"
        assert config.top_k == 20
        assert config.embedding_model == "custom-model"
    
    def test_get_set_methods(self):
        """Test get/set methods for configuration values
        設定値のget/setメソッドのテスト"""
        config = PluginConfig()
        
        # Test standard attributes
        assert config.get("top_k") == 10
        config.set("top_k", 15)
        assert config.get("top_k") == 15
        
        # Test custom settings
        assert config.get("custom_param") is None
        config.set("custom_param", "custom_value")
        assert config.get("custom_param") == "custom_value"
        assert config.custom_settings["custom_param"] == "custom_value"
    
    def test_update_method(self):
        """Test update method for multiple values
        複数値更新メソッドのテスト"""
        config = PluginConfig()
        
        config.update(
            top_k=25,
            custom_param="value",
            another_param=123
        )
        
        assert config.top_k == 25
        assert config.get("custom_param") == "value"
        assert config.get("another_param") == 123
    
    def test_validation(self):
        """Test configuration validation
        設定検証のテスト"""
        # Valid configuration
        config = PluginConfig()
        assert config.validate() is True
        
        # Invalid configurations
        invalid_configs = [
            {"top_k": -1},
            {"similarity_threshold": -0.5},
            {"similarity_threshold": 1.5},
            {"batch_size": 0},
            {"max_context_length": -100},
            {"temperature": -1.0},
            {"temperature": 3.0},
            {"max_tokens": 0}
        ]
        
        for invalid_update in invalid_configs:
            test_config = PluginConfig()
            test_config.update(**invalid_update)
            assert test_config.validate() is False
    
    def test_for_plugin_type(self):
        """Test plugin-type specific configuration
        プラグインタイプ固有設定のテスト"""
        base_config = PluginConfig(plugin_name="base")
        
        # Vector store configuration
        vector_config = base_config.for_plugin_type("vector_store")
        assert vector_config.plugin_type == "vector_store"
        assert vector_config.algorithm == "cosine"
        assert vector_config.plugin_name == "base"  # Preserved
        
        # Keyword store configuration
        keyword_config = base_config.for_plugin_type("keyword_store")
        assert keyword_config.plugin_type == "keyword_store"
        assert keyword_config.algorithm == "bm25"
        
        # Reranker configuration
        reranker_config = base_config.for_plugin_type("reranker")
        assert reranker_config.plugin_type == "reranker"
        assert reranker_config.top_k == min(base_config.top_k, 5)
    
    def test_to_dict(self):
        """Test dictionary conversion
        辞書変換のテスト"""
        config = PluginConfig(
            plugin_name="test",
            top_k=15,
            custom_settings={"param1": "value1"}
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["plugin_name"] == "test"
        assert config_dict["top_k"] == 15
        assert config_dict["param1"] == "value1"  # Merged from custom_settings
        assert "custom_settings" not in config_dict  # Should be merged out
    
    def test_from_dict(self):
        """Test creation from dictionary
        辞書からの作成のテスト"""
        config_dict = {
            "plugin_name": "test",
            "top_k": 15,
            "custom_param": "custom_value",
            "unknown_param": 123
        }
        
        config = PluginConfig.from_dict(config_dict)
        
        assert config.plugin_name == "test"
        assert config.top_k == 15
        assert config.get("custom_param") == "custom_value"
        assert config.get("unknown_param") == 123


class TestYAMLSerialization:
    """Test cases for YAML serialization/deserialization
    YAMLシリアライゼーション/デシリアライゼーションのテストケース"""
    
    def test_yaml_round_trip(self):
        """Test YAML save and load round trip
        YAML保存・読み込みの往復テスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            # Create and save configuration
            original_config = PluginConfig(
                plugin_name="test_plugin",
                plugin_type="vector_store",
                top_k=20,
                embedding_model="custom-model"
            )
            original_config.set("custom_param", "custom_value")
            
            original_config.to_yaml(config_path)
            
            # Load configuration
            loaded_config = PluginConfig.from_yaml(config_path)
            
            # Verify loaded configuration
            assert loaded_config.plugin_name == "test_plugin"
            assert loaded_config.plugin_type == "vector_store"
            assert loaded_config.top_k == 20
            assert loaded_config.embedding_model == "custom-model"
            assert loaded_config.get("custom_param") == "custom_value"
    
    def test_yaml_file_not_found(self):
        """Test behavior when YAML file doesn't exist
        YAMLファイルが存在しない場合の動作テスト"""
        non_existent_path = Path("non_existent_config.yaml")
        config = PluginConfig.from_yaml(non_existent_path)
        
        # Should return default configuration
        assert config.plugin_name == "default"
        assert config.plugin_type == "unknown"
    
    def test_yaml_content_validation(self):
        """Test YAML content structure
        YAML内容構造のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            
            config = PluginConfig(
                plugin_name="yaml_test",
                top_k=15,
                debug=True
            )
            config.set("custom_list", [1, 2, 3])
            config.set("custom_dict", {"nested": "value"})
            
            config.to_yaml(config_path)
            
            # Load and verify YAML structure
            with open(config_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            assert yaml_content["plugin_name"] == "yaml_test"
            assert yaml_content["top_k"] == 15
            assert yaml_content["debug"] is True
            assert yaml_content["custom_list"] == [1, 2, 3]
            assert yaml_content["custom_dict"] == {"nested": "value"}


class TestConfigManager:
    """Test cases for ConfigManager class
    ConfigManagerクラスのテストケース"""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization
        ConfigManager初期化のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(tmpdir)
            
            assert manager.config_dir == Path(tmpdir)
            assert manager.config_dir.exists()
    
    def test_save_and_load_config(self):
        """Test saving and loading configurations
        設定の保存と読み込みのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(tmpdir)
            
            # Create and save configuration
            config = PluginConfig(
                plugin_name="test_save_load",
                plugin_type="keyword_store",
                algorithm="tfidf"
            )
            
            manager.save_config("test_config", config)
            
            # Load configuration
            loaded_config = manager.load_config("test_config")
            
            assert loaded_config.plugin_name == "test_save_load"
            assert loaded_config.plugin_type == "keyword_store"
            assert loaded_config.algorithm == "tfidf"
    
    def test_create_template(self):
        """Test template creation
        テンプレート作成のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(tmpdir)
            
            # Create vector store template
            template = manager.create_template("my_vector_store", "vector_store")
            
            assert template.plugin_name == "my_vector_store"
            assert template.plugin_type == "vector_store"
            assert template.algorithm == "cosine"
            
            # Verify template was saved
            configs = manager.list_configs()
            assert "template_my_vector_store" in configs
    
    def test_list_configs(self):
        """Test configuration listing
        設定一覧のテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(tmpdir)
            
            # Initially empty
            assert manager.list_configs() == []
            
            # Save some configurations
            config1 = PluginConfig(plugin_name="config1")
            config2 = PluginConfig(plugin_name="config2")
            
            manager.save_config("first", config1)
            manager.save_config("second", config2)
            
            configs = manager.list_configs()
            assert "first" in configs
            assert "second" in configs
            assert len(configs) == 2


class TestIntegration:
    """Integration tests for the plugin configuration system
    プラグイン設定システムの統合テスト"""
    
    def test_realistic_plugin_configuration(self):
        """Test realistic plugin configuration scenario
        現実的なプラグイン設定シナリオのテスト"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(tmpdir)
            
            # Create configurations for different plugin types
            configs = {
                "vector_store": PluginConfig(
                    plugin_name="openai_vector_store",
                    plugin_type="vector_store",
                    embedding_model="text-embedding-3-large",
                    top_k=20,
                    similarity_threshold=0.7
                ),
                "keyword_store": PluginConfig(
                    plugin_name="bm25_keyword_store",
                    plugin_type="keyword_store",
                    algorithm="bm25",
                    top_k=15
                ),
                "reranker": PluginConfig(
                    plugin_name="cross_encoder_reranker",
                    plugin_type="reranker",
                    rerank_model="ms-marco-MiniLM-L-12-v2",
                    top_k=5
                ),
                "reader": PluginConfig(
                    plugin_name="gpt4_reader",
                    plugin_type="reader",
                    llm_model="gpt-4o",
                    temperature=0.1,
                    max_tokens=1000
                )
            }
            
            # Save all configurations
            for name, config in configs.items():
                manager.save_config(name, config)
            
            # Load and verify all configurations
            for name in configs.keys():
                loaded = manager.load_config(name)
                assert loaded.plugin_name == configs[name].plugin_name
                assert loaded.plugin_type == configs[name].plugin_type
                assert loaded.validate() is True
            
            # Verify all configs are listed
            config_list = manager.list_configs()
            for name in configs.keys():
                assert name in config_list