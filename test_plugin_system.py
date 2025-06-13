"""
Manual test for plugin system
プラグインシステムの手動テスト

Simple test script to verify plugin system functionality without pytest.
pytestなしでプラグインシステム機能を検証するシンプルなテストスクリプト。
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_plugin_loader_basic():
    """Test basic plugin loader functionality / 基本プラグインローダー機能をテスト"""
    print("🔍 Testing plugin loader basic functionality...")
    
    try:
        from refinire.rag.plugins import PluginLoader, PluginRegistry
        
        # Test registry creation
        registry = PluginRegistry()
        print("✅ PluginRegistry created successfully")
        
        # Test loader creation
        loader = PluginLoader(registry)
        print("✅ PluginLoader created successfully")
        
        # Test plugin discovery
        loader.discover_plugins()
        print("✅ Plugin discovery completed")
        
        # List discovered plugins
        plugins = loader.registry.list_plugins()
        print(f"📦 Discovered {len(plugins)} plugins:")
        
        for plugin in plugins:
            status = "✅ Available" if plugin.is_available else "❌ Not available"
            print(f"   - {plugin.name} ({plugin.plugin_type}) - {status}")
            if not plugin.is_available:
                print(f"     Error: {plugin.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic plugin loader test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_plugin_functions():
    """Test global plugin functions / グローバルプラグイン関数をテスト"""
    print("\n🌐 Testing global plugin functions...")
    
    try:
        from refinire.rag.plugins.plugin_loader import get_plugin_loader, get_available_plugins
        
        # Test global loader
        loader = get_plugin_loader()
        print("✅ Global plugin loader retrieved")
        
        # Test convenience function
        available = get_available_plugins()
        print(f"✅ Available plugins: {available}")
        
        # Test by type
        vector_stores = get_available_plugins("vector_store")
        loaders = get_available_plugins("loader")
        retrievers = get_available_plugins("retriever")
        
        print(f"   Vector stores: {vector_stores}")
        print(f"   Loaders: {loaders}")
        print(f"   Retrievers: {retrievers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in global functions test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chroma_plugin_loading():
    """Test ChromaDB plugin loading if available / ChromaDBプラグイン読み込みをテスト（利用可能な場合）"""
    print("\n📦 Testing ChromaDB plugin loading...")
    
    try:
        from refinire.rag.plugins.plugin_loader import get_plugin_loader, PluginConfig
        
        loader = get_plugin_loader()
        available = loader.get_available_plugins("vector_store")
        
        # Check if ChromaDB plugin is available
        chroma_plugins = [p for p in available if 'chroma' in p]
        
        if not chroma_plugins:
            print("ℹ️ ChromaDB plugin not installed - skipping load test")
            return True
        
        plugin_name = chroma_plugins[0]
        print(f"🎯 Found ChromaDB plugin: {plugin_name}")
        
        # Try to load the plugin
        config = PluginConfig(
            name=plugin_name,
            version="1.0.0",
            config={"collection_name": "test_collection"}
        )
        
        plugin = loader.load_plugin(plugin_name, config)
        
        if plugin:
            print("✅ ChromaDB plugin loaded successfully")
            print(f"   Plugin name: {plugin.name}")
            print(f"   Plugin version: {plugin.version}")
            print(f"   Plugin enabled: {plugin.enabled}")
            
            # Test plugin info
            info = plugin.get_info()
            print(f"   Plugin info: {info}")
            
            return True
        else:
            print("❌ Failed to load ChromaDB plugin")
            return False
            
    except Exception as e:
        print(f"❌ Error in ChromaDB plugin test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_refinire_rag_imports():
    """Test that refinire-rag imports work correctly / refinire-ragインポートが正しく動作するかテスト"""
    print("\n📥 Testing refinire-rag imports...")
    
    try:
        # Test core imports
        from refinire.rag import check_plugin_availability, get_available_plugins
        print("✅ Core plugin functions imported")
        
        # Test plugin availability check
        availability = check_plugin_availability()
        print(f"✅ Plugin availability check: {availability}")
        
        # Test get available plugins
        available = get_available_plugins()
        print(f"✅ Available plugins: {available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in imports test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests / すべてのテストを実行"""
    print("🚀 Plugin System Manual Test")
    print("=" * 50)
    
    tests = [
        ("Basic Plugin Loader", test_plugin_loader_basic),
        ("Global Plugin Functions", test_global_plugin_functions),
        ("Refinire-RAG Imports", test_refinire_rag_imports),
        ("ChromaDB Plugin Loading", test_chroma_plugin_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Plugin system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)