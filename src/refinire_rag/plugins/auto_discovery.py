"""
Auto-discovery system for plugins using entry points
エントリーポイントを使用したプラグインの自動発見システム
"""

import importlib
import logging
from typing import Dict, List, Type, Optional
try:
    from importlib.metadata import entry_points, distributions
except ImportError:
    from importlib_metadata import entry_points, distributions  # type: ignore

logger = logging.getLogger(__name__)


class PluginAutoDiscovery:
    """
    Automatic plugin discovery system
    自動プラグイン発見システム
    
    This system discovers plugins using Python entry points and
    package scanning techniques.
    
    Pythonエントリーポイントとパッケージスキャン技術を使用して
    プラグインを発見するシステム。
    """
    
    def __init__(self):
        self.discovered_plugins = {}
        self._scanned = False
    
    def scan_entry_points(self) -> Dict[str, Dict]:
        """
        Scan for plugins using setuptools entry points
        setuptoolsエントリーポイントを使用してプラグインをスキャン
        
        External plugins should register themselves like:
        
        setup(
            entry_points={
                'refinire_rag.vectorstore': [
                    'ChromaVectorStore = refinire_rag_chroma:ChromaVectorStore',
                ],
                'refinire_rag.keywordstore': [
                    'BM25Store = refinire_rag_bm25:BM25Store',
                ],
                'refinire_rag.loaders': [
                    'DoclingLoader = refinire_rag_docling:DoclingLoader',
                ]
            }
        )
        
        Returns:
            Dict: Discovered plugins by type
                 タイプ別の発見されたプラグイン
        """
        plugins = {
            'vectorstore': {},
            'keywordstore': {},
            'loaders': {}
        }
        
        # Scan entry points for each plugin type
        entry_point_groups = [
            ('vectorstore', 'refinire_rag.vectorstore'),
            ('keywordstore', 'refinire_rag.keywordstore'), 
            ('loaders', 'refinire_rag.loaders')
        ]
        
        for plugin_type, group_name in entry_point_groups:
            try:
                eps = entry_points()
                if hasattr(eps, 'select'):  # Python 3.10+
                    group_eps = eps.select(group=group_name)
                else:  # Python 3.8, 3.9
                    group_eps = eps.get(group_name, [])
                
                for entry_point in group_eps:
                    try:
                        plugin_class = entry_point.load()
                        plugins[plugin_type][entry_point.name] = {
                            'class': plugin_class,
                            'module': entry_point.module,
                            'entry_point': entry_point
                        }
                        logger.info(f"Discovered {plugin_type} plugin: {entry_point.name}")
                    except Exception as e:
                        logger.error(f"Failed to load plugin {entry_point.name}: {e}")
            except Exception as e:
                logger.debug(f"No entry points found for {group_name}: {e}")
        
        return plugins
    
    def scan_installed_packages(self) -> Dict[str, Dict]:
        """
        Scan installed packages for refinire-rag plugins
        refinire-ragプラグイン用にインストール済みパッケージをスキャン
        
        Looks for packages with names matching refinire-rag-* pattern
        refinire-rag-*パターンに一致する名前のパッケージを探索
        
        Returns:
            Dict: Discovered plugins by scanning packages
                 パッケージスキャンで発見されたプラグイン
        """
        plugins = {
            'vectorstore': {},
            'keywordstore': {},
            'loaders': {}
        }
        
        # Get all installed packages
        installed_packages = [dist.metadata['Name'] for dist in distributions()]
        
        # Look for refinire-rag plugin packages
        plugin_packages = [pkg for pkg in installed_packages if pkg.startswith('refinire-rag-')]
        
        for package_name in plugin_packages:
            try:
                # Try to import the package
                module_name = package_name.replace('-', '_')
                module = importlib.import_module(module_name)
                
                # Check if it has plugin metadata
                if hasattr(module, '__plugin_type__') and hasattr(module, '__plugin_class__'):
                    plugin_type = module.__plugin_type__
                    plugin_class = module.__plugin_class__
                    
                    # Map plugin type to our categories
                    if plugin_type in ['vector_store', 'vectorstore']:
                        category = 'vectorstore'
                    elif plugin_type in ['keyword_store', 'keywordstore']:
                        category = 'keywordstore'
                    elif plugin_type in ['loader', 'loaders']:
                        category = 'loaders'
                    else:
                        logger.warning(f"Unknown plugin type: {plugin_type}")
                        continue
                    
                    plugin_name = getattr(module, '__plugin_name__', plugin_class.__name__)
                    
                    plugins[category][plugin_name] = {
                        'class': plugin_class,
                        'module': module_name,
                        'package': package_name
                    }
                    
                    logger.info(f"Discovered {category} plugin: {plugin_name} from {package_name}")
                
            except Exception as e:
                logger.debug(f"Could not scan package {package_name}: {e}")
        
        return plugins
    
    def discover_all_plugins(self) -> Dict[str, Dict]:
        """
        Comprehensive plugin discovery
        包括的なプラグイン発見
        
        Combines entry point scanning and package scanning
        エントリーポイントスキャンとパッケージスキャンを組み合わせ
        
        Returns:
            Dict: All discovered plugins
                 発見されたすべてのプラグイン
        """
        if self._scanned:
            return self.discovered_plugins
        
        logger.info("Starting plugin auto-discovery...")
        
        # Method 1: Entry points (preferred)
        entry_point_plugins = self.scan_entry_points()
        
        # Method 2: Package scanning (fallback)
        package_plugins = self.scan_installed_packages()
        
        # Merge results (entry points take precedence)
        for plugin_type in ['vectorstore', 'keywordstore', 'loaders']:
            self.discovered_plugins[plugin_type] = {}
            
            # Add package-discovered plugins first
            self.discovered_plugins[plugin_type].update(package_plugins[plugin_type])
            
            # Override with entry point plugins (higher priority)
            self.discovered_plugins[plugin_type].update(entry_point_plugins[plugin_type])
        
        self._scanned = True
        
        total_plugins = sum(len(plugins) for plugins in self.discovered_plugins.values())
        logger.info(f"Plugin discovery complete. Found {total_plugins} plugins.")
        
        return self.discovered_plugins
    
    def get_plugins_by_type(self, plugin_type: str) -> Dict[str, Dict]:
        """
        Get discovered plugins of specific type
        特定タイプの発見されたプラグインを取得
        
        Args:
            plugin_type: 'vectorstore', 'keywordstore', or 'loaders'
                        'vectorstore'、'keywordstore'、または'loaders'
        
        Returns:
            Dict: Plugins of the specified type
                 指定されたタイプのプラグイン
        """
        if not self._scanned:
            self.discover_all_plugins()
        
        return self.discovered_plugins.get(plugin_type, {})
    
    def refresh_discovery(self):
        """
        Refresh plugin discovery (useful after installing new plugins)
        プラグイン発見をリフレッシュ（新しいプラグインをインストール後に有用）
        """
        self._scanned = False
        self.discovered_plugins = {}
        self.discover_all_plugins()


# Global discovery instance
auto_discovery = PluginAutoDiscovery()


def get_auto_discovered_plugins() -> Dict[str, Dict]:
    """Get all auto-discovered plugins"""
    return auto_discovery.discover_all_plugins()


def refresh_plugin_discovery():
    """Refresh plugin discovery"""
    auto_discovery.refresh_discovery()


__all__ = ['PluginAutoDiscovery', 'auto_discovery', 'get_auto_discovered_plugins', 'refresh_plugin_discovery']