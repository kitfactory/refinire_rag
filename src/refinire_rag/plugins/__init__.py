"""
Plugin system for refinire-rag
プラグインシステム for refinire-rag

This module provides a plugin loader and registry system for dynamically loading
and managing external plugins for refinire-rag.

このモジュールは、refinire-ragの外部プラグインを動的に読み込み、管理するための
プラグインローダーとレジストリシステムを提供します。
"""

from .plugin_loader import PluginLoader, PluginRegistry, get_plugin_loader
from .base import PluginInterface, PluginConfig
from .plugin_config import ConfigManager

__all__ = [
    "PluginLoader",
    "PluginRegistry", 
    "PluginInterface",
    "PluginConfig",
    "ConfigManager",
    "get_plugin_loader",
]