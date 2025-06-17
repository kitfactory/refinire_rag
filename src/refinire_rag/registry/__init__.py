"""
Registry module for plugin management

This module provides unified registration and discovery of both
built-in components and external plugins.
"""

from .plugin_registry import PluginRegistry

__all__ = ['PluginRegistry']