"""
Factory module for creating components from environment variables

This module provides factories for creating both built-in and
external plugin components based on environment variable configuration.
"""

from .plugin_factory import PluginFactory

__all__ = ['PluginFactory']