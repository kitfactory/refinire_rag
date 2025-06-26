#!/usr/bin/env python3
"""
Simple test script for oneenv template functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag.env_template import get_env_template
from refinire_rag.config import config


def test_env_template():
    """Test the environment template"""
    print("Testing environment template...")
    
    template = get_env_template()
    print(f"✅ Template created successfully")
    print(f"📝 Variables count: {len(template.variables)}")
    print(f"🔧 Source: {template.source}")
    
    # Show critical variables
    critical_vars = [name for name, var in template.variables.items() if var.importance == "critical"]
    important_vars = [name for name, var in template.variables.items() if var.importance == "important"]
    optional_vars = [name for name, var in template.variables.items() if var.importance == "optional"]
    
    print(f"\n🔴 Critical ({len(critical_vars)}): {critical_vars}")
    print(f"🟡 Important ({len(important_vars)}): {important_vars}")
    print(f"🟢 Optional ({len(optional_vars)}): {optional_vars}")
    
    # Assertions for proper test validation
    assert template is not None
    assert hasattr(template, 'variables')
    assert hasattr(template, 'source')
    assert isinstance(template.variables, dict)
    assert len(template.variables) > 0


def test_config():
    """Test the configuration class"""
    print("\nTesting configuration class...")
    
    # Test critical config validation
    missing = config.get_missing_critical_vars()
    if missing:
        print(f"⚠️  Missing critical variables: {missing}")
    else:
        print("✅ All critical variables are set")
    
    # Show config summary
    summary = config.get_config_summary()
    print(f"📊 Config summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Assertions for proper test validation
    assert config is not None
    assert hasattr(config, 'get_missing_critical_vars')
    assert hasattr(config, 'get_config_summary')
    assert isinstance(missing, list)
    assert isinstance(summary, dict)


def test_generate_minimal_env():
    """Test generating a minimal .env.example manually"""
    print("\nGenerating minimal .env.example...")
    
    template = get_env_template()
    
    content = []
    content.append("# Environment Variables for refinire-rag")
    content.append("# Generated using oneenv template")
    content.append("")
    
    # Group by importance
    for importance in ["critical", "important", "optional"]:
        importance_vars = [(name, var) for name, var in template.variables.items() if var.importance == importance]
        if importance_vars:
            content.append(f"# {importance.upper()} Variables")
            content.append("")
            
            for name, var in importance_vars:
                content.append(f"# {var.description}")
                if var.choices:
                    content.append(f"# Choices: {', '.join(var.choices)}")
                if var.required:
                    content.append(f"{name}=")
                else:
                    content.append(f"{name}={var.default}")
                content.append("")
    
    # Write to file
    env_path = Path(".env.example")
    env_path.write_text("\n".join(content), encoding="utf-8")
    
    print(f"✅ Generated .env.example with {len(template.variables)} variables")
    print(f"📁 File saved to: {env_path.absolute()}")
    
    # Assertions for proper test validation
    assert template is not None
    assert len(content) > 0
    assert env_path.exists()
    assert env_path.stat().st_size > 0


if __name__ == "__main__":
    print("🧪 Testing oneenv integration for refinire-rag\n")
    
    success = True
    try:
        test_env_template()
        print("✅ test_env_template passed")
    except Exception as e:
        print(f"❌ test_env_template failed: {e}")
        success = False
    
    try:
        test_config()
        print("✅ test_config passed")
    except Exception as e:
        print(f"❌ test_config failed: {e}")
        success = False
    
    try:
        test_generate_minimal_env()
        print("✅ test_generate_minimal_env passed")
    except Exception as e:
        print(f"❌ test_generate_minimal_env failed: {e}")
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)