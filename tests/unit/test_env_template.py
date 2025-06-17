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
    
    try:
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
        
        return True
    except Exception as e:
        print(f"❌ Error creating template: {e}")
        return False


def test_config():
    """Test the configuration class"""
    print("\nTesting configuration class...")
    
    try:
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
        
        return True
    except Exception as e:
        print(f"❌ Error testing config: {e}")
        return False


def generate_minimal_env():
    """Generate a minimal .env.example manually"""
    print("\nGenerating minimal .env.example...")
    
    try:
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
        
        return True
    except Exception as e:
        print(f"❌ Error generating .env.example: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing oneenv integration for refinire-rag\n")
    
    success = True
    success &= test_env_template()
    success &= test_config()
    success &= generate_minimal_env()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)