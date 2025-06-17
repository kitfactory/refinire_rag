#!/usr/bin/env python3
"""
Script to generate .env.example file using oneenv template.
Run this script to create a comprehensive .env.example file.
"""
"""
oneenvテンプレートを使って.env.exampleファイルを生成するスクリプト。
このスクリプトを実行して包括的な.env.exampleファイルを作成します。
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oneenv import generate_env_example
from refinire_rag.env_template import get_env_template


def main():
    """
    Generates .env.example file from the refinire-rag environment template.
    """
    """
    refinire-rag環境テンプレートから.env.exampleファイルを生成します。
    """
    
    # Get the template
    template = get_env_template()
    
    # Generate .env.example in the project root
    project_root = Path(__file__).parent.parent
    env_example_path = project_root / ".env.example"
    
    print(f"Generating .env.example file...")
    print(f"Output path: {env_example_path}")
    
    # Generate the file using oneenv 0.3.1 API
    generate_env_example(
        output_path=str(env_example_path),
        debug=False
    )
    
    print(f"✅ Successfully generated .env.example!")
    print(f"📝 The file contains {len(template.variables)} environment variables")
    
    # Count by importance
    critical_count = len([v for v in template.variables if v.importance == "critical"])
    important_count = len([v for v in template.variables if v.importance == "important"])
    optional_count = len([v for v in template.variables if v.importance == "optional"])
    
    print(f"🔴 Critical: {critical_count} variables")
    print(f"🟡 Important: {important_count} variables") 
    print(f"🟢 Optional: {optional_count} variables")
    
    print(f"\n📖 Usage:")
    print(f"   1. Copy .env.example to .env")
    print(f"   2. Set your OPENAI_API_KEY (required)")
    print(f"   3. Adjust important settings as needed")
    print(f"   4. Optional variables can be left at defaults")


if __name__ == "__main__":
    main()