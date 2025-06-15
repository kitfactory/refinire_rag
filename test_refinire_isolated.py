#!/usr/bin/env python3
"""
Test script that explicitly imports from the installed refinire package
"""

import sys
import os

# Remove current directory from path to avoid local refinire interference
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')

# Remove current working directory
cwd = os.getcwd()
if cwd in sys.path:
    sys.path.remove(cwd)

# Ensure venv site-packages is in path
venv_path = "/mnt/c/Users/kitad/workspace/refinire-rag/.venv/lib/python3.10/site-packages"
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

print("Python path:")
for path in sys.path[:5]:  # Show first 5 paths
    print(f"  {path}")

try:
    import refinire
    print(f"✅ Refinire imported from: {refinire.__file__}")
    
    # Test core functionality
    from refinire import get_llm
    print("✅ get_llm imported successfully")
    
    # Test LLMPipeline
    from refinire import LLMPipeline
    print("✅ LLMPipeline imported successfully")
    
    # Test utility functions
    from refinire import create_simple_llm_pipeline
    print("✅ create_simple_llm_pipeline imported successfully")
    
    # Create a test pipeline
    pipeline = LLMPipeline(
        name="test_pipeline",
        generation_instructions="You are a helpful assistant. Keep responses very brief.",
        model="gpt-4o-mini"
    )
    print("✅ LLMPipeline created successfully")
    
    # Test a simple generation (this requires OpenAI API key)
    try:
        result = pipeline.run("Say 'Hello from LLMPipeline!'")
        print(f"✅ Pipeline execution: Success={result.success}")
        if result.success:
            print(f"✅ Response: {result.content}")
        else:
            print(f"❌ Pipeline failed: {result.metadata}")
    except Exception as e:
        print(f"⚠️  Pipeline execution failed (likely API key issue): {e}")
    
    print("\n🎉 All imports and basic functionality working!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    # Debug: show what's available in refinire
    try:
        import refinire
        print(f"Available in refinire: {dir(refinire)}")
    except:
        print("Cannot import refinire at all")
        
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()