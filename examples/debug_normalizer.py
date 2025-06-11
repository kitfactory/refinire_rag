#!/usr/bin/env python3
"""
Debug Normalizer - Test normalizer functionality step by step
"""

import sys
import tempfile
import logging
from pathlib import Path

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.processing.normalizer import Normalizer, NormalizerConfig
from refinire_rag.models.document import Document


def create_test_dictionary(temp_dir: Path) -> str:
    """Create a test dictionary file for normalization"""
    
    dict_file = temp_dir / "test_dictionary.md"
    dict_file.write_text("""# ドメイン用語辞書

## 技術用語

- **RAG** (Retrieval-Augmented Generation): 検索拡張生成
  - 表現揺らぎ: 検索拡張生成, 検索強化生成, RAGシステム

- **ベクトル検索** (Vector Search): ベクトル検索  
  - 表現揺らぎ: ベクトル検索, 意味検索, セマンティック検索

- **LLM** (Large Language Model): 大規模言語モデル
  - 表現揺らぎ: 大規模言語モデル, 言語モデル, LLMモデル
""", encoding='utf-8')
    
    return str(dict_file)


def debug_dictionary_parsing(dict_path: str):
    """Debug dictionary parsing step by step"""
    
    print("\n" + "="*60)
    print("🔍 DICTIONARY PARSING DEBUG")
    print("="*60)
    
    # Read dictionary file
    print(f"\n📖 Reading dictionary file: {dict_path}")
    with open(dict_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Dictionary content ({len(content)} chars):")
    print("-" * 40)
    print(content)
    print("-" * 40)
    
    # Test parsing manually
    print(f"\n🔧 Parsing dictionary manually...")
    
    # Import the normalizer to use its parsing method
    from refinire_rag.processing.normalizer import Normalizer
    
    # Create normalizer temporarily to access parsing method
    temp_config = NormalizerConfig(dictionary_file_path=dict_path)
    temp_normalizer = Normalizer(temp_config)
    
    # Parse content
    mappings = temp_normalizer._parse_dictionary_content(content)
    
    print(f"📊 Parsed {len(mappings)} mappings:")
    for variation, standard in mappings.items():
        print(f"   '{variation}' → '{standard}'")
    
    return mappings


def debug_normalization_process(dict_path: str, mappings: dict):
    """Debug the complete normalization process"""
    
    print("\n" + "="*60)
    print("🔧 NORMALIZATION PROCESS DEBUG") 
    print("="*60)
    
    # Create normalizer
    config = NormalizerConfig(
        dictionary_file_path=dict_path,
        normalize_variations=True,
        expand_abbreviations=True,
        whole_word_only=False,  # Disable for Japanese text
        case_sensitive_replacement=False
    )
    
    normalizer = Normalizer(config)
    
    print(f"🔧 Created normalizer with config:")
    print(f"   - Dictionary path: {config.dictionary_file_path}")
    print(f"   - Normalize variations: {config.normalize_variations}")
    print(f"   - Expand abbreviations: {config.expand_abbreviations}")
    print(f"   - Whole word only: {config.whole_word_only}")
    print(f"   - Case sensitive: {config.case_sensitive_replacement}")
    
    # Test queries
    test_queries = [
        "検索強化生成について教えて",
        "意味検索の仕組みは？", 
        "RAGシステムの利点を説明して",
        "セマンティック検索とは何ですか？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📌 Test {i}: '{query}'")
        
        # Create query document
        query_doc = Document(
            id=f"debug_query_{i}",
            content=query,
            metadata={"is_query": True}
        )
        
        print(f"   📄 Created document: {query_doc.id}")
        print(f"   📝 Content: '{query_doc.content}'")
        
        # Test manual normalization first
        print(f"   🔍 Manual normalization test:")
        for variation, standard in mappings.items():
            if variation in query:
                print(f"     Found '{variation}' in query → should become '{standard}'")
                
        # Process with normalizer
        try:
            result = normalizer.process(query_doc)
            
            if result:
                normalized_doc = result[0]
                print(f"   ✅ Processed successfully")
                print(f"   📄 Original: '{query_doc.content}'")
                print(f"   📄 Normalized: '{normalized_doc.content}'")
                print(f"   🔄 Changed: {'Yes' if normalized_doc.content != query_doc.content else 'No'}")
                
                # Check metadata
                if "normalization_stats" in normalized_doc.metadata:
                    stats = normalized_doc.metadata["normalization_stats"]
                    print(f"   📊 Stats: {stats}")
                else:
                    print(f"   📊 No normalization stats in metadata")
            else:
                print(f"   ❌ Processing returned empty result")
                
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Check internal state
    print(f"\n📊 Normalizer internal state:")
    print(f"   - Mappings loaded: {len(normalizer._normalization_mappings)}")
    print(f"   - Dictionary last modified: {normalizer._dictionary_last_modified}")
    print(f"   - Config path: {normalizer.config.dictionary_file_path}")


def main():
    """Main debug function"""
    
    print("🔍 Normalizer Debug Tool")
    print("="*60)
    print("Step-by-step debugging of the normalization process")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test dictionary
        dict_path = create_test_dictionary(temp_dir)
        print(f"📖 Created test dictionary: {dict_path}")
        
        # Debug dictionary parsing
        mappings = debug_dictionary_parsing(dict_path)
        
        # Debug normalization process
        debug_normalization_process(dict_path, mappings)
        
        print("\n🎉 Debug completed!")
        print(f"📁 Files in: {temp_dir}")
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)