#!/usr/bin/env python3
"""
Debug DocumentStoreLoader
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from refinire_rag.application.corpus_manager_new import CorpusManager
from refinire_rag.loader.document_store_loader import DocumentStoreLoader, DocumentLoadConfig, LoadStrategy
from refinire_rag.models.document import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_document_store_loader():
    """Debug DocumentStoreLoader output"""
    logger.info("=== Debugging DocumentStoreLoader ===")
    
    # Create CorpusManager from environment
    corpus_manager = CorpusManager.from_env()
    
    # Create loader config
    loader_config = DocumentLoadConfig(
        strategy=LoadStrategy.FILTERED, 
        metadata_filters={"processing_stage": "original"}
    )
    
    # Create loader
    loader = DocumentStoreLoader(corpus_manager.document_store, load_config=loader_config)
    
    # Create trigger document
    trigger_doc = Document(
        id="debug_trigger",
        content="",
        metadata={
            "trigger_type": "debug_test"
        }
    )
    
    logger.info("Processing trigger document through DocumentStoreLoader...")
    
    # Process the trigger document
    results = list(loader.process([trigger_doc]))
    
    logger.info(f"DocumentStoreLoader returned {len(results)} items")
    
    # Check each result
    for i, result in enumerate(results):
        logger.info(f"Result {i}: type={type(result)}, value={result if not isinstance(result, Document) else f'Document(id={result.id})'}")
        
        if isinstance(result, Document):
            logger.info(f"  Document ID: {result.id}")
            logger.info(f"  Content length: {len(result.content)}")
            logger.info(f"  Metadata: {result.metadata}")
        else:
            logger.warning(f"  Expected Document but got {type(result)}: {result}")
    
    return results

if __name__ == "__main__":
    results = debug_document_store_loader()