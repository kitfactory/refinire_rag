"""
CorpusManager - Document corpus construction and management

Simplified CorpusManager with core functionality for document import and rebuild.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from ..processing.normalizer import NormalizerConfig
from ..processing.chunker import ChunkingConfig
from ..loader.document_store_loader import DocumentStoreLoader, DocumentLoadConfig, LoadStrategy
from ..models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    """Statistics for corpus building operations"""
    total_files_processed: int = 0
    total_documents_created: int = 0
    total_chunks_created: int = 0
    total_processing_time: float = 0.0
    pipeline_stages_executed: int = 0
    documents_by_stage: Dict[str, int] = None
    errors_encountered: int = 0
    
    def __post_init__(self):
        if self.documents_by_stage is None:
            self.documents_by_stage = {}


class CorpusManager:
    """Document corpus construction and management system
    
    Simplified corpus manager that provides:
    - Document import from folders with incremental loading
    - Corpus rebuild from original documents using existing knowledge artifacts
    - Corpus clearing functionality
    
    Environment Variables:
    - REFINIRE_DIR: Base directory for Refinire files (default: './refinire')
    - REFINIRE_RAG_CORPUS_STORE: Corpus store type (default: 'sqlite')
    
    File Naming Convention:
    - Tracking file: {corpus_name}_track.json
    - Dictionary file: {corpus_name}_dictionary.md
    - Knowledge graph file: {corpus_name}_knowledge_graph.md
    """
    
    def __init__(self, document_store, vector_store, config: Optional[Dict[str, Any]] = None):
        """Initialize CorpusManager
        
        Args:
            document_store: DocumentStore for document persistence
            vector_store: VectorStore for vector persistence
            config: Optional global configuration
        """
        self.document_store = document_store
        self.vector_store = vector_store
        self.config = config or {}
        self.stats = CorpusStats()
        
        logger.info(f"Initialized CorpusManager with DocumentStore: {type(document_store).__name__}, "
                   f"VectorStore: {type(vector_store).__name__}")
    
    @staticmethod
    def _get_refinire_rag_dir() -> Path:
        """Get REFINIRE_DIR/rag directory from environment variable or default
        
        環境変数REFINIRE_DIRまたはデフォルトディレクトリを取得し、/ragサブディレクトリを使用
        
        Returns:
            Path to the REFINIRE_DIR/rag directory
        """
        import os
        from pathlib import Path
        
        # Check REFINIRE_DIR environment variable first
        base_dir = os.getenv("REFINIRE_DIR", "./refinire")
        rag_path = Path(base_dir) / "rag"
        rag_path.mkdir(parents=True, exist_ok=True)
        
        return rag_path
    
    @staticmethod
    def _get_corpus_file_path(corpus_name: str, file_type: str, custom_dir: Optional[str] = None) -> Path:
        """Get corpus-specific file path
        
        コーパス固有のファイルパスを取得
        
        Args:
            corpus_name: Name of the corpus
            file_type: Type of file ('track', 'dictionary', 'knowledge_graph')
            custom_dir: Custom directory path (overrides default)
            
        Returns:
            Path to the corpus file
        """
        if custom_dir:
            base_dir = Path(custom_dir)
        else:
            base_dir = CorpusManager._get_refinire_rag_dir()
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # File naming convention: [corpus_name]_[file_type].[ext]
        if file_type == "track":
            filename = f"{corpus_name}_track.json"
        elif file_type == "dictionary":
            filename = f"{corpus_name}_dictionary.md"
        elif file_type == "knowledge_graph":
            filename = f"{corpus_name}_knowledge_graph.md"
        else:
            raise ValueError(f"Unknown file type: {file_type}")
        
        return base_dir / filename
    
    @staticmethod
    def _get_default_output_directory(env_var_name: str, subdir: str) -> Path:
        """Get default output directory from environment variable or .refinire/
        
        環境変数またはデフォルト.refinire/ディレクトリから出力ディレクトリを取得
        
        Args:
            env_var_name: Environment variable name to check
            subdir: Subdirectory name under .refinire/
            
        Returns:
            Path to the output directory
        """
        import os
        from pathlib import Path
        
        # Check environment variable first
        env_path = os.getenv(env_var_name)
        if env_path:
            return Path(env_path)
        
        # Fall back to .refinire/subdir in user's home directory
        home_dir = Path.home()
        default_dir = home_dir / ".refinire" / subdir
        default_dir.mkdir(parents=True, exist_ok=True)
        
        return default_dir
    
    def _create_filter_config_from_glob(self, glob_pattern: str) -> Optional['FilterConfig']:
        """Create FilterConfig from glob pattern
        
        globパターンからFilterConfigを作成
        
        Args:
            glob_pattern: Glob pattern (e.g., "**/*.md", "*.{txt,py}")
            
        Returns:
            FilterConfig object or None if no filtering needed
        """
        from ..loader.models.filter_config import FilterConfig
        import re
        
        # Extract file extensions from glob pattern
        extensions = []
        
        # Handle patterns like "**/*.md", "*.txt"
        if '*.' in glob_pattern:
            # Simple extension pattern
            ext_match = re.search(r'\*\.([a-zA-Z0-9]+)', glob_pattern)
            if ext_match:
                extensions.append('.' + ext_match.group(1))
        
        # Handle patterns like "*.{txt,md,py}"
        brace_match = re.search(r'\*\.{([^}]+)}', glob_pattern)
        if brace_match:
            ext_list = brace_match.group(1).split(',')
            extensions.extend(['.' + ext.strip() for ext in ext_list])
        
        # Create filter config if extensions found
        if extensions:
            from ..loader.filters.extension_filter import ExtensionFilter
            extension_filter = ExtensionFilter(include_extensions=extensions)
            return FilterConfig(extension_filter=extension_filter)
        
        # For complex patterns, return None and let glob be handled by the loader
        return None
    
    def import_original_documents(self, 
                                corpus_name: str,
                                directory: str,
                                glob: str = "**/*",
                                use_multithreading: bool = True,
                                force_reload: bool = False,
                                additional_metadata: Optional[Dict[str, Any]] = None,
                                tracking_file_path: Optional[str] = None,
                                create_dictionary: bool = False,
                                create_knowledge_graph: bool = False,
                                dictionary_output_dir: Optional[str] = None,
                                graph_output_dir: Optional[str] = None) -> CorpusStats:
        """Import original documents from specified directory with incremental loading
        
        指定ディレクトリからIncrementalLoaderを使って元文書を取り込み、
        processing_stage: "original"メタデータを自動設定し、オプションで辞書・グラフを作成
        
        Args:
            corpus_name: Name of the corpus (used in metadata and output filenames)
                       コーパス名（メタデータと出力ファイル名に使用）
            directory: Directory path to import from (similar to LangChain DirectoryLoader)
                     取り込み対象ディレクトリパス（LangChain DirectoryLoaderと同様）
            glob: Glob pattern to match files (default: "**/*" for all files recursively)
                ファイルマッチング用のglobパターン（デフォルト: "**/*" 全ファイル再帰的）
            use_multithreading: Whether to use multithreading for file processing
                              ファイル処理にマルチスレッドを使用するか
            force_reload: Force reload all files ignoring incremental cache
                        増分キャッシュを無視してすべてのファイルを強制再読み込み
            additional_metadata: Additional metadata to add to all imported documents
                               すべての取り込み文書に追加する追加メタデータ
            tracking_file_path: Path to store file tracking data for incremental loading
                              増分ローディング用ファイル追跡データの保存パス
            create_dictionary: Whether to create domain dictionary after import
                             取り込み後にドメイン辞書を作成するか
            create_knowledge_graph: Whether to create knowledge graph after import
                                  取り込み後にナレッジグラフを作成するか
            dictionary_output_dir: Directory to save dictionary file (default: env REFINIRE_DICTIONARY_DIR or ~/.refinire/dictionaries)
                                 辞書ファイルの保存ディレクトリ（デフォルト: 環境変数REFINIRE_DICTIONARY_DIRまたは~/.refinire/dictionaries）
            graph_output_dir: Directory to save graph file (default: env REFINIRE_GRAPH_DIR or ~/.refinire/graphs)
                            グラフファイルの保存ディレクトリ（デフォルト: 環境変数REFINIRE_GRAPH_DIRまたは~/.refinire/graphs）
        
        Returns:
            CorpusStats: Import statistics including files processed and documents created
                        処理ファイル数と作成文書数を含む取り込み統計
        
        Example:
            # 基本的な取り込み（全ファイル）
            stats = corpus_manager.import_original_documents(
                corpus_name="product_docs",
                directory="/path/to/docs"
            )
            
            # Markdownファイルのみ取り込み（LangChain風）
            stats = corpus_manager.import_original_documents(
                corpus_name="markdown_docs",
                directory="/path/to/docs",
                glob="**/*.md",
                use_multithreading=True
            )
            
            # 詳細設定での取り込み
            stats = corpus_manager.import_original_documents(
                corpus_name="engineering_docs",
                directory="/path/to/docs",
                glob="**/*.{txt,md,py}",  # 複数拡張子
                use_multithreading=False,
                additional_metadata={"department": "engineering", "project": "rag"},
                tracking_file_path="./import_tracking.json",
                create_dictionary=True,
                create_knowledge_graph=True,
                dictionary_output_dir="./knowledge",
                graph_output_dir="./knowledge"
            )
            # → ./knowledge/engineering_docs_dictionary.md
            # → ./knowledge/engineering_docs_graph.md が作成される
        """
        from ..loader.incremental_directory_loader import IncrementalDirectoryLoader
        from ..metadata.constant_metadata import ConstantMetadata
        from ..loader.models.filter_config import FilterConfig
        from datetime import datetime
        
        stats = CorpusStats()
        
        # Create ConstantMetadata to automatically add processing_stage: "original"
        base_metadata = {
            "processing_stage": "original",
            "import_timestamp": datetime.now().isoformat(),
            "imported_by": "import_original_documents",
            "corpus_name": corpus_name
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            base_metadata.update(additional_metadata)
        
        constant_metadata_processor = ConstantMetadata(base_metadata)
        
        # Set default tracking file path if not provided
        if tracking_file_path is None:
            tracking_file_path = self._get_corpus_file_path(corpus_name, "track")
            logger.info(f"Using default tracking file: {tracking_file_path}")
        else:
            # Convert string to Path if provided as string
            tracking_file_path = Path(tracking_file_path)
        
        # Create filter configuration from glob pattern
        filter_config = None
        if glob != "**/*":
            # Convert glob pattern to filter configuration
            filter_config = self._create_filter_config_from_glob(glob)
        
        try:
            logger.info(f"Importing documents from: {directory}")
            logger.info(f"Using glob pattern: {glob}")
            if use_multithreading:
                logger.warning("Multithreading not yet supported by IncrementalDirectoryLoader, processing sequentially")
            
            # Create incremental loader for the directory
            incremental_loader = IncrementalDirectoryLoader(
                directory_path=directory,
                document_store=self.document_store,
                filter_config=filter_config,
                tracking_file_path=tracking_file_path,
                recursive=True,  # Always recursive since glob pattern can specify depth
                metadata_processors=[constant_metadata_processor]
                # Note: use_multithreading not yet supported by IncrementalDirectoryLoader
            )
            
            # Handle force reload
            if force_reload:
                incremental_loader.file_tracker.clear_tracking_data()
            
            sync_result = incremental_loader.sync_with_store()
            
            # Update statistics
            documents_processed = len(sync_result.added_documents) + len(sync_result.updated_documents)
            stats.total_files_processed += documents_processed
            stats.total_documents_created += documents_processed
            stats.pipeline_stages_executed += 1
            
            # Track by stage
            stage_key = "original"
            if stage_key not in stats.documents_by_stage:
                stats.documents_by_stage[stage_key] = 0
            stats.documents_by_stage[stage_key] += documents_processed
            
            # Track errors
            if sync_result.has_errors:
                stats.errors_encountered += len(sync_result.errors)
            
            logger.info(f"Imported {documents_processed} documents from {directory}")
            
        except Exception as e:
            logger.error(f"Error importing from directory {directory}: {e}")
            stats.errors_encountered += 1
            raise
        
        logger.info(f"Import completed: {stats.total_documents_created} documents from {directory}")
        
        # Create dictionary and/or knowledge graph if requested
        # 辞書・ナレッジグラフ作成（要求された場合）
        if create_dictionary or create_knowledge_graph:
            self._create_knowledge_artifacts(
                corpus_name=corpus_name,
                create_dictionary=create_dictionary,
                create_knowledge_graph=create_knowledge_graph,
                dictionary_output_dir=dictionary_output_dir,
                graph_output_dir=graph_output_dir,
                stats=stats
            )
        
        return stats
    
    def _create_knowledge_artifacts(self,
                                  corpus_name: str,
                                  create_dictionary: bool,
                                  create_knowledge_graph: bool,
                                  dictionary_output_dir: Optional[str],
                                  graph_output_dir: Optional[str],
                                  stats: CorpusStats):
        """Create dictionary and/or knowledge graph from imported documents
        
        取り込み済み文書から辞書・ナレッジグラフを作成
        """
        from ..processing.dictionary_maker import DictionaryMaker, DictionaryMakerConfig
        from ..processing.graph_builder import GraphBuilder, GraphBuilderConfig
        from ..processing.document_pipeline import DocumentPipeline
        from pathlib import Path
        import os
        
        knowledge_stages = []
        
        # Prepare dictionary creation
        if create_dictionary:
            # Use corpus-specific file path with environment variable support
            dict_file_path = self._get_corpus_file_path(corpus_name, "dictionary", dictionary_output_dir)
            
            dict_config = DictionaryMakerConfig(
                dictionary_file_path=str(dict_file_path),
                focus_on_technical_terms=True,
                extract_abbreviations=True,
                detect_expression_variations=True
            )
            knowledge_stages.append(("dictionary", dict_config))
            logger.info(f"Will create dictionary: {dict_file_path}")
        
        # Prepare knowledge graph creation  
        if create_knowledge_graph:
            # Use corpus-specific file path with environment variable support
            graph_file_path = self._get_corpus_file_path(corpus_name, "knowledge_graph", graph_output_dir)
            
            graph_config = GraphBuilderConfig(
                graph_file_path=str(graph_file_path),
                focus_on_important_relationships=True,
                extract_hierarchical_relationships=True,
                extract_causal_relationships=True
            )
            knowledge_stages.append(("graph", graph_config))
            logger.info(f"Will create knowledge graph: {graph_file_path}")
        
        # Execute knowledge extraction stages
        if knowledge_stages:
            try:
                logger.info(f"Creating knowledge artifacts for corpus '{corpus_name}'...")
                
                # Load original documents
                loader_config = DocumentLoadConfig(strategy=LoadStrategy.FILTERED, metadata_filters={"processing_stage": "original"})
                loader = DocumentStoreLoader(self.document_store, load_config=loader_config)
                
                # Create trigger document
                trigger_doc = Document(
                    id="knowledge_creation_trigger",
                    content="",
                    metadata={
                        "trigger_type": "knowledge_creation",
                        "corpus_name": corpus_name
                    }
                )
                
                # Execute each knowledge stage
                for stage_name, stage_config in knowledge_stages:
                    if stage_name == "dictionary":
                        processors = [loader, DictionaryMaker(stage_config)]
                    elif stage_name == "graph":
                        processors = [loader, GraphBuilder(stage_config)]
                    else:
                        continue
                    
                    pipeline = DocumentPipeline(processors)
                    results = pipeline.process_document(trigger_doc)
                    
                    stats.pipeline_stages_executed += 1
                    logger.info(f"Knowledge stage '{stage_name}' completed")
                
                logger.info(f"Knowledge artifacts created successfully for '{corpus_name}'")
                
            except Exception as e:
                logger.error(f"Error creating knowledge artifacts for '{corpus_name}': {e}")
                stats.errors_encountered += 1
    
    def rebuild_corpus_from_original(self,
                                   corpus_name: str,
                                   use_dictionary: bool = True,
                                   use_knowledge_graph: bool = False,
                                   dictionary_file_path: Optional[str] = None,
                                   graph_file_path: Optional[str] = None,
                                   additional_metadata: Optional[Dict[str, Any]] = None,
                                   stage_configs: Optional[Dict[str, Any]] = None) -> CorpusStats:
        """Rebuild corpus from existing original documents using existing knowledge artifacts
        
        既存のoriginalステージ文書から、既存の辞書・ナレッジグラフを利用してコーパスを再構築
        
        Args:
            corpus_name: Name of the corpus for metadata
                       メタデータ用のコーパス名
            use_dictionary: Whether to use existing dictionary for normalization
                          既存辞書を正規化に使用するか
            use_knowledge_graph: Whether to use existing knowledge graph for normalization
                               既存ナレッジグラフを正規化に使用するか
            dictionary_file_path: Path to existing dictionary file to use
                                既存の辞書ファイルパス
            graph_file_path: Path to existing knowledge graph file to use
                           既存のナレッジグラフファイルパス
            additional_metadata: Additional metadata to add during rebuild
                               再構築時に追加するメタデータ
            stage_configs: Configuration for each processing stage
                         各処理ステージの設定
            
        Returns:
            CorpusStats: Rebuild statistics
                        再構築統計
            
        Note:
            This method does NOT create new dictionary or knowledge graph files.
            It uses existing files for normalization if specified.
            このメソッドは新しい辞書やナレッジグラフファイルを作成しません。
            指定された既存ファイルを正規化に使用します。
            
        Example:
            # 基本的な再構築（既存辞書使用）
            stats = corpus_manager.rebuild_corpus_from_original(
                corpus_name="product_docs",
                use_dictionary=True,
                dictionary_file_path="./knowledge/product_docs_dictionary.md",
                additional_metadata={"rebuild_version": "2.0"}
            )
            
            # 辞書+ナレッジグラフ使用での再構築
            stats = corpus_manager.rebuild_corpus_from_original(
                corpus_name="engineering_docs", 
                use_dictionary=True,
                use_knowledge_graph=True,
                dictionary_file_path="./knowledge/engineering_docs_dictionary.md",
                graph_file_path="./knowledge/engineering_docs_graph.md",
                additional_metadata={
                    "rebuild_timestamp": datetime.now().isoformat(),
                    "rebuild_reason": "parameter_tuning"
                },
                stage_configs={
                    "normalizer_config": NormalizerConfig(
                        case_sensitive_replacement=True,
                        whole_word_only=False
                    ),
                    "chunker_config": ChunkingConfig(
                        chunk_size=1024,
                        overlap=100
                    )
                }
            )
        """
        from ..processing.document_pipeline import DocumentPipeline
        from ..processing.normalizer import Normalizer
        from ..processing.chunker import Chunker
        from datetime import datetime
        
        start_time = time.time()
        logger.info(f"Starting corpus rebuild for '{corpus_name}' from original documents")
        
        # Check if original documents exist
        original_docs = list(self._get_documents_by_stage("original"))
        if not original_docs:
            raise ValueError("No original documents found. Please import documents first using import_original_documents()")
        
        logger.info(f"Found {len(original_docs)} original documents to rebuild from")
        
        # Prepare metadata for rebuilt documents
        rebuild_metadata = {
            "rebuild_timestamp": datetime.now().isoformat(),
            "rebuild_corpus_name": corpus_name,
            "rebuilt_from": "original"
        }
        if additional_metadata:
            rebuild_metadata.update(additional_metadata)
        
        # Validate that knowledge files exist if specified
        if use_dictionary:
            if dictionary_file_path:
                dict_path = Path(dictionary_file_path)
                if not dict_path.exists():
                    raise FileNotFoundError(f"Dictionary file not found: {dictionary_file_path}")
                logger.info(f"Using existing dictionary: {dictionary_file_path}")
            else:
                # Try to find corpus-specific dictionary file using environment variables
                default_dict_path = self._get_corpus_file_path(corpus_name, "dictionary")
                if default_dict_path.exists():
                    dictionary_file_path = str(default_dict_path)
                    logger.info(f"Found corpus dictionary: {dictionary_file_path}")
                else:
                    logger.warning(f"No dictionary file specified and corpus dictionary not found: {default_dict_path}")
                    use_dictionary = False
        
        if use_knowledge_graph:
            if graph_file_path:
                graph_path = Path(graph_file_path)
                if not graph_path.exists():
                    raise FileNotFoundError(f"Knowledge graph file not found: {graph_file_path}")
                logger.info(f"Using existing knowledge graph: {graph_file_path}")
            else:
                # Try to find corpus-specific knowledge graph file using environment variables
                default_graph_path = self._get_corpus_file_path(corpus_name, "knowledge_graph")
                if default_graph_path.exists():
                    graph_file_path = str(default_graph_path)
                    logger.info(f"Found corpus knowledge graph: {graph_file_path}")
                else:
                    logger.warning(f"No graph file specified and corpus knowledge graph not found: {default_graph_path}")
                    use_knowledge_graph = False
        
        # Determine stages to execute based on options
        processors = []
        stage_configs = stage_configs or {}
        
        # Load original documents
        loader_config = DocumentLoadConfig(strategy=LoadStrategy.FILTERED, metadata_filters={"processing_stage": "original"})
        loader = DocumentStoreLoader(self.document_store, load_config=loader_config)
        processors.append(loader)
        
        # Add normalization stage if using dictionary or knowledge graph
        if use_dictionary or use_knowledge_graph:
            # Configure normalizer to use existing files
            if not stage_configs.get("normalizer_config"):
                normalizer_config = NormalizerConfig()
                
                # Set dictionary file path if using dictionary
                if use_dictionary and dictionary_file_path:
                    normalizer_config.dictionary_file_path = dictionary_file_path
                    normalizer_config.auto_detect_dictionary_path = False  # Use specified path
                    normalizer_config.skip_if_no_dictionary = False
                
                # Note: Normalizer currently only supports dictionary normalization
                # Knowledge graph integration would need to be implemented separately
                if use_knowledge_graph and graph_file_path:
                    logger.warning("Knowledge graph normalization not yet supported by Normalizer")
                
                stage_configs["normalizer_config"] = normalizer_config
            
            processors.append(Normalizer(stage_configs["normalizer_config"]))
        
        # Add chunking stage
        if not stage_configs.get("chunker_config"):
            stage_configs["chunker_config"] = ChunkingConfig()
        processors.append(Chunker(stage_configs["chunker_config"]))
        
        # Add vectorization stage
        processors.append(self.vector_store)
        
        try:
            # Execute rebuild pipeline
            pipeline = DocumentPipeline(processors)
            
            # Create trigger document
            trigger_doc = Document(
                id="rebuild_trigger",
                content="",
                metadata={
                    "trigger_type": "corpus_rebuild",
                    "corpus_name": corpus_name,
                    **rebuild_metadata
                }
            )
            
            logger.info(f"Executing rebuild pipeline with {len(processors)} processors")
            results = pipeline.process_document(trigger_doc)
            
            # Calculate statistics
            stats = CorpusStats()
            stats.total_documents_created = len(results)
            stats.pipeline_stages_executed = len(processors)
            
            # Count chunks (documents with chunk metadata)
            chunks = [doc for doc in results if doc.metadata.get("processing_stage") == "chunked"]
            stats.total_chunks_created = len(chunks)
            
            # Update timing
            total_time = time.time() - start_time
            stats.total_processing_time = total_time
            
            logger.info(f"Corpus rebuild completed in {total_time:.3f}s for '{corpus_name}': "
                       f"{stats.total_documents_created} documents processed, "
                       f"{stats.total_chunks_created} chunks created")
            
            # Log knowledge artifacts used
            if use_dictionary and dictionary_file_path:
                logger.info(f"Used dictionary: {dictionary_file_path}")
            
            if use_knowledge_graph and graph_file_path:
                logger.info(f"Used knowledge graph: {graph_file_path}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Corpus rebuild failed for '{corpus_name}': {e}")
            raise
    
    def _get_documents_by_stage(self, processing_stage: str) -> List[Document]:
        """Get documents by processing stage
        
        Args:
            processing_stage: Stage to filter by
            
        Returns:
            List of documents in the specified stage
        """
        loader = DocumentStoreLoader(self.document_store, 
                                   load_config=DocumentLoadConfig(strategy=LoadStrategy.FILTERED, 
                                                                 metadata_filters={"processing_stage": processing_stage}))
        
        # Create trigger document
        trigger = Document(id="stage_query", content="", metadata={})
        return loader.process(trigger)
    
    def clear_corpus(self):
        """Clear all documents from the corpus
        
        コーパスからすべての文書を削除
        
        Note:
            This method will remove all documents from both DocumentStore and VectorStore.
            このメソッドはDocumentStoreとVectorStoreの両方からすべての文書を削除します。
        """
        try:
            logger.info("Starting corpus clearing...")
            
            # Clear document store
            if hasattr(self.document_store, 'clear_all_documents'):
                self.document_store.clear_all_documents()
                logger.info("Cleared all documents from DocumentStore")
            else:
                logger.warning("DocumentStore does not support clear_all_documents method")
            
            # Clear vector store
            if hasattr(self.vector_store, 'clear_all_vectors'):
                self.vector_store.clear_all_vectors()
                logger.info("Cleared all vectors from VectorStore")
            else:
                logger.warning("VectorStore does not support clear_all_vectors method")
            
            # Reset stats
            self.stats = CorpusStats()
            
            logger.info("Corpus clearing completed successfully")
            
        except Exception as e:
            logger.error(f"Error clearing corpus: {e}")
            raise