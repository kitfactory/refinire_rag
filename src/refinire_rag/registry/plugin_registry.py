"""
Universal plugin registry for refinire-rag

Provides unified access to both built-in components and external plugins
through entry points discovery and manual registration.
"""

import importlib.metadata
import logging
from typing import Dict, Type, List, Optional, Any

logger = logging.getLogger(__name__)

class PluginRegistry:
    """Universal registry for all plugin types
    
    すべてのプラグインタイプ用の統合レジストリ
    """
    
    _registries: Dict[str, Dict[str, Type]] = {}
    _discovered_groups: set = set()
    
    # Plugin group definitions
    PLUGIN_GROUPS = {
        'retrievers': 'refinire_rag.retrievers',
        'vector_stores': 'refinire_rag.vector_stores', 
        'keyword_stores': 'refinire_rag.keyword_stores',
        'rerankers': 'refinire_rag.rerankers',
        'synthesizers': 'refinire_rag.synthesizers',
        'evaluators': 'refinire_rag.evaluators',
        'embedders': 'refinire_rag.embedders',
        'loaders': 'refinire_rag.loaders',
        'processors': 'refinire_rag.processors',
        'splitters': 'refinire_rag.splitters',
        'filters': 'refinire_rag.filters',
        'metadata': 'refinire_rag.metadata',
        'document_stores': 'refinire_rag.document_stores',
        'evaluation_stores': 'refinire_rag.evaluation_stores',
        'contradiction_detectors': 'refinire_rag.contradiction_detectors',
        'test_suites': 'refinire_rag.test_suites',
        'insight_reporters': 'refinire_rag.insight_reporters',
        'reporters': 'refinire_rag.reporters',
        'oneenv_templates': 'refinire_rag.oneenv_templates'
    }
    
    # Built-in components registry
    BUILTIN_COMPONENTS = {
        'retrievers': {
            'simple': 'refinire_rag.retrieval.simple_retriever:SimpleRetriever',
            'hybrid': 'refinire_rag.retrieval.hybrid_retriever:HybridRetriever',
            'keyword': 'refinire_rag.retrieval.keyword_retriever:KeywordRetriever',
        },
        'vector_stores': {
            'inmemory_vector': 'refinire_rag.storage.in_memory_vector_store:InMemoryVectorStore',
            'pickle_vector': 'refinire_rag.storage.pickle_vector_store:PickleVectorStore',
            'openai_vector': 'refinire_rag.vectorstore.openai_vector_store:OpenAIVectorStore',
        },
        'keyword_stores': {
            'tfidf_keyword': 'refinire_rag.keywordstore.tfidf_keyword_store:TFIDFKeywordStore',
        },
        'document_stores': {
            'sqlite': 'refinire_rag.storage.sqlite_store:SQLiteDocumentStore',
            'document': 'refinire_rag.storage.document_store:DocumentStore',
        },
        'evaluation_stores': {
            'sqlite_evaluation': 'refinire_rag.storage.evaluation_store:SQLiteEvaluationStore',
        },
        'embedders': {
            'openai': 'refinire_rag.embedding.openai_embedder:OpenAIEmbedder',
            'tfidf': 'refinire_rag.embedding.tfidf_embedder:TFIDFEmbedder',
        },
        'evaluators': {
            'bleu': 'refinire_rag.evaluation.bleu_evaluator:BLEUEvaluator',
            'rouge': 'refinire_rag.evaluation.rouge_evaluator:ROUGEEvaluator',
            'llm_judge': 'refinire_rag.evaluation.llm_judge_evaluator:LLMJudgeEvaluator',
            'questeval': 'refinire_rag.evaluation.questeval_evaluator:QuestEvalEvaluator',
            'standard': 'refinire_rag.plugins.evaluators:StandardEvaluatorPlugin',
            'detailed': 'refinire_rag.plugins.evaluators:DetailedEvaluatorPlugin',
        },
        'rerankers': {
            'heuristic': 'refinire_rag.retrieval.heuristic_reranker:HeuristicReranker',
            'rrf': 'refinire_rag.retrieval.rrf_reranker:RRFReranker',
            'llm': 'refinire_rag.retrieval.llm_reranker:LLMReranker',
        },
        'synthesizers': {
            'answer': 'refinire_rag.retrieval.simple_reader:SimpleReader',
        },
        'loaders': {
            'text': 'refinire_rag.loader.text_loader:TextLoader',
            'csv': 'refinire_rag.loader.csv_loader:CSVLoader',
            'json': 'refinire_rag.loader.json_loader:JSONLoader',
            'html': 'refinire_rag.loader.html_loader:HTMLLoader',
            'directory': 'refinire_rag.loader.directory_loader:DirectoryLoader',
            'incremental_directory': 'refinire_rag.loader.incremental_directory_loader:IncrementalDirectoryLoader',
            'document_store': 'refinire_rag.loader.document_store_loader:DocumentStoreLoader',
        },
        'processors': {
            'normalizer': 'refinire_rag.processing.normalizer:Normalizer',
            'chunker': 'refinire_rag.processing.chunker:Chunker',
            'dictionary_maker': 'refinire_rag.processing.dictionary_maker:DictionaryMaker',
            'graph_builder': 'refinire_rag.processing.graph_builder:GraphBuilder',
            'evaluator': 'refinire_rag.processing.evaluator:Evaluator',
            'test_suite': 'refinire_rag.processing.test_suite:TestSuite',
            'contradiction_detector': 'refinire_rag.processing.contradiction_detector:ContradictionDetector',
            'insight_reporter': 'refinire_rag.processing.insight_reporter:InsightReporter',
        },
        'test_suites': {
            'llm': 'refinire_rag.plugins.test_suites:LLMTestSuitePlugin',
            'rule_based': 'refinire_rag.plugins.test_suites:RuleBasedTestSuitePlugin',
        },
        'contradiction_detectors': {
            'llm': 'refinire_rag.plugins.contradiction_detectors:LLMContradictionDetectorPlugin',
            'rule_based': 'refinire_rag.plugins.contradiction_detectors:RuleBasedContradictionDetectorPlugin',
            'hybrid': 'refinire_rag.plugins.contradiction_detectors:HybridContradictionDetectorPlugin',
        },
        'insight_reporters': {
            'standard': 'refinire_rag.plugins.insight_reporters:StandardInsightReporterPlugin',
            'executive': 'refinire_rag.plugins.insight_reporters:ExecutiveInsightReporterPlugin',
            'detailed': 'refinire_rag.plugins.insight_reporters:DetailedInsightReporterPlugin',
        },
        'splitters': {
            'character': 'refinire_rag.splitter.character_splitter:CharacterSplitter',
            'recursive_character': 'refinire_rag.splitter.recursive_character_splitter:RecursiveCharacterSplitter',
            'code': 'refinire_rag.splitter.code_splitter:CodeSplitter',
            'html': 'refinire_rag.splitter.html_splitter:HTMLSplitter',
            'markdown': 'refinire_rag.splitter.markdown_splitter:MarkdownSplitter',
            'size': 'refinire_rag.splitter.size_splitter:SizeSplitter',
            'token': 'refinire_rag.splitter.token_splitter:TokenSplitter',
        },
        'filters': {
            'extension': 'refinire_rag.loader.filters.extension_filter:ExtensionFilter',
            'size': 'refinire_rag.loader.filters.size_filter:SizeFilter',
            'date': 'refinire_rag.loader.filters.date_filter:DateFilter',
            'path': 'refinire_rag.loader.filters.path_filter:PathFilter',
        },
        'metadata': {
            'constant': 'refinire_rag.metadata.constant_metadata:ConstantMetadata',
            'file_info': 'refinire_rag.metadata.file_info_metadata:FileInfoMetadata',
            'path_map': 'refinire_rag.metadata.path_map_metadata:PathMapMetadata',
        }
    }
    
    @classmethod
    def _load_class_from_string(cls, class_string: str) -> Optional[Type]:
        """Load class from module:class string
        
        module:classの文字列からクラスをロード
        
        Args:
            class_string: String in format "module.path:ClassName"
        
        Returns:
            Class type or None if loading failed
        """
        try:
            module_path, class_name = class_string.split(':')
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except Exception as e:
            logger.debug(f"Failed to load built-in component {class_string}: {e}")
            return None
    
    @classmethod
    def _register_builtin_components(cls, group_name: str) -> None:
        """Register built-in components for a group
        
        グループの内蔵コンポーネントを登録
        """
        if group_name not in cls._registries:
            cls._registries[group_name] = {}
        
        builtin_components = cls.BUILTIN_COMPONENTS.get(group_name, {})
        for name, class_string in builtin_components.items():
            component_class = cls._load_class_from_string(class_string)
            if component_class:
                cls._registries[group_name][name] = component_class
                logger.debug(f"Registered built-in {group_name} component: {name}")
    
    @classmethod
    def discover_plugins(cls, group_name: str = None) -> None:
        """Discover plugins from entry points and register built-in components
        
        entry pointsからプラグインを発見し、内蔵コンポーネントを登録
        """
        groups_to_discover = [group_name] if group_name else cls.PLUGIN_GROUPS.keys()
        
        for group in groups_to_discover:
            if group in cls._discovered_groups:
                continue
                
            # Initialize group registry
            if group not in cls._registries:
                cls._registries[group] = {}
            
            # First, register built-in components
            cls._register_builtin_components(group)
            
            # Then discover external plugins via entry points
            entry_point_group = cls.PLUGIN_GROUPS.get(group)
            if entry_point_group:
                try:
                    for entry_point in importlib.metadata.entry_points(group=entry_point_group):
                        try:
                            plugin_class = entry_point.load()
                            # External plugins override built-in components with same name
                            cls._registries[group][entry_point.name] = plugin_class
                            logger.info(f"Discovered external {group} plugin: {entry_point.name}")
                        except Exception as e:
                            logger.warning(f"Failed to load external {group} plugin {entry_point.name}: {e}")
                            
                except Exception as e:
                    logger.debug(f"No external plugins found for {group}: {e}")
            
            cls._discovered_groups.add(group)
            total_components = len(cls._registries[group])
            logger.info(f"Registered {total_components} {group} components (built-in + external)")
    
    @classmethod
    def get_plugin_class(cls, group: str, name: str) -> Optional[Type]:
        """Get plugin class by group and name
        
        グループと名前でプラグインクラスを取得
        """
        cls.discover_plugins(group)
        return cls._registries.get(group, {}).get(name)
    
    @classmethod
    def list_available_plugins(cls, group: str) -> List[str]:
        """List all available plugin names for a group
        
        グループの利用可能なプラグイン名のリストを取得
        """
        cls.discover_plugins(group)
        return list(cls._registries.get(group, {}).keys())
    
    @classmethod
    def create_plugin(cls, group: str, name: str, **kwargs) -> Any:
        """Create plugin instance by group and name
        
        グループと名前でプラグインインスタンスを作成
        """
        plugin_class = cls.get_plugin_class(group, name)
        if plugin_class is None:
            available = cls.list_available_plugins(group)
            raise ValueError(f"Unknown {group} plugin: {name}. Available: {available}")
        
        return plugin_class(**kwargs)
    
    @classmethod
    def is_builtin(cls, group: str, name: str) -> bool:
        """Check if a component is built-in
        
        コンポーネントが内蔵かどうかをチェック
        """
        return name in cls.BUILTIN_COMPONENTS.get(group, {})
    
    @classmethod
    def get_all_plugins_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all available plugins
        
        すべての利用可能なプラグインの包括的な情報を取得
        """
        # Discover all plugin groups
        for group in cls.PLUGIN_GROUPS.keys():
            cls.discover_plugins(group)
        
        info = {}
        for group, plugins in cls._registries.items():
            info[group] = {}
            for name, plugin_class in plugins.items():
                info[group][name] = {
                    "class": plugin_class.__name__,
                    "module": plugin_class.__module__,
                    "description": getattr(plugin_class, "__doc__", "No description"),
                    "version": getattr(plugin_class, "__version__", "Unknown"),
                    "builtin": cls.is_builtin(group, name)
                }
        
        return info
    
    @classmethod
    def list_builtin_components(cls, group: str = None) -> Dict[str, List[str]]:
        """List built-in components by group
        
        グループ別の内蔵コンポーネントを一覧表示
        """
        if group:
            return {group: list(cls.BUILTIN_COMPONENTS.get(group, {}).keys())}
        else:
            return {g: list(components.keys()) for g, components in cls.BUILTIN_COMPONENTS.items()}
    
    @classmethod
    def register_builtin_component(cls, group: str, name: str, component_class: Type) -> None:
        """Manually register a built-in component
        
        内蔵コンポーネントを手動登録
        """
        if group not in cls._registries:
            cls._registries[group] = {}
        
        cls._registries[group][name] = component_class
        logger.info(f"Manually registered built-in {group} component: {name}")