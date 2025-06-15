"""
GraphBuilder - Knowledge graph cumulative construction processor

A DocumentProcessor that uses LLM to extract important relationships from documents
and maintains a cumulative Markdown knowledge graph across all processed documents.
"""

import logging
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Type
from pathlib import Path

from ..document_processor import DocumentProcessor, DocumentProcessorConfig
from ..models.document import Document
from ..utils.model_config import get_default_llm_model

try:
    from refinire import LLMPipeline
except ImportError:
    LLMPipeline = None

logger = logging.getLogger(__name__)


@dataclass
class GraphBuilderConfig(DocumentProcessorConfig):
    """Configuration for GraphBuilder processor"""
    
    # Graph file settings
    graph_file_path: str = "./domain_knowledge_graph.md"
    dictionary_file_path: str = "./domain_dictionary.md"
    backup_graph: bool = True
    
    # LLM settings
    llm_model: str = None  # Will be set to default in __post_init__
    llm_temperature: float = 0.3
    max_tokens: int = 3000
    
    # Relationship extraction settings
    focus_on_important_relationships: bool = True
    extract_hierarchical_relationships: bool = True
    extract_causal_relationships: bool = True
    extract_composition_relationships: bool = True
    min_relationship_importance: str = "medium"  # "low", "medium", "high"
    
    # Processing settings
    use_dictionary_terms: bool = True
    auto_detect_dictionary_path: bool = True
    skip_if_no_new_relationships: bool = False
    validate_extracted_relationships: bool = True
    deduplicate_relationships: bool = True
    
    # Output settings
    update_document_metadata: bool = True
    preserve_original_document: bool = True

    def __post_init__(self):
        """Initialize default values"""
        # Set default LLM model from environment variables if not specified
        if self.llm_model is None:
            self.llm_model = get_default_llm_model()
        
        # Ensure file paths are absolute
        if not os.path.isabs(self.graph_file_path):
            self.graph_file_path = os.path.abspath(self.graph_file_path)
        if self.dictionary_file_path and not os.path.isabs(self.dictionary_file_path):
            self.dictionary_file_path = os.path.abspath(self.dictionary_file_path)


class GraphBuilder(DocumentProcessor):
    """Processor that builds cumulative knowledge graph using LLM
    
    This processor:
    1. Reads existing Markdown knowledge graph file
    2. Reads domain dictionary for term context
    3. Sends document + existing graph + dictionary to LLM
    4. Extracts important domain-specific relationships
    5. Updates the cumulative Markdown knowledge graph
    6. Maintains one graph across all processed documents
    """
    
    def __init__(self, config: Optional[GraphBuilderConfig] = None):
        """Initialize GraphBuilder processor
        
        Args:
            config: Configuration for the processor
        """
        super().__init__(config or GraphBuilderConfig())
        
        # Initialize Refinire LLM Pipeline
        if LLMPipeline is not None:
            try:
                self._llm_pipeline = LLMPipeline(
                    name="graph_builder",
                    generation_instructions="You are a knowledge graph expert that extracts important relationships from documents.",
                    model=self.config.llm_model
                )
                logger.info(f"Initialized Refinire LLMPipeline with model: {self.config.llm_model}")
            except Exception as e:
                self._llm_pipeline = None
                logger.warning(f"Failed to initialize Refinire LLMPipeline: {e}. GraphBuilder will use mock data.")
        else:
            self._llm_pipeline = None
            logger.warning("Refinire not available. GraphBuilder will use mock data.")
        
        # Processing statistics
        self.processing_stats.update({
            "documents_processed": 0,
            "relationships_extracted": 0,
            "graph_updates": 0,
            "llm_api_calls": 0,
            "duplicate_relationships_avoided": 0
        })
        
        logger.info(f"Initialized GraphBuilder with graph: {self.config.graph_file_path}")
    
    @classmethod
    def get_config_class(cls) -> Type[GraphBuilderConfig]:
        """Get the configuration class for this processor"""
        return GraphBuilderConfig
    
    def process(self, document: Document, config: Optional[GraphBuilderConfig] = None) -> List[Document]:
        """Process document to extract relationships and update knowledge graph
        
        Args:
            document: Input document to analyze for relationships
            config: Optional configuration override
            
        Returns:
            List containing the original document (optionally with graph metadata)
        """
        try:
            # Use provided config or fall back to instance config
            graph_config = config or self.config
            
            logger.debug(f"Processing document {document.id} with GraphBuilder")
            
            # Read existing graph
            existing_graph = self._read_existing_graph(graph_config)
            
            # Read dictionary for context
            dictionary_content = self._read_dictionary_content(document, graph_config)
            
            # Extract relationships using LLM
            extracted_data = self._extract_relationships_with_llm(
                document, existing_graph, dictionary_content, graph_config
            )
            
            if not extracted_data or not extracted_data.get("has_new_relationships", False):
                if graph_config.skip_if_no_new_relationships:
                    logger.debug(f"No new relationships found for document {document.id}, skipping")
                    self.processing_stats["documents_processed"] += 1
                    return [document]
            
            # Update graph file
            updated_graph = self._update_graph_file(existing_graph, extracted_data, graph_config)
            
            # Update statistics
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["relationships_extracted"] += extracted_data.get("new_relationships_count", 0)
            self.processing_stats["graph_updates"] += 1
            self.processing_stats["llm_api_calls"] += 1
            self.processing_stats["duplicate_relationships_avoided"] += extracted_data.get("duplicates_avoided", 0)
            
            # Create output document
            if graph_config.update_document_metadata:
                enriched_document = self._add_graph_metadata(document, extracted_data, graph_config)
                return [enriched_document]
            else:
                return [document]
            
        except Exception as e:
            logger.error(f"Error in GraphBuilder for document {document.id}: {e}")
            return [document]  # Return original on error
    
    def _read_existing_graph(self, config: GraphBuilderConfig) -> str:
        """Read existing Markdown knowledge graph file
        
        Args:
            config: Configuration containing graph file path
            
        Returns:
            Content of existing graph file, or empty template if file doesn't exist
        """
        try:
            graph_path = Path(config.graph_file_path)
            
            if graph_path.exists():
                with open(graph_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.debug(f"Read existing knowledge graph: {len(content)} characters")
                return content
            else:
                # Create empty graph template
                template = self._create_empty_graph_template()
                logger.info(f"Creating new knowledge graph file at: {graph_path}")
                
                # Ensure directory exists
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create initial file
                with open(graph_path, 'w', encoding='utf-8') as f:
                    f.write(template)
                
                return template
                
        except Exception as e:
            logger.error(f"Error reading knowledge graph file: {e}")
            return self._create_empty_graph_template()
    
    def _create_empty_graph_template(self) -> str:
        """Create empty knowledge graph template"""
        return """# ドメイン知識グラフ

## エンティティ関係

### 主要概念

### 技術関係

### 機能関係

### 評価関係

---
*このナレッジグラフは自動生成され、処理された文書から継続的に更新されます。*
"""
    
    def _read_dictionary_content(self, document: Document, config: GraphBuilderConfig) -> str:
        """Read dictionary content for relationship extraction context
        
        Args:
            document: Document being processed (may contain dictionary metadata)
            config: Configuration
            
        Returns:
            Dictionary content or empty string if not available
        """
        # Try to get dictionary path from document metadata
        if config.auto_detect_dictionary_path:
            dict_metadata = document.metadata.get("dictionary_metadata", {})
            if dict_metadata and "dictionary_file_path" in dict_metadata:
                auto_path = dict_metadata["dictionary_file_path"]
                if Path(auto_path).exists():
                    try:
                        with open(auto_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        logger.debug(f"Read dictionary from document metadata: {len(content)} characters")
                        return content
                    except Exception as e:
                        logger.warning(f"Failed to read dictionary from metadata path: {e}")
        
        # Fall back to configured dictionary path
        if config.dictionary_file_path and Path(config.dictionary_file_path).exists():
            try:
                with open(config.dictionary_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.debug(f"Read dictionary from config path: {len(content)} characters")
                return content
            except Exception as e:
                logger.warning(f"Failed to read dictionary from config path: {e}")
        
        return ""
    
    def _extract_relationships_with_llm(self, document: Document, existing_graph: str,
                                       dictionary_content: str, config: GraphBuilderConfig) -> Dict:
        """Extract relationships using LLM
        
        Args:
            document: Document to analyze
            existing_graph: Current graph content
            dictionary_content: Dictionary content for context
            config: Configuration
            
        Returns:
            Dictionary containing extracted relationships and metadata
        """
        # Create prompt for LLM
        prompt = self._create_relationship_extraction_prompt(
            document, existing_graph, dictionary_content, config
        )
        
        # Use Refinire LLM Pipeline for relationship extraction
        if self._llm_pipeline is not None:
            try:
                result = self._llm_pipeline.run(prompt)
                response = result.content
                
                # Parse JSON response
                try:
                    # Clean response - remove markdown code block markers if present
                    clean_response = response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    
                    extracted_data = json.loads(clean_response)
                    logger.info(f"Extracted {extracted_data.get('new_relationships_count', 0)} relationships from document {document.id}")
                    return extracted_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.debug(f"Raw LLM response: {response}")
                    return {"has_new_relationships": False, "new_relationships_count": 0, "duplicates_avoided": 0, "extracted_relationships": []}
                    
            except Exception as e:
                logger.error(f"LLM call failed for document {document.id}: {e}")
                return {"has_new_relationships": False, "new_relationships_count": 0, "duplicates_avoided": 0, "extracted_relationships": []}
        else:
            # Fallback to mock data when Refinire is not available
            logger.info(f"Using mock data for document {document.id} (Refinire LLM not available)")
            mock_data = {
                "has_new_relationships": True,
                "new_relationships_count": 4,
                "duplicates_avoided": 1,
                "extracted_relationships": [
                    {
                        "subject": "RAG システム",
                        "predicate": "含む",
                        "object": "検索機能",
                        "category": "機能関係",
                        "importance": "high"
                    },
                    {
                        "subject": "RAG システム", 
                        "predicate": "含む",
                        "object": "生成機能",
                        "category": "機能関係",
                        "importance": "high"
                    },
                    {
                        "subject": "ベクトル検索",
                        "predicate": "基づく",
                        "object": "類似度計算",
                        "category": "技術関係",
                        "importance": "medium"
                    },
                    {
                        "subject": "評価システム",
                        "predicate": "計算する",
                        "object": "精度メトリクス",
                        "category": "評価関係",
                        "importance": "medium"
                    }
                ],
                "reasoning": "Document contains important relationships between RAG components and evaluation metrics that should be added to domain knowledge graph."
            }
            return mock_data
    
    def _create_relationship_extraction_prompt(self, document: Document, existing_graph: str,
                                             dictionary_content: str, config: GraphBuilderConfig) -> str:
        """Create prompt for LLM relationship extraction"""
        
        prompt = f"""以下のドメイン文書から、重要な関係性を抽出して知識グラフを更新してください。

## 既存の知識グラフ
{existing_graph}

## ドメイン用語辞書（参考）
{dictionary_content}

## 分析対象文書
タイトル: {document.metadata.get('title', 'Unknown')}
内容:
{document.content[:2500]}  # Limit content to avoid token limits

## 抽出指示
1. 文書から重要なエンティティ間の関係を特定
2. 既存グラフとの重複を避ける
3. ドメインに特化した重要な関係のみを抽出
4. 関係は主語-述語-目的語の形式で表現

## 関係の種類
- 機能関係: システムが何を含む/提供するか
- 技術関係: 技術的な依存関係や基盤
- 評価関係: 評価や測定に関する関係
- 主要概念: 概念間の分類や階層関係

## 出力形式
以下のJSON形式で回答してください:
{{
    "has_new_relationships": true/false,
    "new_relationships_count": 数値,
    "duplicates_avoided": 数値,
    "extracted_relationships": [
        {{
            "subject": "主語エンティティ",
            "predicate": "述語関係",
            "object": "目的語エンティティ",
            "category": "機能関係/技術関係/評価関係/主要概念",
            "importance": "high/medium/low"
        }}
    ],
    "reasoning": "抽出理由の説明"
}}

重要度レベル: {config.min_relationship_importance}
重要関係重視: {config.focus_on_important_relationships}
階層関係抽出: {config.extract_hierarchical_relationships}
"""
        
        return prompt
    
    def _update_graph_file(self, existing_graph: str, extracted_data: Dict,
                          config: GraphBuilderConfig) -> str:
        """Update knowledge graph file with extracted relationships
        
        Args:
            existing_graph: Current graph content
            extracted_data: Data extracted by LLM
            config: Configuration
            
        Returns:
            Updated graph content
        """
        try:
            # Backup existing graph if configured
            if config.backup_graph:
                self._backup_graph_file(config)
            
            # Parse and update graph
            updated_graph = self._merge_relationships_into_graph(existing_graph, extracted_data)
            
            # Write updated graph
            graph_path = Path(config.graph_file_path)
            with open(graph_path, 'w', encoding='utf-8') as f:
                f.write(updated_graph)
            
            logger.info(f"Updated knowledge graph with {extracted_data.get('new_relationships_count', 0)} new relationships")
            return updated_graph
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph file: {e}")
            return existing_graph
    
    def _backup_graph_file(self, config: GraphBuilderConfig):
        """Create backup of knowledge graph file"""
        try:
            graph_path = Path(config.graph_file_path)
            if graph_path.exists():
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = graph_path.with_suffix(f".backup_{timestamp}.md")
                
                import shutil
                shutil.copy2(graph_path, backup_path)
                logger.debug(f"Created knowledge graph backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create knowledge graph backup: {e}")
    
    def _merge_relationships_into_graph(self, existing_graph: str, extracted_data: Dict) -> str:
        """Merge extracted relationships into existing knowledge graph
        
        Args:
            existing_graph: Current graph content
            extracted_data: Extracted relationships data
            
        Returns:
            Updated graph content
        """
        # Simple implementation - in practice would need sophisticated parsing and merging
        lines = existing_graph.split('\n')
        
        # Find insertion points for different categories
        category_indices = {}
        for i, line in enumerate(lines):
            if line.strip() == "### 主要概念":
                category_indices["主要概念"] = i
            elif line.strip() == "### 技術関係":
                category_indices["技術関係"] = i
            elif line.strip() == "### 機能関係":
                category_indices["機能関係"] = i
            elif line.strip() == "### 評価関係":
                category_indices["評価関係"] = i
        
        # Group relationships by category
        relationships_by_category = {}
        for rel_data in extracted_data.get("extracted_relationships", []):
            category = rel_data.get("category", "主要概念")
            if category not in relationships_by_category:
                relationships_by_category[category] = []
            relationships_by_category[category].append(rel_data)
        
        # Add relationships to appropriate sections
        for category, relationships in relationships_by_category.items():
            if category in category_indices:
                insert_idx = category_indices[category] + 1
                
                # Add relationships under this category
                for rel_data in relationships:
                    rel_entry = self._format_relationship_entry(rel_data)
                    lines.insert(insert_idx, rel_entry)
                    insert_idx += 1
                    
                    # Update other indices
                    for cat, idx in category_indices.items():
                        if idx >= insert_idx - 1:
                            category_indices[cat] += 1
        
        return '\n'.join(lines)
    
    def _format_relationship_entry(self, rel_data: Dict) -> str:
        """Format relationship data as Markdown entry"""
        subject = rel_data.get("subject", "")
        predicate = rel_data.get("predicate", "")
        obj = rel_data.get("object", "")
        importance = rel_data.get("importance", "medium")
        
        # Format with importance indicator
        importance_marker = "**" if importance == "high" else "*" if importance == "medium" else ""
        
        if importance_marker:
            entry = f"- {importance_marker}{subject} → {predicate} → {obj}{importance_marker}"
        else:
            entry = f"- {subject} → {predicate} → {obj}"
        
        return entry
    
    def _add_graph_metadata(self, document: Document, extracted_data: Dict,
                           config: GraphBuilderConfig) -> Document:
        """Add graph extraction metadata to document"""
        
        graph_metadata = {
            "graph_extraction_applied": True,
            "new_relationships_extracted": extracted_data.get("new_relationships_count", 0),
            "duplicates_avoided": extracted_data.get("duplicates_avoided", 0),
            "extraction_reasoning": extracted_data.get("reasoning", ""),
            "graph_file_path": config.graph_file_path,
            "extracted_by": "GraphBuilder"
        }
        
        enriched_metadata = {
            **document.metadata,
            "graph_metadata": graph_metadata,
            "processing_stage": "graph_analyzed"
        }
        
        enriched_document = Document(
            id=document.id,
            content=document.content,
            metadata=enriched_metadata
        )
        
        return enriched_document
    
    def get_graph_path(self) -> str:
        """Get current knowledge graph file path"""
        return self.config.graph_file_path
    
    def get_graph_content(self) -> str:
        """Get current knowledge graph content"""
        return self._read_existing_graph(self.config)
    
    def get_graph_stats(self) -> Dict:
        """Get knowledge graph construction statistics"""
        return {
            **self.get_processing_stats(),
            "graph_file": self.config.graph_file_path,
            "dictionary_file": self.config.dictionary_file_path,
            "llm_model": self.config.llm_model,
            "focus_important_relationships": self.config.focus_on_important_relationships
        }