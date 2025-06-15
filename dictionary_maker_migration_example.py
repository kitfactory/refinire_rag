"""
Example showing how to migrate DictionaryMaker from get_llm to LLMPipeline

This demonstrates the before/after code for migrating your existing DictionaryMaker
and GraphBuilder classes to use the new LLMPipeline interface.
"""

# BEFORE: Using get_llm (current approach)
class DictionaryMakerOld:
    def __init__(self, config):
        from refinire import get_llm
        
        # Old way - simple LLM client
        self._llm_client = get_llm(config.llm_model)
    
    def _extract_terms_with_llm(self, document_content: str, existing_dictionary: str) -> str:
        """Extract terms using old get_llm approach"""
        prompt = self._build_extraction_prompt(document_content, existing_dictionary)
        
        # Simple call, no error handling, retries, or evaluation
        response = self._llm_client(prompt)
        return response

# AFTER: Using LLMPipeline (new approach)
class DictionaryMakerNew:
    def __init__(self, config):
        from refinire import LLMPipeline
        
        # New way - feature-rich pipeline
        self._llm_pipeline = LLMPipeline(
            name="dictionary_maker",
            generation_instructions=self._get_generation_instructions(),
            evaluation_instructions=self._get_evaluation_instructions(),
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.max_tokens,
            threshold=80.0,  # Quality threshold
            max_retries=3,   # Automatic retries
            timeout=30.0
        )
    
    def _get_generation_instructions(self) -> str:
        """System instructions for term extraction"""
        return """You are a domain expert specialized in extracting technical terms and their variations from documents.

Your task is to:
1. Extract important domain-specific terms from the provided document
2. Identify expression variations and synonyms
3. Update the existing dictionary with new terms
4. Maintain consistent formatting

Focus on:
- Technical terminology
- Acronyms and abbreviations
- Key concepts and their definitions
- Expression variations (e.g., "machine learning" vs "ML")

Return the updated dictionary in Markdown format with clear definitions and examples."""

    def _get_evaluation_instructions(self) -> str:
        """Evaluation criteria for term extraction quality"""
        return """Evaluate the quality of the term extraction based on:

1. Completeness (70-100): Are all important technical terms identified?
2. Accuracy (70-100): Are definitions correct and precise?
3. Consistency (70-100): Is formatting consistent with existing dictionary?
4. Relevance (70-100): Are extracted terms truly domain-specific and important?

Provide a score from 0-100 and brief feedback on what could be improved."""

    def _extract_terms_with_llm(self, document_content: str, existing_dictionary: str) -> str:
        """Extract terms using new LLMPipeline approach"""
        prompt = self._build_extraction_prompt(document_content, existing_dictionary)
        
        # Rich pipeline execution with automatic retries and evaluation
        result = self._llm_pipeline.run(prompt)
        
        if result.success:
            logger.info(f"Term extraction successful (score: {result.evaluation_score})")
            return result.content
        else:
            logger.error(f"Term extraction failed: {result.metadata.get('error')}")
            logger.error(f"Failed after {result.attempts} attempts")
            return existing_dictionary  # Fallback to existing dictionary

# STRUCTURED OUTPUT VERSION: Using Pydantic models for better parsing
from pydantic import BaseModel, Field
from typing import List, Dict

class ExtractedTerm(BaseModel):
    """Single extracted term with metadata"""
    term: str = Field(description="The technical term")
    definition: str = Field(description="Clear definition of the term")
    variations: List[str] = Field(default=[], description="Alternative expressions or synonyms")
    category: str = Field(description="Category or domain area")
    importance: str = Field(description="Importance level: low, medium, high")
    examples: List[str] = Field(default=[], description="Usage examples")

class DictionaryUpdate(BaseModel):
    """Structured output for dictionary updates"""
    new_terms: List[ExtractedTerm] = Field(description="Newly identified terms")
    updated_terms: List[ExtractedTerm] = Field(description="Terms with updated definitions")
    removed_terms: List[str] = Field(default=[], description="Terms to remove (if any)")
    metadata: Dict[str, str] = Field(default={}, description="Additional metadata")

class DictionaryMakerStructured:
    def __init__(self, config):
        from refinire import LLMPipeline
        
        # Pipeline with structured output
        self._llm_pipeline = LLMPipeline(
            name="dictionary_maker_structured",
            generation_instructions=self._get_structured_instructions(),
            model=config.llm_model,
            output_model=DictionaryUpdate,  # Structured output
            temperature=config.llm_temperature,
            max_tokens=config.max_tokens,
            threshold=85.0,
            max_retries=3
        )
    
    def _get_structured_instructions(self) -> str:
        """Instructions for structured term extraction"""
        return """You are a domain expert for technical term extraction.

Analyze the provided document and existing dictionary to identify:
1. New technical terms not in the existing dictionary
2. Existing terms that need updated definitions
3. Terms that should be removed (rare)

For each term, provide:
- Clear, precise definition
- Alternative expressions/synonyms
- Category/domain area
- Importance level (low/medium/high)
- Usage examples where helpful

Return your analysis in the specified JSON format."""
    
    def _extract_terms_with_llm(self, document_content: str, existing_dictionary: str) -> DictionaryUpdate:
        """Extract terms with structured output"""
        prompt = f"""
Document to analyze:
{document_content}

Existing dictionary:
{existing_dictionary}

Please analyze and return the dictionary updates in the specified format.
"""
        
        result = self._llm_pipeline.run(prompt)
        
        if result.success and isinstance(result.content, DictionaryUpdate):
            logger.info(f"Structured extraction successful: {len(result.content.new_terms)} new terms")
            return result.content
        else:
            logger.error(f"Structured extraction failed: {result.metadata.get('error')}")
            # Return empty update
            return DictionaryUpdate(new_terms=[], updated_terms=[])

# TOOL-ENABLED VERSION: Pipeline with function calling capabilities
class DictionaryMakerWithTools:
    def __init__(self, config):
        from refinire import create_tool_enabled_llm_pipeline
        
        # Define tools for the pipeline
        tools = [
            self.search_existing_dictionary,
            self.validate_term_definition,
            self.check_term_importance
        ]
        
        # Pipeline with tools
        self._llm_pipeline = create_tool_enabled_llm_pipeline(
            name="dictionary_maker_tools",
            instructions=self._get_tool_instructions(),
            tools=tools,
            model=config.llm_model,
            temperature=config.llm_temperature
        )
        
        self.existing_dictionary = {}  # Load from file
    
    def _get_tool_instructions(self) -> str:
        """Instructions for tool-enabled extraction"""
        return """You are a domain expert with tools to help extract and validate technical terms.

Available tools:
- search_existing_dictionary: Search for existing terms
- validate_term_definition: Check if a definition is accurate
- check_term_importance: Assess importance of a term

Use these tools to:
1. Avoid duplicating existing terms
2. Ensure accurate definitions
3. Focus on important terms
4. Maintain consistency

Process the document systematically and return updated dictionary in Markdown format."""
    
    def search_existing_dictionary(self, search_term: str) -> str:
        """Search for a term in the existing dictionary"""
        # Implementation to search dictionary
        if search_term.lower() in self.existing_dictionary:
            return f"Found: {self.existing_dictionary[search_term.lower()]}"
        else:
            return f"Term '{search_term}' not found in existing dictionary"
    
    def validate_term_definition(self, term: str, definition: str) -> str:
        """Validate that a term definition is accurate"""
        # Implementation for validation logic
        # This could use external APIs, databases, or other validation methods
        return f"Definition for '{term}' appears accurate"
    
    def check_term_importance(self, term: str, context: str) -> str:
        """Assess the importance of a term in the given context"""
        # Implementation for importance scoring
        # Could use frequency analysis, domain knowledge, etc.
        return f"Term '{term}' has medium importance in this context"

# USAGE EXAMPLES

def example_basic_migration():
    """Example of basic migration from get_llm to LLMPipeline"""
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        llm_model: str = "gpt-4o-mini"
        llm_temperature: float = 0.3
        max_tokens: int = 2000
    
    config = Config()
    
    # Create new dictionary maker
    maker = DictionaryMakerNew(config)
    
    # Example usage
    document = "Machine learning (ML) is a subset of artificial intelligence (AI)..."
    existing_dict = "# Domain Dictionary\n\n## AI Terms\n..."
    
    # Extract terms with automatic retries and evaluation
    updated_dictionary = maker._extract_terms_with_llm(document, existing_dict)
    print("Updated dictionary:", updated_dictionary)

def example_structured_output():
    """Example using structured output with Pydantic models"""
    from dataclasses import dataclass
    
    @dataclass
    class Config:
        llm_model: str = "gpt-4o-mini"
        llm_temperature: float = 0.3
        max_tokens: int = 2000
    
    config = Config()
    maker = DictionaryMakerStructured(config)
    
    document = "Neural networks use backpropagation for training..."
    existing_dict = "# Terms\n- AI: Artificial Intelligence"
    
    # Get structured results
    update = maker._extract_terms_with_llm(document, existing_dict)
    
    # Process structured data
    for term in update.new_terms:
        print(f"New term: {term.term}")
        print(f"Definition: {term.definition}")
        print(f"Category: {term.category}")
        print(f"Variations: {term.variations}")
        print("---")

if __name__ == "__main__":
    example_basic_migration()
    print("\n" + "="*50 + "\n")
    example_structured_output()