"""
Example: Customizing AnswerSynthesizer Instructions

This example demonstrates how to customize the generation instructions
and system prompts for different use cases and domains.
"""

from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.base import SearchResult
from refinire_rag.models.document import Document


def example_customer_service_instructions():
    """Example: Customer service instructions"""
    
    customer_service_instructions = """You are a friendly and helpful customer service representative. When answering questions:

1. **Be Empathetic**: Always acknowledge the customer's concern or situation
2. **Use Simple Language**: Avoid technical jargon unless absolutely necessary
3. **Be Solution-Oriented**: Focus on how you can help resolve their issue
4. **Be Positive**: Maintain an upbeat and encouraging tone
5. **Offer Additional Help**: Always end by asking if there's anything else you can assist with

Guidelines:
- If you don't know something, admit it and offer to find out
- Provide step-by-step instructions when helpful
- Use "I understand" and "I'd be happy to help" type phrases
- Keep responses concise but thorough"""

    config = SimpleAnswerSynthesizerConfig(
        generation_instructions=customer_service_instructions,
        system_prompt=customer_service_instructions,  # Same for both Refinire and OpenAI
        temperature=0.7,  # More creative and conversational
        max_tokens=300
    )
    
    return SimpleAnswerSynthesizer(config)


def example_technical_documentation_instructions():
    """Example: Technical documentation instructions"""
    
    technical_instructions = """You are a technical documentation expert specializing in software engineering. When answering questions:

1. **Be Precise and Accurate**: Use exact technical terminology
2. **Provide Code Examples**: Include relevant code snippets when applicable
3. **Structure Clearly**: Use headings, bullet points, and numbered lists
4. **Include Context**: Explain why something works the way it does
5. **Reference Sources**: Mention specific documentation or standards when relevant

Response Format:
- Start with a brief summary
- Provide detailed explanation with examples
- Include any important warnings or considerations
- End with references or next steps

Technical Guidelines:
- Always prefer official documentation terms
- Include version numbers when relevant
- Explain both the "what" and the "why"
- Use code blocks for any code examples"""

    config = SimpleAnswerSynthesizerConfig(
        generation_instructions=technical_instructions,
        system_prompt=technical_instructions,
        temperature=0.2,  # More factual and consistent
        max_tokens=600,   # Longer for detailed explanations
        max_context_length=3000  # More context for technical details
    )
    
    return SimpleAnswerSynthesizer(config)


def example_academic_research_instructions():
    """Example: Academic research instructions"""
    
    academic_instructions = """You are an academic research assistant specializing in scientific literature. When answering questions:

1. **Academic Rigor**: Ensure all statements are evidence-based
2. **Formal Language**: Use precise, scholarly terminology
3. **Citations**: Reference specific studies, papers, or data when available
4. **Balanced Perspective**: Present multiple viewpoints when controversies exist
5. **Confidence Levels**: Indicate certainty levels for different claims

Response Structure:
- Executive summary of findings
- Detailed analysis with evidence
- Limitations of current knowledge
- Suggestions for further research

Academic Standards:
- Distinguish between established facts and current research
- Use formal academic language and tone
- Include methodological considerations when relevant
- Acknowledge uncertainty and gaps in knowledge"""

    config = SimpleAnswerSynthesizerConfig(
        generation_instructions=academic_instructions,
        system_prompt=academic_instructions,
        temperature=0.1,  # Very factual and consistent
        max_tokens=800,   # Longer for comprehensive academic responses
        max_context_length=4000  # Maximum context for thorough analysis
    )
    
    return SimpleAnswerSynthesizer(config)


def example_creative_writing_instructions():
    """Example: Creative writing instructions"""
    
    creative_instructions = """You are a creative writing assistant that helps with storytelling and narrative development. When responding:

1. **Be Imaginative**: Use vivid descriptions and creative language
2. **Tell Stories**: Frame responses as narratives when appropriate
3. **Use Analogies**: Make complex concepts relatable through metaphors
4. **Engage Emotions**: Connect with the reader on an emotional level
5. **Inspire Action**: Motivate and encourage creative thinking

Creative Elements:
- Use descriptive language and imagery
- Include relevant anecdotes or examples
- Vary sentence structure for rhythm
- Use active voice and dynamic verbs
- Create memorable phrases or insights

Tone Guidelines:
- Be inspiring and encouraging
- Use conversational, friendly language
- Include humor when appropriate
- Make abstract concepts tangible through stories"""

    config = SimpleAnswerSynthesizerConfig(
        generation_instructions=creative_instructions,
        system_prompt=creative_instructions,
        temperature=0.8,  # High creativity
        max_tokens=500,
        max_context_length=2500
    )
    
    return SimpleAnswerSynthesizer(config)


def example_legal_research_instructions():
    """Example: Legal research instructions"""
    
    legal_instructions = """You are a legal research assistant providing information about legal concepts. When responding:

1. **Legal Precision**: Use exact legal terminology and concepts
2. **Cite Authorities**: Reference relevant laws, cases, or regulations when available
3. **Distinguish Jurisdictions**: Specify which legal system applies
4. **Include Disclaimers**: Always note that this is not legal advice
5. **Structure Logically**: Use legal reasoning patterns

Legal Response Format:
- Issue identification
- Applicable law and precedents
- Analysis and application
- Conclusion with limitations

Important Disclaimers:
- This information is for educational purposes only
- Not a substitute for professional legal advice
- Laws vary by jurisdiction and change over time
- Consult qualified attorney for specific legal matters

Legal Writing Style:
- Use precise legal terminology
- Support arguments with authority
- Address counterarguments when relevant
- Maintain objective, analytical tone"""

    config = SimpleAnswerSynthesizerConfig(
        generation_instructions=legal_instructions,
        system_prompt=legal_instructions,
        temperature=0.15,  # Very precise and consistent
        max_tokens=700,
        max_context_length=3500
    )
    
    return SimpleAnswerSynthesizer(config)


def demonstrate_different_styles():
    """Demonstrate different instruction styles with the same query"""
    
    # Create sample context
    sample_doc = Document(
        id="ai_overview",
        content="""Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. 
        Machine learning is a subset of AI that enables computers to learn and improve from experience 
        without being explicitly programmed. Deep learning, a subset of machine learning, uses neural 
        networks with multiple layers to process data and make decisions.""",
        metadata={"source": "AI_handbook", "type": "educational"}
    )
    
    context = [SearchResult(
        document_id="ai_overview",
        document=sample_doc,
        score=0.95,
        metadata={"relevance": "high"}
    )]
    
    query = "What is artificial intelligence and how does it work?"
    
    # Different synthesizer styles
    styles = {
        "Customer Service": example_customer_service_instructions(),
        "Technical Documentation": example_technical_documentation_instructions(),
        "Academic Research": example_academic_research_instructions(),
        "Creative Writing": example_creative_writing_instructions(),
        "Legal Research": example_legal_research_instructions()
    }
    
    print("=== Demonstration: Same Query, Different Instruction Styles ===\n")
    print(f"Query: {query}\n")
    print("Context: AI overview document\n")
    print("-" * 70)
    
    for style_name, synthesizer in styles.items():
        print(f"\n{style_name.upper()} STYLE:")
        print(f"Instructions: {synthesizer.config.generation_instructions[:100]}...")
        print(f"Temperature: {synthesizer.config.temperature}")
        print(f"Max Tokens: {synthesizer.config.max_tokens}")
        
        # In a real scenario, you would call synthesizer.synthesize(query, context)
        # For this example, we just show the configuration
        print("✓ Synthesizer configured and ready")
        print("-" * 50)


def main():
    """Main example function"""
    print("AnswerSynthesizer Instruction Customization Examples")
    print("=" * 55)
    
    # Show different instruction styles
    demonstrate_different_styles()
    
    print("\n" + "=" * 55)
    print("Key Benefits of Custom Instructions:")
    print("• Domain-specific expertise and terminology")
    print("• Appropriate tone and style for different audiences")
    print("• Consistent response formatting and structure")
    print("• Customizable creativity vs. precision levels")
    print("• Specialized knowledge presentation patterns")
    
    print("\n" + "=" * 55)
    print("Usage Tips:")
    print("• Start with clear role definition ('You are a...')")
    print("• Include specific guidelines and examples")
    print("• Adjust temperature based on desired creativity")
    print("• Set max_tokens based on expected response length")
    print("• Test instructions with representative queries")
    print("• Consider different instructions for Refinire vs OpenAI if needed")


if __name__ == "__main__":
    main()