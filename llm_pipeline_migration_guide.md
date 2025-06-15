# LLMPipeline Migration Guide

## Overview

Refinire has introduced a new **LLMPipeline** interface that replaces the old `get_llm` function. The LLMPipeline provides a more modern, feature-rich API with built-in evaluation, retry logic, tool support, and structured output capabilities.

## Key Differences

### Old Approach (get_llm)
```python
from refinire import get_llm

# Simple LLM call
llm = get_llm("gpt-4o-mini")
result = llm("Hello, world!")
```

### New Approach (LLMPipeline)
```python
from refinire import LLMPipeline

# Create a pipeline with instructions
pipeline = LLMPipeline(
    name="assistant",
    generation_instructions="You are a helpful assistant.",
    model="gpt-4o-mini"
)

# Run the pipeline
result = pipeline.run("Hello, world!")
print(result.content)  # Access the generated content
print(result.success)  # Check if generation was successful
```

## Basic Usage Examples

### 1. Simple Text Generation

```python
from refinire import create_simple_llm_pipeline

# Create a simple pipeline
pipeline = create_simple_llm_pipeline(
    name="text_generator",
    instructions="You are a creative writing assistant.",
    model="gpt-4o-mini"
)

# Generate text
result = pipeline.run("Write a short story about a robot.")
if result.success:
    print(result.content)
else:
    print(f"Generation failed: {result.metadata.get('error')}")
```

### 2. Pipeline with Evaluation

```python
from refinire import create_evaluated_llm_pipeline

# Create pipeline with evaluation
pipeline = create_evaluated_llm_pipeline(
    name="quality_writer",
    generation_instructions="Write high-quality technical documentation.",
    evaluation_instructions="Rate the clarity and technical accuracy of the documentation from 0-100.",
    model="gpt-4o-mini",
    threshold=80.0,  # Only accept results with score >= 80
    max_retries=3
)

# Generate with automatic evaluation
result = pipeline.run("Explain how REST APIs work.")
if result.success:
    print(f"Generated content (score: {result.evaluation_score}): {result.content}")
else:
    print("Failed to meet quality threshold")
```

### 3. Structured Output with Pydantic

```python
from refinire import LLMPipeline
from pydantic import BaseModel
from typing import List

# Define output structure
class PersonInfo(BaseModel):
    name: str
    age: int
    skills: List[str]

# Create pipeline with structured output
pipeline = LLMPipeline(
    name="person_extractor",
    generation_instructions="Extract person information from text and return as JSON.",
    model="gpt-4o-mini",
    output_model=PersonInfo
)

# Extract structured data
result = pipeline.run("John is 30 years old and skilled in Python, JavaScript, and Docker.")
if result.success and isinstance(result.content, PersonInfo):
    person = result.content
    print(f"Name: {person.name}, Age: {person.age}, Skills: {person.skills}")
```

### 4. Pipeline with Tools/Functions

```python
from refinire import create_tool_enabled_llm_pipeline

# Define tools
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 25°C"

def calculate(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)  # Note: Use safer eval in production

# Create pipeline with tools
pipeline = create_tool_enabled_llm_pipeline(
    name="assistant_with_tools",
    instructions="You are a helpful assistant with access to weather and calculator tools.",
    tools=[get_weather, calculate],
    model="gpt-4o-mini"
)

# Use tools
result = pipeline.run("What's the weather in Tokyo and what's 15 * 7?")
print(result.content)
```

### 5. Interactive Multi-Turn Conversations

```python
from refinire import create_simple_interactive_pipeline

# Define completion check
def is_conversation_complete(response: str) -> bool:
    """Check if conversation should end."""
    return "goodbye" in response.lower() or "bye" in response.lower()

# Create interactive pipeline
pipeline = create_simple_interactive_pipeline(
    name="chatbot",
    instructions="You are a friendly chatbot. Have a conversation with the user.",
    completion_check=is_conversation_complete,
    max_turns=10,
    model="gpt-4o-mini"
)

# Start conversation
result = pipeline.run_interactive("Hello! How are you?")

while not result.is_complete:
    if hasattr(result.content, 'question'):
        print(f"Bot: {result.content.question}")
        user_input = input("You: ")
        result = pipeline.continue_interaction(user_input)
    else:
        break

print(f"Conversation ended: {result.content}")
```

## Migration Strategy for Your Codebase

### Current Usage in DictionaryMaker and GraphBuilder

Your current code uses `get_llm` like this:
```python
from refinire import get_llm

# In __init__
self._llm_client = get_llm(self.config.llm_model)

# In processing method
response = self._llm_client(prompt)
```

### Migrated Version

Replace with LLMPipeline:
```python
from refinire import LLMPipeline

# In __init__
self._llm_pipeline = LLMPipeline(
    name=f"{self.__class__.__name__}_pipeline",
    generation_instructions="You are a domain expert assistant for extracting terms and relationships.",
    model=self.config.llm_model,
    temperature=self.config.llm_temperature,
    max_tokens=self.config.max_tokens
)

# In processing method
result = self._llm_pipeline.run(prompt)
if result.success:
    response = result.content
else:
    # Handle failure
    logger.error(f"LLM generation failed: {result.metadata.get('error')}")
    response = None
```

### Benefits of Migration

1. **Better Error Handling**: LLMResult provides success/failure status and error details
2. **Built-in Retry Logic**: Automatic retries with configurable max attempts
3. **Evaluation Support**: Optional quality scoring and threshold checking
4. **Structured Output**: Native Pydantic model support for JSON output
5. **Tool Integration**: Easy function calling and tool registration
6. **History Management**: Automatic conversation history tracking
7. **Guardrails**: Input/output validation support

### Configuration Options

The LLMPipeline constructor accepts many parameters for fine-tuning:

```python
pipeline = LLMPipeline(
    name="my_pipeline",
    generation_instructions="System prompt here",
    evaluation_instructions="Evaluation prompt here",  # Optional
    model="gpt-4o-mini",
    evaluation_model="gpt-4o-mini",  # Can be different model
    output_model=MyPydanticModel,  # Optional structured output
    temperature=0.7,
    max_tokens=2000,
    timeout=30.0,
    threshold=85.0,  # Evaluation threshold
    max_retries=3,
    input_guardrails=[my_input_validator],  # Optional validation
    output_guardrails=[my_output_validator],  # Optional validation
    session_history=[],  # Conversation history
    history_size=10,
    improvement_callback=my_improvement_function,  # Optional
    locale="en",
    tools=[],  # Function calling tools
    mcp_servers=[]  # MCP server configs
)
```

## Async Support

LLMPipeline also supports async execution:

```python
import asyncio
from refinire import LLMPipeline

async def main():
    pipeline = LLMPipeline(
        name="async_pipeline",
        generation_instructions="You are a helpful assistant.",
        model="gpt-4o-mini"
    )
    
    result = await pipeline.run_async("Hello, world!")
    print(result.content)

asyncio.run(main())
```

## Migration Checklist

- [ ] Replace `get_llm()` calls with `LLMPipeline` instances
- [ ] Update error handling to use `LLMResult.success` and `LLMResult.content`
- [ ] Consider adding evaluation instructions for quality control
- [ ] Add structured output models where appropriate
- [ ] Configure retry logic and timeouts
- [ ] Add input/output validation if needed
- [ ] Update tests to work with new API

## Utility Functions

Refinire provides several utility functions for common patterns:

- `create_simple_llm_pipeline()` - Basic pipeline
- `create_evaluated_llm_pipeline()` - Pipeline with evaluation
- `create_tool_enabled_llm_pipeline()` - Pipeline with function calling
- `create_web_search_pipeline()` - Pipeline with web search (template)
- `create_calculator_pipeline()` - Pipeline with calculation tools
- `create_simple_interactive_pipeline()` - Multi-turn conversations
- `create_evaluated_interactive_pipeline()` - Multi-turn with evaluation

These utility functions make it easy to create pipelines with common configurations.