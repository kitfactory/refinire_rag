#!/usr/bin/env python3
"""
Interactive test script for LLMPipeline functionality

This script demonstrates various LLMPipeline features and allows you to test
the new interface interactively.

Run with: python test_llm_pipeline.py
"""

import asyncio
import sys
from typing import List
from pydantic import BaseModel, Field

def test_basic_pipeline():
    """Test basic LLMPipeline functionality"""
    print("🧪 Testing Basic LLMPipeline...")
    
    try:
        from refinire import LLMPipeline
        
        # Create a simple pipeline
        pipeline = LLMPipeline(
            name="test_basic",
            generation_instructions="You are a helpful assistant. Keep responses concise and friendly.",
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        # Test basic generation
        test_input = "Explain what Python is in one sentence."
        result = pipeline.run(test_input)
        
        print(f"✅ Input: {test_input}")
        print(f"✅ Success: {result.success}")
        print(f"✅ Output: {result.content}")
        print(f"✅ Attempts: {result.attempts}")
        print(f"✅ Model: {result.metadata.get('model')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic pipeline test failed: {e}")
        return False

def test_evaluated_pipeline():
    """Test pipeline with evaluation"""
    print("\n🧪 Testing Evaluated LLMPipeline...")
    
    try:
        from refinire import create_evaluated_llm_pipeline
        
        # Create pipeline with evaluation
        pipeline = create_evaluated_llm_pipeline(
            name="test_evaluated",
            generation_instructions="Write clear, accurate technical explanations.",
            evaluation_instructions="""Rate the technical accuracy and clarity of the explanation from 0-100.
            
            Criteria:
            - Technical accuracy (0-40 points)
            - Clarity and readability (0-40 points)  
            - Completeness (0-20 points)
            
            Return a JSON response with 'score' and 'feedback' fields.""",
            model="gpt-4o-mini",
            threshold=75.0,  # Require score >= 75
            max_retries=2
        )
        
        # Test with evaluation
        test_input = "How does HTTP work?"
        result = pipeline.run(test_input)
        
        print(f"✅ Input: {test_input}")
        print(f"✅ Success: {result.success}")
        print(f"✅ Evaluation Score: {result.evaluation_score}")
        print(f"✅ Attempts: {result.attempts}")
        print(f"✅ Output: {result.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluated pipeline test failed: {e}")
        return False

def test_structured_output():
    """Test pipeline with structured output"""
    print("\n🧪 Testing Structured Output...")
    
    try:
        from refinire import LLMPipeline
        
        # Define output structure
        class PersonInfo(BaseModel):
            name: str = Field(description="Person's full name")
            age: int = Field(description="Person's age")
            occupation: str = Field(description="Person's job or profession")
            skills: List[str] = Field(description="List of skills or expertise")
        
        # Create pipeline with structured output
        pipeline = LLMPipeline(
            name="test_structured",
            generation_instructions="""Extract person information from the text and return as JSON.
            
            Follow the provided schema exactly. If information is missing, use reasonable defaults:
            - age: 0 if not specified
            - occupation: "Unknown" if not specified
            - skills: empty list if not specified""",
            model="gpt-4o-mini",
            output_model=PersonInfo
        )
        
        # Test structured extraction
        test_input = "Sarah Johnson is a 28-year-old software engineer who specializes in Python, React, and DevOps practices."
        result = pipeline.run(test_input)
        
        print(f"✅ Input: {test_input}")
        print(f"✅ Success: {result.success}")
        
        if result.success and isinstance(result.content, PersonInfo):
            person = result.content
            print(f"✅ Structured Output:")
            print(f"   - Name: {person.name}")
            print(f"   - Age: {person.age}")
            print(f"   - Occupation: {person.occupation}")
            print(f"   - Skills: {person.skills}")
        else:
            print(f"✅ Raw Output: {result.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Structured output test failed: {e}")
        return False

def test_tools_pipeline():
    """Test pipeline with function calling"""
    print("\n🧪 Testing Pipeline with Tools...")
    
    try:
        from refinire import create_tool_enabled_llm_pipeline
        
        # Define tools
        def get_weather(city: str) -> str:
            """Get current weather for a city"""
            # Mock weather data
            weather_data = {
                "tokyo": "Sunny, 22°C",
                "london": "Cloudy, 15°C", 
                "new york": "Rainy, 18°C",
                "paris": "Partly cloudy, 20°C"
            }
            return weather_data.get(city.lower(), f"Weather data not available for {city}")
        
        def calculate(expression: str) -> str:
            """Calculate a mathematical expression safely"""
            try:
                # Simple safe evaluation for basic math
                import ast
                import operator
                
                operators = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                }
                
                def eval_expr(expr):
                    if isinstance(expr, ast.Num):
                        return expr.n
                    elif isinstance(expr, ast.Constant):
                        return expr.value
                    elif isinstance(expr, ast.BinOp):
                        return operators[type(expr.op)](eval_expr(expr.left), eval_expr(expr.right))
                    else:
                        raise TypeError(f"Unsupported operation: {type(expr)}")
                
                tree = ast.parse(expression, mode='eval')
                result = eval_expr(tree.body)
                return str(result)
                
            except Exception as e:
                return f"Error calculating '{expression}': {str(e)}"
        
        # Create pipeline with tools
        pipeline = create_tool_enabled_llm_pipeline(
            name="test_tools",
            instructions="""You are a helpful assistant with access to weather and calculator tools.
            
            Use the available tools when needed:
            - get_weather: Get weather information for cities
            - calculate: Perform mathematical calculations
            
            Always use tools for weather queries and math operations.""",
            tools=[get_weather, calculate],
            model="gpt-4o-mini"
        )
        
        # Test tool usage
        test_input = "What's the weather in Tokyo and what's 15 * 7?"
        result = pipeline.run(test_input)
        
        print(f"✅ Input: {test_input}")
        print(f"✅ Success: {result.success}")
        print(f"✅ Output: {result.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tools pipeline test failed: {e}")
        return False

def test_interactive_pipeline():
    """Test interactive multi-turn conversation"""
    print("\n🧪 Testing Interactive Pipeline...")
    
    try:
        from refinire import create_simple_interactive_pipeline
        
        # Define completion check
        def is_complete(response: str) -> bool:
            """Check if conversation should end"""
            end_phrases = ["goodbye", "bye", "end conversation", "thanks, that's all"]
            return any(phrase in response.lower() for phrase in end_phrases)
        
        # Create interactive pipeline
        pipeline = create_simple_interactive_pipeline(
            name="test_interactive",
            instructions="""You are a friendly customer service assistant.
            
            Help users with their questions. When they seem ready to end the conversation
            (saying goodbye, thanks, etc.), politely end with a farewell message that includes
            one of these phrases: "goodbye", "bye", or "thanks, that's all".""",
            completion_check=is_complete,
            max_turns=5,
            model="gpt-4o-mini"
        )
        
        # Simulate conversation
        conversations = [
            "Hello, I need help with my account",
            "I forgot my password",
            "Thanks, that helps. Goodbye!"
        ]
        
        result = pipeline.run_interactive(conversations[0])
        print(f"✅ Turn 1 - User: {conversations[0]}")
        
        for i, user_input in enumerate(conversations[1:], 2):
            if result.is_complete:
                break
                
            if hasattr(result.content, 'question'):
                print(f"✅ Turn {i-1} - Bot: {result.content.question}")
            else:
                print(f"✅ Turn {i-1} - Bot: {result.content}")
            
            print(f"✅ Turn {i} - User: {user_input}")
            result = pipeline.continue_interaction(user_input)
        
        print(f"✅ Final - Bot: {result.content}")
        print(f"✅ Conversation Complete: {result.is_complete}")
        print(f"✅ Total Turns: {result.turn}")
        
        return True
        
    except Exception as e:
        print(f"❌ Interactive pipeline test failed: {e}")
        return False

async def test_async_pipeline():
    """Test async pipeline execution"""
    print("\n🧪 Testing Async Pipeline...")
    
    try:
        from refinire import LLMPipeline
        
        # Create pipeline
        pipeline = LLMPipeline(
            name="test_async",
            generation_instructions="You are a helpful assistant. Be concise.",
            model="gpt-4o-mini"
        )
        
        # Test async execution
        test_input = "What is async programming?"
        result = await pipeline.run_async(test_input)
        
        print(f"✅ Input: {test_input}")
        print(f"✅ Success: {result.success}")
        print(f"✅ Output: {result.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Async pipeline test failed: {e}")
        return False

def test_utility_functions():
    """Test utility function alternatives"""
    print("\n🧪 Testing Utility Functions...")
    
    try:
        from refinire import (
            create_simple_llm_pipeline,
            create_calculator_pipeline
        )
        
        # Test simple pipeline creation
        simple = create_simple_llm_pipeline(
            name="test_simple_util",
            instructions="You are a helpful assistant.",
            model="gpt-4o-mini"
        )
        
        result1 = simple.run("Say hello")
        print(f"✅ Simple pipeline: {result1.content}")
        
        # Test calculator pipeline
        calc = create_calculator_pipeline(
            name="test_calc_util",
            model="gpt-4o-mini"
        )
        
        result2 = calc.run("What's 25 * 4 + 10?")
        print(f"✅ Calculator pipeline: {result2.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 LLMPipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Pipeline", test_basic_pipeline),
        ("Evaluated Pipeline", test_evaluated_pipeline),
        ("Structured Output", test_structured_output),
        ("Tools Pipeline", test_tools_pipeline),
        ("Interactive Pipeline", test_interactive_pipeline),
        ("Utility Functions", test_utility_functions)
    ]
    
    results = []
    
    # Run sync tests
    for test_name, test_func in tests:
        print(f"\n📝 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run async test
    print(f"\n📝 Running Async Pipeline...")
    try:
        success = asyncio.run(test_async_pipeline())
        results.append(("Async Pipeline", success))
    except Exception as e:
        print(f"❌ Async Pipeline failed with exception: {e}")
        results.append(("Async Pipeline", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LLMPipeline is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your refinire installation and configuration.")
    
    return passed == total

if __name__ == "__main__":
    # Ensure environment is set up
    try:
        import refinire
        # Try to get version, fallback if not available
        try:
            version = refinire.__version__
            print(f"✅ Refinire version: {version}")
        except AttributeError:
            print("✅ Refinire is installed (version info not available)")
    except ImportError:
        print("❌ Refinire not installed. Please install with: uv add refinire")
        sys.exit(1)
    
    # Run tests
    success = main()
    sys.exit(0 if success else 1)