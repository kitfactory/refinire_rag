#!/usr/bin/env python3
"""
BufferMemory Usage Examples
BufferMemory使用例

This script demonstrates how to use the BufferMemory classes for conversation
history management in RAG applications.

このスクリプトは、RAGアプリケーションでの会話履歴管理のために
BufferMemoryクラスを使用する方法を示します。
"""

import os
import sys
from pathlib import Path

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from refinire_rag.memory import (
    BufferMemory,
    ConversationBufferMemory,
    create_buffer_memory_from_env,
    Message
)


def basic_buffer_memory_example():
    """Basic BufferMemory usage / 基本的なBufferMemory使用法"""
    print("=== Basic BufferMemory Example ===")
    
    # Create a buffer memory with limits
    memory = BufferMemory(max_token_limit=1000, max_messages=10)
    
    # Add conversation messages
    memory.add_user_message("Hello, I need help with Python programming")
    memory.add_ai_message("I'd be happy to help you with Python! What would you like to learn?")
    memory.add_user_message("Can you explain list comprehensions?")
    memory.add_ai_message("List comprehensions are a concise way to create lists in Python. Here's the syntax: [expression for item in iterable if condition]")
    
    # Get formatted conversation
    print("Conversation History:")
    print(memory.get_buffer_string())
    print()
    
    # Get statistics
    print(f"Messages: {memory.get_message_count()}")
    print(f"Estimated tokens: {memory.get_token_count()}")
    print()


def conversation_buffer_memory_example():
    """ConversationBufferMemory with custom prefixes / カスタムプレフィックス付きConversationBufferMemory"""
    print("=== ConversationBufferMemory Example ===")
    
    # Create conversation memory with custom prefixes
    memory = ConversationBufferMemory(
        human_prefix="User",
        ai_prefix="Assistant",
        max_messages=5
    )
    
    # Simulate a customer service conversation
    memory.add_user_message("I'm having trouble with my order")
    memory.add_ai_message("I'm sorry to hear that. Can you provide your order number?")
    memory.add_user_message("My order number is 12345")
    memory.add_ai_message("Thank you. I can see your order. What specific issue are you experiencing?")
    memory.add_user_message("The delivery was supposed to be today but nothing arrived")
    memory.add_ai_message("I apologize for the delay. Let me check the shipping status for you.")
    
    print("Customer Service Conversation:")
    print(memory.get_buffer_string())
    print()


def memory_trimming_example():
    """Demonstrate automatic memory trimming / 自動メモリトリミングのデモ"""
    print("=== Memory Trimming Example ===")
    
    # Create memory with very small limits to demonstrate trimming
    memory = BufferMemory(max_token_limit=200, max_messages=3)
    
    print("Adding messages to trigger trimming...")
    
    # Add many messages
    messages_to_add = [
        "This is the first message in our conversation",
        "This is the second message that adds more content",
        "Here's a third message with even more text content",
        "Fourth message - this should start triggering trimming",
        "Fifth message - older messages should be removed",
        "Sixth and final message - only recent ones should remain"
    ]
    
    for i, msg in enumerate(messages_to_add, 1):
        memory.add_user_message(msg)
        print(f"After message {i}: {memory.get_message_count()} messages, {memory.get_token_count()} tokens")
    
    print("\nFinal conversation (after trimming):")
    print(memory.get_buffer_string())
    print()


def environment_configuration_example():
    """Demonstrate environment variable configuration / 環境変数設定のデモ"""
    print("=== Environment Configuration Example ===")
    
    # Set environment variables
    os.environ.update({
        'REFINIRE_RAG_MEMORY_TYPE': 'conversation',
        'REFINIRE_RAG_MEMORY_MAX_TOKENS': '500',
        'REFINIRE_RAG_MEMORY_MAX_MESSAGES': '8',
        'REFINIRE_RAG_MEMORY_HUMAN_PREFIX': 'Student',
        'REFINIRE_RAG_MEMORY_AI_PREFIX': 'Teacher'
    })
    
    # Create memory from environment
    memory = create_buffer_memory_from_env()
    
    # Simulate educational conversation
    memory.add_user_message("What is machine learning?")
    memory.add_ai_message("Machine learning is a type of artificial intelligence that allows computers to learn and improve from experience without being explicitly programmed.")
    memory.add_user_message("Can you give me a simple example?")
    memory.add_ai_message("Sure! Think of email spam filters. They learn to identify spam by looking at thousands of examples of spam and legitimate emails.")
    
    print("Educational Conversation (from environment config):")
    print(memory.get_buffer_string())
    print(f"Configuration: max_tokens={memory.max_token_limit}, max_messages={memory.max_messages}")
    print()


def metadata_usage_example():
    """Demonstrate metadata usage with messages / メッセージでのメタデータ使用のデモ"""
    print("=== Metadata Usage Example ===")
    
    memory = BufferMemory()
    
    # Add messages with metadata
    memory.add_user_message(
        "What's the weather like today?",
        metadata={
            "user_id": "user123",
            "session_id": "session456",
            "intent": "weather_query",
            "confidence": 0.95
        }
    )
    
    memory.add_ai_message(
        "I don't have access to current weather data, but I can help you find weather information.",
        metadata={
            "response_type": "helpful_redirect",
            "capabilities": ["general_info", "guidance"],
            "confidence": 0.90
        }
    )
    
    # Access messages with metadata
    messages = memory.get_messages()
    
    print("Messages with metadata:")
    for i, msg in enumerate(messages, 1):
        print(f"Message {i}:")
        print(f"  Role: {msg.role}")
        print(f"  Content: {msg.content}")
        print(f"  Metadata: {msg.metadata}")
        print(f"  Timestamp: {msg.timestamp}")
        print()


def message_filtering_example():
    """Demonstrate message filtering and retrieval / メッセージフィルタリングと取得のデモ"""
    print("=== Message Filtering Example ===")
    
    memory = BufferMemory()
    
    # Add a longer conversation
    conversation = [
        ("Hello", "human"),
        ("Hi there! How can I help you today?", "ai"),
        ("I want to learn about Python", "human"),
        ("Great choice! Python is a versatile programming language.", "ai"),
        ("What can I build with Python?", "human"),
        ("You can build web applications, data analysis tools, machine learning models, and much more!", "ai"),
        ("That sounds exciting!", "human"),
        ("It really is! Would you like to start with a specific area?", "ai")
    ]
    
    for content, role in conversation:
        if role == "human":
            memory.add_user_message(content)
        else:
            memory.add_ai_message(content)
    
    print(f"Total conversation has {memory.get_message_count()} messages")
    print()
    
    # Get last 4 messages
    recent_messages = memory.get_messages(limit=4)
    print("Last 4 messages:")
    for msg in recent_messages:
        role_name = "User" if msg.role == "human" else "AI"
        print(f"{role_name}: {msg.content}")
    print()
    
    # Get all messages as formatted string
    print("Complete conversation:")
    print(memory.get_buffer_string())
    print()


def main():
    """Run all examples / すべての例を実行"""
    print("BufferMemory Examples")
    print("=" * 50)
    print()
    
    basic_buffer_memory_example()
    conversation_buffer_memory_example()
    memory_trimming_example()
    environment_configuration_example()
    metadata_usage_example()
    message_filtering_example()
    
    print("All examples completed!")


if __name__ == "__main__":
    main()