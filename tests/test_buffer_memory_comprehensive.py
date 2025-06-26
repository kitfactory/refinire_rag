"""
Comprehensive tests for BufferMemory implementation
BufferMemory実装の包括的テスト
"""

import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from refinire_rag.memory.buffer_memory import (
    Message,
    BaseMemory,
    BufferMemory,
    ConversationBufferMemory,
    create_buffer_memory_from_env
)
from refinire_rag.exceptions import RefinireRAGError


class TestMessage:
    """Test Message model functionality"""
    
    def test_message_creation(self):
        """Test basic message creation"""
        message = Message(content="Hello world", role="human")
        
        assert message.content == "Hello world"
        assert message.role == "human"
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}
    
    def test_message_with_metadata(self):
        """Test message creation with metadata"""
        metadata = {"source": "test", "importance": "high"}
        message = Message(
            content="Test message",
            role="ai",
            metadata=metadata
        )
        
        assert message.content == "Test message"
        assert message.role == "ai"
        assert message.metadata == metadata
    
    def test_message_timestamp(self):
        """Test message timestamp functionality"""
        before = datetime.now()
        message = Message(content="Time test", role="system")
        after = datetime.now()
        
        assert before <= message.timestamp <= after
    
    def test_message_serialization(self):
        """Test message serialization/deserialization"""
        original = Message(
            content="Serialization test",
            role="human",
            metadata={"test": True}
        )
        
        # Test dict conversion
        message_dict = original.model_dump()
        assert message_dict['content'] == "Serialization test"
        assert message_dict['role'] == "human"
        assert message_dict['metadata'] == {"test": True}
        
        # Test reconstruction
        reconstructed = Message(**message_dict)
        assert reconstructed.content == original.content
        assert reconstructed.role == original.role
        assert reconstructed.metadata == original.metadata


class TestBufferMemory:
    """Test BufferMemory functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.memory = BufferMemory(max_token_limit=1000, max_messages=10)
    
    def test_initialization(self):
        """Test BufferMemory initialization"""
        memory = BufferMemory()
        assert memory.max_token_limit == 4000  # Default value
        assert memory.max_messages == 100      # Default value
        assert memory.get_message_count() == 0
        assert memory.get_token_count() == 0
    
    def test_initialization_with_params(self):
        """Test BufferMemory initialization with parameters"""
        memory = BufferMemory(max_token_limit=500, max_messages=5)
        assert memory.max_token_limit == 500
        assert memory.max_messages == 5
    
    def test_add_message(self):
        """Test adding messages to memory"""
        message = Message(content="Hello", role="human")
        self.memory.add_message(message)
        
        assert self.memory.get_message_count() == 1
        assert self.memory.get_token_count() > 0
        
        messages = self.memory.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "Hello"
        assert messages[0].role == "human"
    
    def test_add_user_message(self):
        """Test convenience method for adding user messages"""
        self.memory.add_user_message("User input", {"type": "question"})
        
        messages = self.memory.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "User input"
        assert messages[0].role == "human"
        assert messages[0].metadata == {"type": "question"}
    
    def test_add_ai_message(self):
        """Test convenience method for adding AI messages"""
        self.memory.add_ai_message("AI response", {"confidence": 0.95})
        
        messages = self.memory.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "AI response"
        assert messages[0].role == "ai"
        assert messages[0].metadata == {"confidence": 0.95}
    
    def test_conversation_flow(self):
        """Test typical conversation flow"""
        # Add conversation
        self.memory.add_user_message("What is machine learning?")
        self.memory.add_ai_message("Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.")
        self.memory.add_user_message("Can you give me an example?")
        self.memory.add_ai_message("Sure! A spam email filter that learns to identify spam based on training data is a common example.")
        
        messages = self.memory.get_messages()
        assert len(messages) == 4
        
        # Check conversation order
        assert messages[0].role == "human"
        assert messages[1].role == "ai"
        assert messages[2].role == "human"
        assert messages[3].role == "ai"
        
        # Check content
        assert "machine learning" in messages[0].content.lower()
        assert "subset of AI" in messages[1].content
    
    def test_get_messages_with_limit(self):
        """Test getting messages with limit"""
        # Add multiple messages
        for i in range(5):
            self.memory.add_user_message(f"Message {i}")
        
        # Test different limits
        assert len(self.memory.get_messages(limit=3)) == 3
        assert len(self.memory.get_messages(limit=0)) == 0
        assert len(self.memory.get_messages(limit=10)) == 5  # Only 5 messages exist
        
        # Test that limit returns most recent messages
        recent_messages = self.memory.get_messages(limit=2)
        assert recent_messages[0].content == "Message 3"
        assert recent_messages[1].content == "Message 4"
    
    def test_get_buffer_string(self):
        """Test getting buffer as formatted string"""
        self.memory.add_user_message("Hello")
        self.memory.add_ai_message("Hi there!")
        self.memory.add_user_message("How are you?")
        
        buffer_string = self.memory.get_buffer_string()
        lines = buffer_string.split('\n')
        
        assert len(lines) == 3
        assert lines[0] == "Human: Hello"
        assert lines[1] == "AI: Hi there!"
        assert lines[2] == "Human: How are you?"
    
    def test_get_buffer_string_custom_separator(self):
        """Test getting buffer string with custom separator"""
        self.memory.add_user_message("First")
        self.memory.add_ai_message("Second")
        
        buffer_string = self.memory.get_buffer_string(separator=" | ")
        assert buffer_string == "Human: First | AI: Second"
    
    def test_clear_memory(self):
        """Test clearing memory"""
        self.memory.add_user_message("Test message")
        assert self.memory.get_message_count() == 1
        assert self.memory.get_token_count() > 0
        
        self.memory.clear()
        assert self.memory.get_message_count() == 0
        assert self.memory.get_token_count() == 0
        assert len(self.memory.get_messages()) == 0
    
    def test_message_limit_trimming(self):
        """Test automatic trimming when message limit is exceeded"""
        memory = BufferMemory(max_messages=3)
        
        # Add more messages than limit
        for i in range(5):
            memory.add_user_message(f"Message {i}")
        
        messages = memory.get_messages()
        assert len(messages) == 3  # Should be trimmed to limit
        
        # Check that oldest messages were removed
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"
        assert messages[2].content == "Message 4"
    
    def test_token_limit_trimming(self):
        """Test automatic trimming when token limit is exceeded"""
        memory = BufferMemory(max_token_limit=20)  # Very small limit
        
        # Add messages that exceed token limit
        memory.add_user_message("This is a very long message that should exceed the token limit")
        memory.add_user_message("Another long message")
        memory.add_user_message("Short")
        
        # Should trim older messages to stay under limit
        messages = memory.get_messages()
        
        # At minimum, should keep the most recent message
        assert len(messages) >= 1
        assert messages[-1].content == "Short"
        
        # Token count should be under or at limit
        assert memory.get_token_count() <= memory.max_token_limit
    
    def test_token_estimation(self):
        """Test token count estimation"""
        memory = BufferMemory()
        
        # Test basic estimation
        memory.add_user_message("Hello")  # ~1-2 tokens
        initial_count = memory.get_token_count()
        assert initial_count > 0
        
        memory.add_user_message("This is a longer message with more words")  # More tokens
        final_count = memory.get_token_count()
        assert final_count > initial_count
    
    def test_role_prefix_mapping(self):
        """Test role prefix mapping for different roles"""
        memory = BufferMemory()
        
        # Test various role names
        memory.add_message(Message(content="Test", role="human"))
        memory.add_message(Message(content="Test", role="user"))  # Alternative
        memory.add_message(Message(content="Test", role="ai"))
        memory.add_message(Message(content="Test", role="assistant"))  # Alternative
        memory.add_message(Message(content="Test", role="system"))
        memory.add_message(Message(content="Test", role="custom"))
        
        buffer_string = memory.get_buffer_string()
        lines = buffer_string.split('\n')
        
        assert lines[0].startswith("Human: ")
        assert lines[1].startswith("Human: ")  # user -> Human
        assert lines[2].startswith("AI: ")
        assert lines[3].startswith("AI: ")     # assistant -> AI
        assert lines[4].startswith("System: ")
        assert lines[5].startswith("Custom: ") # custom -> Custom


class TestConversationBufferMemory:
    """Test ConversationBufferMemory functionality"""
    
    def test_initialization(self):
        """Test ConversationBufferMemory initialization"""
        memory = ConversationBufferMemory()
        assert memory.human_prefix == "Human"
        assert memory.ai_prefix == "AI"
    
    def test_custom_prefixes(self):
        """Test custom prefixes"""
        memory = ConversationBufferMemory(
            human_prefix="User",
            ai_prefix="Assistant"
        )
        assert memory.human_prefix == "User"
        assert memory.ai_prefix == "Assistant"
        
        memory.add_user_message("Hello")
        memory.add_ai_message("Hi there")
        
        buffer_string = memory.get_buffer_string()
        lines = buffer_string.split('\n')
        
        assert lines[0] == "User: Hello"
        assert lines[1] == "Assistant: Hi there"
    
    def test_inheritance_from_buffer_memory(self):
        """Test that ConversationBufferMemory inherits BufferMemory functionality"""
        memory = ConversationBufferMemory(max_messages=2)
        
        # Test inherited functionality
        memory.add_user_message("Message 1")
        memory.add_user_message("Message 2")
        memory.add_user_message("Message 3")  # Should trigger trimming
        
        messages = memory.get_messages()
        assert len(messages) == 2  # Trimmed to max_messages


class TestEnvironmentIntegration:
    """Test environment variable integration"""
    
    def test_create_buffer_memory_from_env_defaults(self):
        """Test creating BufferMemory from environment variables with defaults"""
        with patch.dict(os.environ, {}, clear=True):
            memory = create_buffer_memory_from_env()
            
            assert isinstance(memory, BufferMemory)
            assert not isinstance(memory, ConversationBufferMemory)
            assert memory.max_token_limit == 4000
            assert memory.max_messages == 100
    
    def test_create_buffer_memory_from_env_custom(self):
        """Test creating BufferMemory with custom environment variables"""
        env_vars = {
            'REFINIRE_RAG_MEMORY_TYPE': 'buffer',
            'REFINIRE_RAG_MEMORY_MAX_TOKENS': '2000',
            'REFINIRE_RAG_MEMORY_MAX_MESSAGES': '50'
        }
        
        with patch.dict(os.environ, env_vars):
            memory = create_buffer_memory_from_env()
            
            assert isinstance(memory, BufferMemory)
            assert memory.max_token_limit == 2000
            assert memory.max_messages == 50
    
    def test_create_conversation_memory_from_env(self):
        """Test creating ConversationBufferMemory from environment variables"""
        env_vars = {
            'REFINIRE_RAG_MEMORY_TYPE': 'conversation',
            'REFINIRE_RAG_MEMORY_MAX_TOKENS': '3000',
            'REFINIRE_RAG_MEMORY_MAX_MESSAGES': '75',
            'REFINIRE_RAG_MEMORY_HUMAN_PREFIX': 'User',
            'REFINIRE_RAG_MEMORY_AI_PREFIX': 'Bot'
        }
        
        with patch.dict(os.environ, env_vars):
            memory = create_buffer_memory_from_env()
            
            assert isinstance(memory, ConversationBufferMemory)
            assert memory.max_token_limit == 3000
            assert memory.max_messages == 75
            assert memory.human_prefix == "User"
            assert memory.ai_prefix == "Bot"
    
    def test_buffer_memory_env_vars_in_init(self):
        """Test BufferMemory reading environment variables during initialization"""
        env_vars = {
            'REFINIRE_RAG_MEMORY_MAX_TOKENS': '1500',
            'REFINIRE_RAG_MEMORY_MAX_MESSAGES': '25'
        }
        
        with patch.dict(os.environ, env_vars):
            memory = BufferMemory()  # Should read from environment
            
            assert memory.max_token_limit == 1500
            assert memory.max_messages == 25


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_message_data(self):
        """Test handling of invalid message data"""
        with pytest.raises(ValueError):
            Message(content="", role="human")  # Empty content should fail
    
    def test_zero_limits(self):
        """Test behavior with zero limits"""
        # Zero token limit should still allow some messages
        memory = BufferMemory(max_token_limit=0, max_messages=1)
        memory.add_user_message("Test")
        
        # Should handle gracefully
        assert memory.get_message_count() >= 0
    
    def test_negative_limits(self):
        """Test behavior with negative limits"""
        memory = BufferMemory(max_token_limit=-1, max_messages=-1)
        memory.add_user_message("Test")
        
        # Should handle gracefully without crashing
        assert memory.get_message_count() >= 0
    
    def test_empty_content_messages(self):
        """Test handling of messages with empty content"""
        memory = BufferMemory()
        
        # Empty content should raise ValueError
        with pytest.raises(ValueError):
            Message(content="", role="human")
    
    def test_large_message_handling(self):
        """Test handling of very large messages"""
        memory = BufferMemory(max_token_limit=100)
        
        # Add a message that exceeds the entire token limit
        large_content = "A" * 1000  # Very large message
        memory.add_user_message(large_content)
        
        # Memory should handle it (may trim to fit or keep minimum)
        assert memory.get_message_count() >= 0
        assert memory.get_token_count() >= 0


class TestMemoryPersistence:
    """Test memory state management"""
    
    def test_message_order_preservation(self):
        """Test that message order is preserved"""
        memory = BufferMemory()
        
        messages_to_add = [
            ("First", "human"),
            ("Second", "ai"),
            ("Third", "human"),
            ("Fourth", "ai")
        ]
        
        for content, role in messages_to_add:
            memory.add_message(Message(content=content, role=role))
        
        retrieved_messages = memory.get_messages()
        
        for i, (expected_content, expected_role) in enumerate(messages_to_add):
            assert retrieved_messages[i].content == expected_content
            assert retrieved_messages[i].role == expected_role
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved across operations"""
        memory = BufferMemory()
        
        metadata = {
            "session_id": "test_123",
            "user_id": "user_456",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        memory.add_user_message("Test with metadata", metadata)
        
        retrieved_messages = memory.get_messages()
        assert len(retrieved_messages) == 1
        assert retrieved_messages[0].metadata == metadata
    
    def test_timestamp_consistency(self):
        """Test timestamp handling"""
        memory = BufferMemory()
        
        before_add = datetime.now()
        memory.add_user_message("Timestamp test")
        after_add = datetime.now()
        
        messages = memory.get_messages()
        message_timestamp = messages[0].timestamp
        
        # Timestamp should be between before and after
        assert before_add <= message_timestamp <= after_add


@pytest.mark.integration
class TestBufferMemoryIntegration:
    """Integration tests for BufferMemory"""
    
    def test_realistic_conversation_scenario(self):
        """Test realistic conversation scenario"""
        memory = BufferMemory(max_token_limit=2000, max_messages=20)
        
        # Simulate a realistic conversation
        conversation = [
            ("Hello, I need help with Python programming", "human"),
            ("I'd be happy to help you with Python! What specific topic would you like to learn about?", "ai"),
            ("I want to learn about list comprehensions", "human"),
            ("List comprehensions are a concise way to create lists. Here's the basic syntax: [expression for item in iterable if condition]", "ai"),
            ("Can you show me an example?", "human"),
            ("Sure! Here's an example: squares = [x**2 for x in range(10)] creates a list of squares from 0 to 9.", "ai"),
            ("That's helpful! What about filtering?", "human"),
            ("You can add conditions: even_squares = [x**2 for x in range(10) if x % 2 == 0] gets squares of even numbers only.", "ai")
        ]
        
        for content, role in conversation:
            if role == "human":
                memory.add_user_message(content)
            else:
                memory.add_ai_message(content)
        
        # Test conversation integrity
        messages = memory.get_messages()
        assert len(messages) == len(conversation)
        
        # Test buffer string formatting
        buffer_string = memory.get_buffer_string()
        assert "Python programming" in buffer_string
        assert "list comprehensions" in buffer_string
        assert "Human:" in buffer_string
        assert "AI:" in buffer_string
        
        # Test token management
        assert memory.get_token_count() > 0
        assert memory.get_token_count() <= memory.max_token_limit
    
    def test_memory_with_trimming_scenario(self):
        """Test memory behavior with automatic trimming"""
        memory = BufferMemory(max_token_limit=200, max_messages=5)
        
        # Add many messages to trigger trimming
        for i in range(10):
            memory.add_user_message(f"This is message number {i} with some additional content to increase token count")
        
        final_messages = memory.get_messages()
        
        # Should be trimmed to limits
        assert len(final_messages) <= 5
        assert memory.get_token_count() <= 200
        
        # Should keep most recent messages
        assert "message number 9" in final_messages[-1].content
    
    def test_mixed_role_conversation(self):
        """Test conversation with mixed roles including system messages"""
        memory = ConversationBufferMemory()
        
        # Add system message
        memory.add_message(Message(content="You are a helpful assistant.", role="system"))
        
        # Add conversation
        memory.add_user_message("What's the weather like?")
        memory.add_ai_message("I don't have access to current weather data.")
        memory.add_user_message("Can you tell me a joke instead?")
        memory.add_ai_message("Why don't scientists trust atoms? Because they make up everything!")
        
        buffer_string = memory.get_buffer_string()
        lines = buffer_string.split('\n')
        
        # Check formatting for different roles
        assert lines[0].startswith("System:")
        assert lines[1].startswith("Human:")
        assert lines[2].startswith("AI:")
        assert lines[3].startswith("Human:")
        assert lines[4].startswith("AI:")
        
        # Check content
        assert "helpful assistant" in lines[0]
        assert "joke" in lines[3]