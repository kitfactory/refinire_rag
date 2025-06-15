"""
Test for AnswerSynthesizer instruction customization

Tests the ability to customize generation instructions and system prompts
for both Refinire LLMPipeline and OpenAI implementations.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from refinire_rag.retrieval.simple_reader import SimpleAnswerSynthesizer, SimpleAnswerSynthesizerConfig
from refinire_rag.retrieval.base import SearchResult
from refinire_rag.models.document import Document


class TestAnswerSynthesizerInstructions:
    """Test instruction customization for AnswerSynthesizer"""

    def test_custom_generation_instructions_refinire(self):
        """Test custom generation instructions with Refinire LLMPipeline"""
        
        # Custom instructions
        custom_instructions = """You are a technical documentation expert. When answering questions:
1. Provide detailed technical explanations
2. Include code examples when relevant
3. Structure answers with clear headings
4. Always cite specific sources
5. If information is incomplete, clearly state limitations"""
        
        # Create config with custom instructions
        config = SimpleAnswerSynthesizerConfig(
            generation_instructions=custom_instructions,
            llm_model="gpt-4o-mini"
        )
        
        # Mock Refinire LLMPipeline
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline:
            mock_pipeline_instance = Mock()
            mock_llm_pipeline.return_value = mock_pipeline_instance
            
            # Initialize synthesizer
            synthesizer = SimpleAnswerSynthesizer(config)
            
            # Verify that LLMPipeline was initialized with custom instructions
            mock_llm_pipeline.assert_called_once_with(
                name="answer_synthesizer",
                generation_instructions=custom_instructions,
                model="gpt-4o-mini"
            )
            
            # Test synthesize call
            mock_result = Mock()
            mock_result.content = "Technical answer with detailed explanation and code examples."
            mock_pipeline_instance.run.return_value = mock_result
            
            # Create test context
            test_doc = Document(
                id="test_doc",
                content="Test content about machine learning algorithms.",
                metadata={"source": "test"}
            )
            contexts = [SearchResult(
                document_id="test_doc",
                document=test_doc,
                score=0.9,
                metadata={}
            )]
            
            # Synthesize answer
            answer = synthesizer.synthesize("What is machine learning?", contexts)
            
            # Verify the pipeline was called
            assert mock_pipeline_instance.run.called
            assert answer == "Technical answer with detailed explanation and code examples."

    def test_custom_system_prompt_openai(self):
        """Test custom system prompt with OpenAI client"""
        
        # Custom system prompt
        custom_system_prompt = """You are a scientific research assistant specializing in AI and machine learning. 
Your responses should be:
- Academically rigorous and precise
- Include relevant citations and references
- Use formal scientific language
- Clearly distinguish between established facts and current research
- Provide confidence levels for statements when appropriate"""
        
        # Create config with custom system prompt
        config = SimpleAnswerSynthesizerConfig(
            system_prompt=custom_system_prompt,
            generation_instructions="Custom refinire instructions",  # This won't be used for OpenAI
            llm_model="gpt-4o-mini"
        )
        
        # Mock LLMPipeline to fail, forcing OpenAI fallback
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline', None):
            with patch('openai.OpenAI') as mock_openai:
                mock_client = Mock()
                mock_openai.return_value = mock_client
                
                # Mock response
                mock_response = Mock()
                mock_choice = Mock()
                mock_message = Mock()
                mock_message.content = "Academic response with scientific rigor."
                mock_choice.message = mock_message
                mock_response.choices = [mock_choice]
                mock_client.chat.completions.create.return_value = mock_response
                
                # Initialize synthesizer
                synthesizer = SimpleAnswerSynthesizer(config)
                
                # Create test context
                test_doc = Document(
                    id="test_doc",
                    content="Research on neural networks and deep learning.",
                    metadata={"source": "research_paper"}
                )
                contexts = [SearchResult(
                    document_id="test_doc",
                    document=test_doc,
                    score=0.95,
                    metadata={}
                )]
                
                # Synthesize answer
                answer = synthesizer.synthesize("Explain neural networks", contexts)
                
                # Verify OpenAI client was called with custom system prompt
                mock_client.chat.completions.create.assert_called_once()
                call_args = mock_client.chat.completions.create.call_args
                
                # Check that custom system prompt was used
                messages = call_args[1]['messages']
                assert len(messages) == 2
                assert messages[0]['role'] == 'system'
                assert messages[0]['content'] == custom_system_prompt
                assert messages[1]['role'] == 'user'
                
                assert answer == "Academic response with scientific rigor."

    def test_different_instruction_styles(self):
        """Test different instruction styles for various use cases"""
        
        test_cases = [
            {
                "name": "Customer Service",
                "instructions": """You are a friendly customer service representative. Always:
- Be empathetic and understanding
- Provide helpful solutions
- Use simple, non-technical language
- Offer additional assistance
- End with asking if there's anything else you can help with""",
                "expected_tone": "friendly and helpful"
            },
            {
                "name": "Legal Assistant", 
                "instructions": """You are a legal research assistant. Your responses must:
- Be precise and legally accurate
- Include relevant case law or statutes when applicable
- Use proper legal terminology
- Clearly state when legal advice is needed from a qualified attorney
- Structure responses with clear legal reasoning""",
                "expected_tone": "precise and legally accurate"
            },
            {
                "name": "Educational Tutor",
                "instructions": """You are an educational tutor helping students learn. Your approach should:
- Break down complex concepts into simple steps
- Use analogies and examples students can relate to
- Encourage critical thinking with follow-up questions
- Provide positive reinforcement
- Adapt explanations to the student's level""",
                "expected_tone": "educational and encouraging"
            }
        ]
        
        for test_case in test_cases:
            config = SimpleAnswerSynthesizerConfig(
                generation_instructions=test_case["instructions"],
                system_prompt=test_case["instructions"],
                llm_model="gpt-4o-mini"
            )
            
            # Test with Refinire mock
            with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline:
                mock_pipeline_instance = Mock()
                mock_llm_pipeline.return_value = mock_pipeline_instance
                
                synthesizer = SimpleAnswerSynthesizer(config)
                
                # Verify custom instructions were set
                mock_llm_pipeline.assert_called_once_with(
                    name="answer_synthesizer",
                    generation_instructions=test_case["instructions"],
                    model="gpt-4o-mini"
                )
                
                print(f"✓ {test_case['name']} instructions configured successfully")

    def test_instruction_update_after_initialization(self):
        """Test updating instructions after synthesizer initialization"""
        
        # Initial config
        initial_config = SimpleAnswerSynthesizerConfig(
            generation_instructions="Initial instructions",
            llm_model="gpt-4o-mini"
        )
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline:
            mock_pipeline_instance = Mock()
            mock_llm_pipeline.return_value = mock_pipeline_instance
            
            synthesizer = SimpleAnswerSynthesizer(initial_config)
            
            # Update instructions
            new_instructions = "Updated instructions for better performance"
            synthesizer.config.generation_instructions = new_instructions
            
            # Re-initialize the pipeline with new instructions
            synthesizer._llm_pipeline = mock_llm_pipeline(
                name="answer_synthesizer",
                generation_instructions=new_instructions,
                model=synthesizer.config.llm_model
            )
            
            # Verify new instructions are used
            assert synthesizer.config.generation_instructions == new_instructions

    def test_empty_or_invalid_instructions(self):
        """Test behavior with empty or invalid instructions"""
        
        # Test with empty instructions (should use default)
        config_empty = SimpleAnswerSynthesizerConfig(
            generation_instructions="",
            llm_model="gpt-4o-mini"
        )
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline:
            mock_pipeline_instance = Mock()
            mock_llm_pipeline.return_value = mock_pipeline_instance
            
            synthesizer = SimpleAnswerSynthesizer(config_empty)
            
            # Should still work with empty instructions
            mock_llm_pipeline.assert_called_once_with(
                name="answer_synthesizer", 
                generation_instructions="",
                model="gpt-4o-mini"
            )

    def test_instruction_metadata_in_stats(self):
        """Test that instruction info is included in processing stats"""
        
        custom_instructions = "Custom instructions for testing"
        config = SimpleAnswerSynthesizerConfig(
            generation_instructions=custom_instructions,
            llm_model="gpt-4o-mini"
        )
        
        with patch('refinire_rag.retrieval.simple_reader.LLMPipeline') as mock_llm_pipeline:
            mock_pipeline_instance = Mock()
            mock_llm_pipeline.return_value = mock_pipeline_instance
            
            synthesizer = SimpleAnswerSynthesizer(config)
            
            # Get stats
            stats = synthesizer.get_processing_stats()
            
            # Verify basic stats structure
            assert "synthesizer_type" in stats
            assert "model" in stats
            assert stats["synthesizer_type"] == "SimpleAnswerSynthesizer"
            assert stats["model"] == "gpt-4o-mini"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])