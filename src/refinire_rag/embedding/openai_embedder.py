"""
OpenAI Embeddings implementation

Provides embedding functionality using OpenAI's embedding models via their API.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time
import numpy as np

from .base import Embedder, EmbeddingConfig, EmbeddingResult
from ..exceptions import EmbeddingError


@dataclass
class OpenAIEmbeddingConfig(EmbeddingConfig):
    """Configuration for OpenAI embeddings
    OpenAI埋め込みの設定"""
    
    # OpenAI specific settings
    model_name: str = "text-embedding-3-small"  # or "text-embedding-3-large", "text-embedding-ada-002"
    api_key: Optional[str] = None  # Will use OPENAI_API_KEY env var if None
    api_base: Optional[str] = None  # Custom API base URL if needed
    organization: Optional[str] = None  # OpenAI organization ID
    
    # Model specific dimensions
    embedding_dimension: int = 1536  # text-embedding-3-small default
    
    # Request parameters
    batch_size: int = 100  # OpenAI allows up to 2048 inputs per request
    max_tokens: int = 8191  # Maximum tokens per input for text-embedding-3-small
    
    # Rate limiting and retries
    requests_per_minute: int = 3000  # Rate limit for API calls
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Advanced options
    strip_newlines: bool = True  # Remove newlines from text
    user_identifier: Optional[str] = None  # Optional user identifier for OpenAI


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings implementation
    OpenAI埋め込みの実装"""
    
    def __init__(self, config: Optional[OpenAIEmbeddingConfig] = None):
        """Initialize OpenAI embedder"""
        self.config = config or OpenAIEmbeddingConfig()
        super().__init__(self.config)
        
        # Initialize OpenAI client
        self._client = None
        self._init_client()
        
        # Set embedding dimension based on model
        self._set_model_dimensions()
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_count = 0
        self._rate_limit_window_start = time.time()
    
    def _init_client(self):
        """Initialize the OpenAI client"""
        try:
            import openai
            
            # Get API key from config or environment
            api_key = self.config.api_key
            if api_key is None:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                
            if not api_key:
                raise EmbeddingError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key in config."
                )
            
            # Initialize client with configuration
            client_kwargs = {"api_key": api_key}
            
            if self.config.api_base:
                client_kwargs["base_url"] = self.config.api_base
            
            if self.config.organization:
                client_kwargs["organization"] = self.config.organization
            
            self._client = openai.OpenAI(**client_kwargs)
            
        except ImportError:
            raise EmbeddingError(
                "OpenAI library not found. Install with: pip install openai"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}")
    
    def _set_model_dimensions(self):
        """Set embedding dimensions based on model name"""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        if self.config.model_name in model_dimensions:
            self.config.embedding_dimension = model_dimensions[self.config.model_name]
    
    def _apply_rate_limiting(self):
        """Apply rate limiting to respect OpenAI's limits"""
        current_time = time.time()
        
        # Reset counter if we're in a new minute window
        if current_time - self._rate_limit_window_start >= 60:
            self._request_count = 0
            self._rate_limit_window_start = current_time
        
        # Check if we're approaching rate limit
        if self._request_count >= self.config.requests_per_minute:
            sleep_time = 60 - (current_time - self._rate_limit_window_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self._request_count = 0
                self._rate_limit_window_start = time.time()
        
        # Minimum delay between requests
        time_since_last = current_time - self._last_request_time
        min_delay = 60.0 / self.config.requests_per_minute
        if time_since_last < min_delay:
            time.sleep(min_delay - time_since_last)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string using OpenAI API"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cached_vector = self._get_from_cache(cache_key)
        if cached_vector is not None:
            return cached_vector
        
        try:
            # Validate and prepare text
            if not text.strip():
                raise EmbeddingError("Cannot embed empty text")
            
            # Truncate if necessary
            processed_text = self.truncate_text(text)
            if self.config.strip_newlines:
                processed_text = processed_text.replace('\n', ' ')
            
            # Apply rate limiting
            self._apply_rate_limiting()
            
            # Prepare request parameters
            request_params = {
                "input": [processed_text],
                "model": self.config.model_name,
                "encoding_format": "float"
            }
            
            if self.config.user_identifier:
                request_params["user"] = self.config.user_identifier
            
            # Custom dimension (for text-embedding-3-small/large)
            if "text-embedding-3" in self.config.model_name and self.config.embedding_dimension != self._get_default_dimension():
                request_params["dimensions"] = self.config.embedding_dimension
            
            # Make API call with retries
            response = self._make_request_with_retries(request_params)
            
            # Extract embedding
            embedding_data = response.data[0]
            vector = np.array(embedding_data.embedding)
            
            # Normalize if requested
            if self.config.normalize_vectors:
                vector = vector / np.linalg.norm(vector)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, success=True)
            
            # Cache result
            self._store_in_cache(cache_key, vector)
            
            return vector
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, success=False)
            
            error_msg = f"OpenAI embedding failed: {e}"
            
            if self.config.fail_on_error:
                raise EmbeddingError(error_msg)
            
            # Return zero vector on error if not failing
            return np.zeros(self.config.embedding_dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently using OpenAI batch API"""
        start_time = time.time()
        vectors = []
        
        # Check cache for all texts first
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cached_vector = self._get_from_cache(cache_key)
            
            if cached_vector is not None:
                vectors.append((i, cached_vector))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            try:
                # Prepare texts
                processed_texts = []
                for text in uncached_texts:
                    if not text.strip():
                        processed_texts.append("")
                        continue
                    
                    processed_text = self.truncate_text(text)
                    if self.config.strip_newlines:
                        processed_text = processed_text.replace('\n', ' ')
                    processed_texts.append(processed_text)
                
                # Apply rate limiting
                self._apply_rate_limiting()
                
                # Prepare batch request
                request_params = {
                    "input": processed_texts,
                    "model": self.config.model_name,
                    "encoding_format": "float"
                }
                
                if self.config.user_identifier:
                    request_params["user"] = self.config.user_identifier
                
                # Custom dimension
                if "text-embedding-3" in self.config.model_name and self.config.embedding_dimension != self._get_default_dimension():
                    request_params["dimensions"] = self.config.embedding_dimension
                
                # Make batch API call
                response = self._make_request_with_retries(request_params)
                
                # Process results
                for i, (text, embedding_data) in enumerate(zip(uncached_texts, response.data)):
                    vector = np.array(embedding_data.embedding)
                    
                    if self.config.normalize_vectors:
                        vector = vector / np.linalg.norm(vector)
                    
                    # Cache result
                    cache_key = self._get_cache_key(text)
                    self._store_in_cache(cache_key, vector)
                    
                    vectors.append((uncached_indices[i], vector))
                
                # Update stats
                processing_time = time.time() - start_time
                self._update_stats(processing_time / len(uncached_texts), success=True)
                
            except Exception as e:
                # Handle batch failure
                processing_time = time.time() - start_time
                error_msg = f"OpenAI batch embedding failed: {e}"
                
                for i, text in enumerate(uncached_texts):
                    self._update_stats(processing_time / len(uncached_texts), success=False)
                    
                    if self.config.fail_on_error:
                        raise EmbeddingError(error_msg)
                    
                    # Add zero vector on error
                    error_vector = np.zeros(self.config.embedding_dimension)
                    vectors.append((uncached_indices[i], error_vector))
        
        # Sort vectors by original index
        vectors.sort(key=lambda x: x[0])
        return [vector for _, vector in vectors]
    
    def _make_request_with_retries(self, request_params: Dict[str, Any]):
        """Make API request with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.embeddings.create(**request_params)
                return response
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    # Wait before retry
                    wait_time = self.config.retry_delay_seconds * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
        raise last_exception
    
    def _get_default_dimension(self) -> int:
        """Get default dimension for the current model"""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.config.model_name, 1536)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder"""
        return self.config.embedding_dimension
    
    def validate_text_length(self, text: str) -> bool:
        """Validate text length using proper tokenization if available"""
        try:
            import tiktoken
            
            # Get tokenizer for the model
            encoding = tiktoken.encoding_for_model(self.config.model_name)
            token_count = len(encoding.encode(text))
            return token_count <= self.config.max_tokens
            
        except ImportError:
            # Fallback to character-based estimation
            return super().validate_text_length(text)
        except Exception:
            # If tiktoken fails for this model, use character estimation
            return super().validate_text_length(text)
    
    def truncate_text(self, text: str) -> str:
        """Truncate text using proper tokenization if available"""
        if self.validate_text_length(text):
            return text
        
        try:
            import tiktoken
            
            encoding = tiktoken.encoding_for_model(self.config.model_name)
            tokens = encoding.encode(text)
            
            if len(tokens) <= self.config.max_tokens:
                return text
            
            # Truncate to max tokens
            truncated_tokens = tokens[:self.config.max_tokens]
            return encoding.decode(truncated_tokens)
            
        except (ImportError, Exception):
            # Fallback to character-based truncation
            return super().truncate_text(text)