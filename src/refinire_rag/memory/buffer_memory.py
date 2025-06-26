"""
Buffer Memory - Conversation history management (LangChain compatible)
バッファメモリ - 会話履歴管理（LangChain互換）

This module provides memory functionality for storing and retrieving conversation history
in RAG applications, compatible with LangChain's memory interface.

このモジュールは、LangChainのメモリインターフェースと互換性のある、RAGアプリケーションでの
会話履歴の保存と取得機能を提供します。
"""

import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator
from ..exceptions import RefinireRAGError


class Message(BaseModel):
    """
    Represents a single message in conversation history
    会話履歴内の単一メッセージを表現
    """
    content: str = Field(..., description="Message content / メッセージ内容")
    role: str = Field(..., description="Message role (human/ai/system) / メッセージの役割")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp / メッセージのタイムスタンプ")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata / 追加メタデータ")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class BaseMemory(ABC):
    """
    Abstract base class for memory implementations
    メモリ実装の抽象基底クラス
    """
    
    @abstractmethod
    def add_message(self, message: Message) -> None:
        """Add a message to memory / メモリにメッセージを追加"""
        pass
    
    @abstractmethod
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory / メモリからメッセージを取得"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from memory / メモリからすべてのメッセージをクリア"""
        pass
    
    @abstractmethod
    def get_buffer_string(self) -> str:
        """Get messages as formatted string / フォーマットされた文字列としてメッセージを取得"""
        pass


class BufferMemory(BaseMemory):
    """
    Simple buffer memory that stores all conversation history
    すべての会話履歴を保存するシンプルなバッファメモリ
    
    LangChain compatible buffer memory implementation that stores conversation
    history in a simple list with configurable limits.
    
    設定可能な制限を持つシンプルなリストに会話履歴を保存する、
    LangChain互換のバッファメモリ実装。
    """
    
    def __init__(self, max_token_limit: Optional[int] = None, max_messages: Optional[int] = None):
        """
        Initialize BufferMemory
        BufferMemoryを初期化
        
        Args:
            max_token_limit: Maximum token count (approximate) / 最大トークン数（概算）
            max_messages: Maximum number of messages to keep / 保持する最大メッセージ数
        """
        self.max_token_limit = max_token_limit or int(os.getenv('REFINIRE_RAG_MEMORY_MAX_TOKENS', '4000'))
        self.max_messages = max_messages or int(os.getenv('REFINIRE_RAG_MEMORY_MAX_MESSAGES', '100'))
        self.messages: List[Message] = []
        self._current_token_count = 0
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the buffer
        バッファにメッセージを追加
        
        Args:
            message: Message to add / 追加するメッセージ
        """
        self.messages.append(message)
        self._current_token_count += self._estimate_tokens(message.content)
        
        # Trim buffer if necessary
        self._trim_buffer()
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a user message (convenience method)
        ユーザーメッセージを追加（便利メソッド）
        
        Args:
            content: Message content / メッセージ内容
            metadata: Additional metadata / 追加メタデータ
        """
        message = Message(
            content=content,
            role="human",
            metadata=metadata or {}
        )
        self.add_message(message)
    
    def add_ai_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an AI message (convenience method)
        AIメッセージを追加（便利メソッド）
        
        Args:
            content: Message content / メッセージ内容
            metadata: Additional metadata / 追加メタデータ
        """
        message = Message(
            content=content,
            role="ai",
            metadata=metadata or {}
        )
        self.add_message(message)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """
        Get messages from memory
        メモリからメッセージを取得
        
        Args:
            limit: Maximum number of messages to return / 返す最大メッセージ数
            
        Returns:
            List of messages / メッセージのリスト
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:] if limit > 0 else []
    
    def get_buffer_string(self, separator: str = "\n") -> str:
        """
        Get messages as formatted string
        フォーマットされた文字列としてメッセージを取得
        
        Args:
            separator: Separator between messages / メッセージ間のセパレータ
            
        Returns:
            Formatted string of all messages / すべてのメッセージのフォーマット済み文字列
        """
        formatted_messages = []
        for message in self.messages:
            role_prefix = self._get_role_prefix(message.role)
            formatted_messages.append(f"{role_prefix}: {message.content}")
        
        return separator.join(formatted_messages)
    
    def clear(self) -> None:
        """
        Clear all messages from memory
        メモリからすべてのメッセージをクリア
        """
        self.messages.clear()
        self._current_token_count = 0
    
    def get_token_count(self) -> int:
        """
        Get approximate token count
        概算トークン数を取得
        
        Returns:
            Approximate number of tokens / 概算トークン数
        """
        return self._current_token_count
    
    def get_message_count(self) -> int:
        """
        Get number of messages in memory
        メモリ内のメッセージ数を取得
        
        Returns:
            Number of messages / メッセージ数
        """
        return len(self.messages)
    
    def _trim_buffer(self) -> None:
        """
        Trim buffer to stay within limits
        制限内に収まるようにバッファをトリミング
        """
        # Trim by message count
        if len(self.messages) > self.max_messages:
            messages_to_remove = len(self.messages) - self.max_messages
            removed_messages = self.messages[:messages_to_remove]
            self.messages = self.messages[messages_to_remove:]
            
            # Update token count
            for msg in removed_messages:
                self._current_token_count -= self._estimate_tokens(msg.content)
        
        # Trim by token count
        while self._current_token_count > self.max_token_limit and self.messages:
            removed_message = self.messages.pop(0)
            self._current_token_count -= self._estimate_tokens(removed_message.content)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)
        テキストのトークン数を推定（粗い近似）
        
        Args:
            text: Input text / 入力テキスト
            
        Returns:
            Estimated token count / 推定トークン数
        """
        # Simple approximation: ~4 characters per token for English
        # For more accurate counting, could integrate tiktoken or similar
        return max(1, len(text) // 4)
    
    def _get_role_prefix(self, role: str) -> str:
        """
        Get display prefix for role
        役割の表示プレフィックスを取得
        
        Args:
            role: Message role / メッセージの役割
            
        Returns:
            Display prefix / 表示プレフィックス
        """
        role_mapping = {
            "human": "Human",
            "ai": "AI",
            "system": "System",
            "user": "Human",  # Alternative naming
            "assistant": "AI"  # Alternative naming
        }
        return role_mapping.get(role.lower(), role.title())


class ConversationBufferMemory(BufferMemory):
    """
    LangChain-style conversation buffer memory
    LangChainスタイルの会話バッファメモリ
    
    Extended buffer memory with LangChain-compatible interface for
    conversation management.
    
    会話管理のためのLangChain互換インターフェースを持つ拡張バッファメモリ。
    """
    
    def __init__(self, 
                 human_prefix: str = "Human",
                 ai_prefix: str = "AI",
                 **kwargs):
        """
        Initialize ConversationBufferMemory
        ConversationBufferMemoryを初期化
        
        Args:
            human_prefix: Prefix for human messages / 人間のメッセージのプレフィックス
            ai_prefix: Prefix for AI messages / AIメッセージのプレフィックス
            **kwargs: Additional arguments passed to BufferMemory / BufferMemoryに渡す追加引数
        """
        super().__init__(**kwargs)
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
    
    def _get_role_prefix(self, role: str) -> str:
        """
        Get display prefix for role (with custom prefixes)
        役割の表示プレフィックスを取得（カスタムプレフィックス付き）
        
        Args:
            role: Message role / メッセージの役割
            
        Returns:
            Display prefix / 表示プレフィックス
        """
        if role.lower() in ["human", "user"]:
            return self.human_prefix
        elif role.lower() in ["ai", "assistant"]:
            return self.ai_prefix
        else:
            return super()._get_role_prefix(role)


# Environment variable-based factory function
def create_buffer_memory_from_env() -> BufferMemory:
    """
    Create BufferMemory instance from environment variables
    環境変数からBufferMemoryインスタンスを作成
    
    Environment variables:
    - REFINIRE_RAG_MEMORY_TYPE: "buffer" or "conversation" (default: "buffer")
    - REFINIRE_RAG_MEMORY_MAX_TOKENS: Maximum token limit (default: 4000)
    - REFINIRE_RAG_MEMORY_MAX_MESSAGES: Maximum message count (default: 100)
    - REFINIRE_RAG_MEMORY_HUMAN_PREFIX: Human message prefix (default: "Human")
    - REFINIRE_RAG_MEMORY_AI_PREFIX: AI message prefix (default: "AI")
    
    Returns:
        BufferMemory instance / BufferMemoryインスタンス
    """
    memory_type = os.getenv('REFINIRE_RAG_MEMORY_TYPE', 'buffer').lower()
    max_tokens = int(os.getenv('REFINIRE_RAG_MEMORY_MAX_TOKENS', '4000'))
    max_messages = int(os.getenv('REFINIRE_RAG_MEMORY_MAX_MESSAGES', '100'))
    
    if memory_type == 'conversation':
        human_prefix = os.getenv('REFINIRE_RAG_MEMORY_HUMAN_PREFIX', 'Human')
        ai_prefix = os.getenv('REFINIRE_RAG_MEMORY_AI_PREFIX', 'AI')
        
        return ConversationBufferMemory(
            max_token_limit=max_tokens,
            max_messages=max_messages,
            human_prefix=human_prefix,
            ai_prefix=ai_prefix
        )
    else:
        return BufferMemory(
            max_token_limit=max_tokens,
            max_messages=max_messages
        )


__all__ = [
    'Message',
    'BaseMemory', 
    'BufferMemory',
    'ConversationBufferMemory',
    'create_buffer_memory_from_env'
]