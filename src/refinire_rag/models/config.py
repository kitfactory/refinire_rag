"""
Configuration data models for refinire-rag
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
from ..models.document import Document


@dataclass
class LoadingConfig:
    """Configuration for document loading"""
    
    parallel: bool = True
    max_workers: Optional[int] = None  # None = CPU count
    use_multiprocessing: bool = False  # Thread vs Process
    chunk_size: int = 10  # Batch size for parallel processing
    timeout_per_file: Optional[float] = None  # Timeout per file in seconds
    skip_errors: bool = True  # Continue on individual file errors


@dataclass
class LoadingResult:
    """Result of loading operation"""
    
    documents: List[Document]
    failed_paths: List[str]
    errors: List[Exception]
    total_time_seconds: float
    successful_count: int
    failed_count: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        total = self.successful_count + self.failed_count
        if total == 0:
            return 0.0
        return (self.successful_count / total) * 100.0
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred"""
        return len(self.errors) > 0
    
    def summary(self) -> str:
        """Get loading summary"""
        return (
            f"Loaded {self.successful_count} documents successfully, "
            f"{self.failed_count} failed ({self.success_rate:.1f}% success rate) "
            f"in {self.total_time_seconds:.2f} seconds"
        )