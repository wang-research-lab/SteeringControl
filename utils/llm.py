"""Abstract base class for language models used in steering pipeline."""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Literal, List, Any

class LLM(ABC):
    """Abstract base class for language models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the language model."""
        pass

    @abstractmethod
    def __call__(self, prompt: str, **kwargs) -> Any:
        """Call the language model with given prompt and return output."""
        pass