"""Stores the possible evaluation methods to pair with a dataset, e.g., token-based metric, LLM-as-a-judge, etc."""

from typing import List, Literal, Any, Union
from abc import ABC, abstractmethod
from utils.enums import OutputType

class EvalMethod(ABC):
    """Represents an evaluation method."""
    def __init__(self, name: str, description: str, output_type: Literal[OutputType.MC, OutputType.GENERATION, "activations"]):
        self.name = name
        self.description = description
        self.output_type = output_type

    @abstractmethod
    def evaluate(self, output: Any, reference: Any = None, **kwargs) -> Union[float, List[float]]:
        """
        Evaluate the model output against an optional reference (for reference-free metrics, reference may be None).
        Returns a float or list of floats (higher is better).
        """
        pass

    def evaluate_batched(self, outputs: List[Any], references: List[Any] = None, **kwargs) -> List[float]:
        """
        Evaluate a batch of model outputs against their references. Only implement if there exists a way to do evaluation faster if we save all outputs and references and evaluate them together, e.g., with batched inference.
        """
        pass

    def reset(self):
        """
        Reset any internal state in preparation for a new batch of evaluations.
        """
        # Clear any previously gathered outputs/references
        self._gathered_outputs = []  # type: List[Any]
        self._gathered_references = []  # type: List[Any]
        self._gathered_kwargs = [] # type: List[Dict[str, Any]]

    def gather(self, output: Any, reference: Any = None, **kwargs):
        """
        Collect a single model output and its reference for aggregate evaluation later.
        """
        if not hasattr(self, '_gathered_outputs'):
            self.reset()
        self._gathered_outputs.append(output)
        self._gathered_references.append(reference)
        self._gathered_kwargs.append(kwargs)

    def finalize(self) -> List[float]:
        """
        Compute metrics over all gathered outputs and references.
        Default: calls evaluate() on each pair and returns the list of scores.
        Methods may override for more efficient aggregation.
        """
        results: List[float] = []
        for out, ref, kwargs in zip(getattr(self, '_gathered_outputs', []), getattr(self, '_gathered_references', []), getattr(self, '_gathered_kwargs', [])):
            sc = self.evaluate(out, ref, **kwargs)
            if isinstance(sc, list):
                results.extend(sc)
            else:
                results.append(sc)
        return results
