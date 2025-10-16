"""Evaluation metrics for reference-based methods, e.g., substring matching."""

import torch
import re
import string
from typing import List, Any, Literal
from utils.evaluation_methods.base import EvalMethod
from utils.enums import OutputType

def normalize_answer(s):
    """From FaithEval: Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def normalize_for_mc(s):
    """Normalize answer for multiple choice questions."""
    # Lowercase and remove punctuation
    s = s.lower()
    exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
    s = ''.join(ch if ch not in exclude else ' ' for ch in s)
    # Remove extra whitespace
    return ' '.join(s.split()).strip()

class SubstringMatching(EvalMethod):
    async def evaluate(self, output: str, reference: List[str], **kwargs) -> float:
        """Match substring(s) in text_to_evaluate against ground_truths."""
        if self.output_type == OutputType.GENERATION:
            assert len(reference) > 0, "Ground truths must be provided for substring matching evaluation."
            # print(f"Reference: {reference}")
            # print(f"Output: {output}")
            if any(phrase in normalize_answer(output) for phrase in reference):
                # print("Substring match found. Score: 1.0")
                return 1.0
            # print("No substring match found. Score: 0.0")
            return 0.0
        elif self.output_type == OutputType.MC:
            if isinstance(reference, list):
                assert len(reference) == 1, "For MCQ, there should be exactly one ground truth."
                target_answer = reference[0].strip()
            else:  # toxigen
                target_answer = reference.strip()
            # print(f"Reference: {target_answer}")
            # print(f"Output: {output}")
            
            # Check for exact match first
            if target_answer in output:
                # print(f"Exact match found for '{target_answer}'. Score: 1.0")
                return 1.0
            
            # Check for various MC patterns
            import re
            
            # Pattern 1: Single capital letter (A, B, C, D)
            if re.search(rf'\b{re.escape(target_answer)}\b', output, re.IGNORECASE):
                # print(f"Word boundary match found for '{target_answer}'. Score: 1.0")
                return 1.0
            
            # Pattern 2: Letter with period (A., B., C., D.)
            if re.search(rf'\b{re.escape(target_answer)}\.\b', output, re.IGNORECASE):
                # print(f"Letter with period match found for '{target_answer}.'. Score: 1.0")
                return 1.0
            
            # Pattern 3: Letter in parentheses ((A), (B), (C), (D))
            if re.search(rf'\(\s*{re.escape(target_answer)}\s*\)', output, re.IGNORECASE):
                # print(f"Parentheses match found for '({target_answer})'. Score: 1.0")
                return 1.0
            
            # print("No substring match found. Score: 0.0")
            return 0.0

class HighestLikelihood(EvalMethod):
    """Determine most likely answer from the choices using the logits of the first token.
    
    Higher (1.0) is correct (desired), lower (0.0) is incorrect (undesired).
    
    """
    async def evaluate(self, output: str, reference: str, incorrect_choices: List[str], tokenizer: Any, first_token_logits: torch.Tensor, metric_type: Literal["acc", "diff", "norm_diff"] = "acc", **kwargs) -> float:
        """Given the logits of the output, return the accuracy, i.e., whether the reference answer logit is > the max logit of the incorrect choices."""
        assert self.output_type == OutputType.MC, "Highest likelihood evaluation is only applicable for MCQ outputs." 
        assert reference not in incorrect_choices, "Reference answer should not be in the incorrect choices."
        # tokenize the reference and incorrect choices
        tokenized_best = tokenizer.encode(reference, add_special_tokens=False)
        tokenized_incorrect = torch.tensor([tokenizer.encode(choice, add_special_tokens=False) for choice in incorrect_choices])
        assert len(tokenized_best) == 1, "Reference answer should be a single token."
        assert len(tokenized_incorrect) == len(incorrect_choices), "Incorrect choices should be a list of tokens."
        # Get the logits for the reference and incorrect choices
        ref_logits = first_token_logits[tokenized_best[0]]
        inc_logits = first_token_logits[tokenized_incorrect]
        if metric_type == "acc":
            return 1.0 if ref_logits > inc_logits.max() else 0.0
        else:
            raise NotImplementedError("difference metrics are not yet implemented. We can do something like [TruthfulQA MC2](https://github.com/yfqiu-nlp/sea-llm/blob/main/src/benchmark_evaluation/bbq_eval.py#L173) or [Analysing the Generalisation and Reliability of Steering Vectors](https://arxiv.org/pdf/2407.12404) or CAA.")
