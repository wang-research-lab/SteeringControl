from datasets import load_dataset, Dataset
from typing import List, Optional, Dict, Union
from data.steering_data import FormattedDataset, REPE_FORMAT, REPE_FORMAT_INSTR_ONLY, CAA_FORMAT_INSTR_ONLY
from utils.enums import SteeringFormat, FormatMode, OutputType, Concept
from utils.evaluation_methods.llm_judges import LlamaGuard
from utils.evaluation_methods.reference_free import RefusalSubstringMatching
import random

random.seed(42)  # For reproducibility

class Alpaca(FormattedDataset):
    """
    train: Dataset({
        features: ['instruction', 'input', 'output', 'text'],
        num_rows: 52002
    })

    We follow Arditi et al. and only use the instructions that don't have inputs (https://github.com/andyrdt/refusal_direction/blob/main/dataset/generate_datasets.ipynb).
    """
    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, paired_concept: Concept = Concept.REFUSAL, _existing_dataset: Optional[Dataset] = None):
        super().__init__("tatsu-lab/alpaca", "train", n, format_mode, format, _existing_dataset=_existing_dataset)
        self.paired_concept = paired_concept  # Concept to use for paired datasets, e.g., RepE or CAA. This will ensure that the templates are the same for the positive refusal dataset and the negative (this) dataset. Can also be extended to other concepts.

    def load_dataset(self):
        super().load_dataset(select=False)
        self.dataset = self.dataset.filter(lambda x: x['input'].strip() == "")  # Filter out examples with input, as per Arditi et al.
        self.select_n(self.n)

    @property
    def deduplication_key(self) -> str:
        return "instruction"

    @property
    def stratification_key(self) -> Optional[str]:
        return None  # No stratification key for Alpaca.

    @property
    def output_type(self) -> str:
        return OutputType.GENERATION
    
    @property
    def concept(self) -> Concept:
        return Concept.REFUSAL

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        prompt = example['instruction']
        if self.format == SteeringFormat.DEFAULT:
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}  # ideally we use example['output'] but we don't have for SaladBench so not doing it.
            ]
        elif self.format == SteeringFormat.ARDITI:
            return [
                {"role": "user", "content": prompt}, 
                {"role": "assistant", "content": ""}
            ]
        elif self.format == SteeringFormat.CAA:
            # We will make fake answers for CAA by adding a template like in RepE.
            yes_first = random.choice([True, False])  # Randomly choose whether to put 'Yes' or 'No' first.
            choices = "A. Yes\nB. No" if yes_first else "A. No\nB. Yes"
            answer = "B" if yes_first else "A"
            template = CAA_FORMAT_INSTR_ONLY + choices
            return [
                {"role": "user", "content": template.format(instruction=prompt)},
                {"role": "assistant", "content": answer}
            ]
        elif self.format == SteeringFormat.REPE:
            out = []
            for msg in REPE_FORMAT_INSTR_ONLY:
                content = msg["content"].format(concept=self.paired_concept.value, instruction=prompt)
                out.append({"role": msg["role"], "content": content})
            return out

    def format_single_test(self, example) -> Dict[str, str]:
        """Return prompt and reference for evaluation."""
        input_str = f"\nInput: {example['input']}" if example['input'] else ""
        prompt = [{"role": "user", "content": example['instruction'] + input_str}]
        # Ground-truth continuation
        reference = example['output']
        return {"prompt": prompt, "reference": reference}
        
    def get_evaluation_methods(self):
        return [
            LlamaGuard("alpaca", "classifying harmfulness", self.output_type),
        ]

class SaladBench(FormattedDataset):
    """SALAD-Bench safety benchmark. Has 21k test samples in a detailed hierarchy of 6 domains, 16 tasks, 66 categories. Also has attack-enhanced, defense-enhanced, and MC questions.

    """
    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        super().__init__("SC_saladbench_with_category_columns.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset)

    @property
    def deduplication_key(self) -> str:
        return "prompt"

    @property
    def stratification_key(self) -> Optional[str]:
        return "category_secondary"  # More specificity than 1st level but not as specific as 3rd level.

    @property
    def paired(self) -> bool:
        return False  # Use Alpaca for negative.
        
    @property
    def output_type(self) -> str:
        return OutputType.GENERATION
    
    @property
    def concept(self) -> Concept:
        return Concept.REFUSAL
    
    def format_single_train(self, example):
        prompt = example['prompt']
        if self.format == SteeringFormat.DEFAULT:
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}
            ]
        elif self.format == SteeringFormat.ARDITI:
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""}
            ]
        elif self.format == SteeringFormat.CAA:
            # We will make fake answers for CAA by adding a template like in RepE.
            # Note we need to extend this to Alpaca as well.
            yes_first = random.choice([True, False])  # Randomly choose whether to put 'Yes' or 'No' first.
            choices = "A. Yes\nB. No" if yes_first else "A. No\nB. Yes"
            answer = "A" if yes_first else "B"
            template = CAA_FORMAT_INSTR_ONLY + choices
            return [
                {"role": "user", "content": template.format(instruction=prompt)},
                {"role": "assistant", "content": answer}
            ]
        elif self.format == SteeringFormat.REPE:
            out = []
            for msg in REPE_FORMAT_INSTR_ONLY:
                content = msg["content"].format(concept=self.concept.value, instruction=prompt)
                out.append({"role": msg["role"], "content": content})
            return out

    def format_single_test(self, example) -> Dict[str, str]:
        """Return prompt and reference (None if not applicable) for evaluation."""
        prompt = [{"role": "user", "content": example['prompt']}]
        return {"prompt": prompt, "reference": None}  # We will LLM-as-a-judge for evaluation, so no reference is needed.

    def get_evaluation_methods(self):
        return [
            LlamaGuard("saladbench", "classifying harmfulness", self.output_type),
        ]
