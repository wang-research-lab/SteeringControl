import os
import random
from typing import Dict, List, Literal, Optional, Union
from datasets import Dataset, load_dataset
from data.steering_data import FormattedDataset
from utils.dark_patterns import ANTHROPOMORPHIZATION, BRAND_BIAS, SNEAKING, SYCOPHANCY, USER_RETENTION, DarkPattern
from utils.enums import Concept, FormatMode, OutputType, SteeringFormat
from utils.evaluation_methods.base import EvalMethod
from utils.evaluation_methods.llm_judges import DarkBenchJudge
from utils.evaluation_methods.reference_based import HighestLikelihood, SubstringMatching
from utils.evaluation_methods.reference_free import DarkPatternScore

random.seed(42)
DATA_PATH = "data/datasets/secondary"


class GPQA(FormattedDataset):
    """
    train: Dataset({
        features: ['prompt', 'correct_index'],
        num_rows: 448
    })
    """

    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True
        super().__init__("cleaned_GPQA.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)

    @property
    def deduplication_key(self) -> str:
        return "prompt"

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    def load_dataset(self, select=False, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.MC

    @property
    def concept(self) -> Concept:
        return Concept.EXPERT

    def get_prompt(self, example):
        """Change 1), 2), 3), 4) to A., B., C., D. for consistency with other datasets."""
        prompt = example['prompt']
        prompt = prompt.replace("Choose the correct answer and respond only with the corresponding number:", "Choose the correct answer and respond only with the corresponding letter:")
        return prompt.replace("\n1)", "\nA.").replace("\n2)", "\nB.").replace("\n3)", "\nC.").replace("\n4)", "\nD.")

    def get_answer(self, example):
        """Get the answer in the format 'A', 'B', 'C', 'D'."""
        return f"{chr(ord('A') + example['correct_index'])}"

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt = self.get_prompt(example)
        prompt = [{"role": "user", "content": prompt}]
        # Ground-truth answer, formatted like in training
        reference = self.get_answer(example)
        return {"prompt": prompt, "reference": reference, "incorrect_choices": [f"{chr(ord('A') + i)}" for i in range(3) if i != example['correct_index']]}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            HighestLikelihood("Accuracy (likelihood)", "Accuracy (likelihood) for GPQA", self.output_type),
            SubstringMatching("Accuracy (substring)", "Accuracy (substring matching) for GPQA", self.output_type)
        ]

class ARC_C(FormattedDataset):
    """
    train: Dataset({
    features: ['prompt', 'answerKey'],
    num_rows: 1172
    })
    """

    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True

        super().__init__("cleaned_arcc.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)

    @property
    def deduplication_key(self) -> str:
        return "prompt"

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    def load_dataset(self, select=False, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.MC

    @property
    def concept(self) -> Concept:
        return Concept.COMMONSENSE

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt = [{"role": "user", "content": example['prompt']}]
        return {"prompt": prompt, "reference": example['answerKey'], "incorrect_choices": [f"{i}" for i in ['A', 'B', 'C', 'D'] if i != example['answerKey']]}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            HighestLikelihood("Accuracy (likelihood)", "Accuracy (likelihood) for ARC-C", self.output_type),
            SubstringMatching("Accuracy (substring)", "Accuracy (substring matching) for ARC-C", self.output_type)
        ]

class CMTEST(FormattedDataset):
    """
    train: Dataset({
    features: ['prompt', 'is_short', 'label'],
    num_rows: 3885
    })
    """

    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True

        super().__init__("cleaned_cmtest.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)

    @property
    def deduplication_key(self) -> str:
        return "prompt"

    @property
    def stratification_key(self) -> Optional[str]:
        return 'is_short'

    def load_dataset(self, select=False, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)
        # Add index column for deterministic A/B assignment
        self.dataset = self.dataset.add_column("_index", list(range(len(self.dataset))))

    @property
    def paired(self) -> bool:
        return True

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.MC

    @property
    def concept(self) -> Concept:
        return Concept.MORALITY

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt = example['prompt']

        # Deterministically assign "wrong" and "not wrong" to A/B based on row index
        options = ["wrong", "not wrong"]
        shuffled = options.copy()
        # Use row index as seed to ensure deterministic shuffle across runs
        prompt_seed = example.get('_index', 0)
        rng = random.Random(prompt_seed)
        rng.shuffle(shuffled)

        # Create the replacement string with randomized A/B assignment
        orig_string = "The answer should be exact 'wrong' or 'not wrong'."
        replacement_string = f"The answer should be exact 'A' for '{shuffled[0]}' or 'B' for '{shuffled[1]}'. "  # missing a space if we look at the prompt.
        prompt = prompt.replace(orig_string, replacement_string)
        prompt = [{"role": "user", "content": prompt}]

        # Determine correct label based on the randomized assignment
        true_answer = "wrong" if example['label'] == '[1]' else "not wrong"
        correct_label = chr(ord('A') + shuffled.index(true_answer))
        incorrect_label = chr(ord('A') + (1 - shuffled.index(true_answer)))  # The other option

        return {"prompt": prompt, "reference": correct_label, "incorrect_choices": [incorrect_label]}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            HighestLikelihood("Accuracy (likelihood)", "Accuracy (likelihood) for CMTEST", self.output_type),
            SubstringMatching("Accuracy (substring)", "Accuracy (substring matching) for CMTEST", self.output_type)
        ]

class GSM8K(FormattedDataset):
    """
    train: Dataset({
    features: ['question', 'answer'],
    num_rows: 500
    })
    """

    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True

        super().__init__("cleaned_gsm8k.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)
        
    @property
    def deduplication_key(self) -> str:
        return 'question'

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.GENERATION

    @property
    def concept(self) -> Concept:
        return Concept.MATH

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt = [{"role": "user", "content": example['question']}]
        return {"prompt": prompt, "reference": str(example['answer'])}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            SubstringMatching("GSM8K", "Checks if the answer matches reference answer.", self.output_type)
        ]

class TruthfulQA(FormattedDataset):
    """
    train: Dataset({
    features: ['prompt', 'correct_label'],
    num_rows: 790
    })
    """

    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True

        super().__init__("cleaned_truthfulqa.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)

    @property
    def deduplication_key(self) -> str:
        return 'prompt'

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.MC

    @property
    def concept(self) -> Concept:
        return Concept.MISONCEPTIONS

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt=[{"role": "user", "content": example['prompt']}]
        return {"prompt": prompt, "reference": example['correct_label'], "incorrect_choices": [i for i in ['A', 'B'] if i != example['correct_label']]}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            HighestLikelihood("TruthfulQA (likelihood)", "Checks via likelihood if the output letter matches reference answer.", self.output_type),
            SubstringMatching("TruthfulQA (substring)", "Checks via substring matching if the output letter matches reference answer.", self.output_type)
        ]

class Twinviews(FormattedDataset):
    """
    train: Dataset({
    features: ['prompts', 'topic', 'left_answer'],
    num_rows: 13855
    })
    """

    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True
        super().__init__("cleaned_twinviews.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)

    @property
    def deduplication_key(self) -> str:
        return 'prompts'

    @property
    def stratification_key(self) -> Optional[str]:
        return 'topic'

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.MC

    @property
    def concept(self) -> Concept:
        return Concept.POLITICS

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt = [{"role": "user", "content": example['prompts']}]
        return {"prompt": prompt, "reference": example['left_answer'], "incorrect_choices": [i for i in ['A', 'B'] if i != example['left_answer']]}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            HighestLikelihood("Twinviews (likelihood)", "Checks via likelihood if the output letter matches reference answer.", self.output_type),
            SubstringMatching("Twinviews (substring)", "Checks via substring matching if the output letter matches reference answer.", self.output_type)
        ]
    
class EnronEmail(FormattedDataset):
    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        super().__init__("cleaned_enronemail.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=True)

    @property
    def deduplication_key(self) -> str:
        return 'prompt'

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        if self.n is not None:
            self.select_n(self.n)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.GENERATION

    @property
    def concept(self) -> Concept:
        return Concept.LEAKAGE

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        prompt = [{"role": "user", "content": example['prompt']}]
        return {"prompt": prompt, "reference": example['email']}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            SubstringMatching("EnronEmail", "Checks to see if the output email matches reference.", self.output_type)
        ]

class DarkBench(FormattedDataset):
    """
    train: Dataset({
        features: ['prompt', 'behavior'],
        num_rows: 448
    })
    """

    def __init__(self, n, format_mode: FormatMode, issue: DarkPattern, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        self.secondary = True
        super().__init__("cleaned_darkbench.csv", "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset, secondary=self.secondary)
        self.issue = issue

    @property
    def deduplication_key(self) -> str:
        return "prompt"

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)

    @property
    def paired(self) -> bool:
        return False

    @property
    def output_type(self) -> Literal[OutputType.MC, OutputType.GENERATION]:
        return OutputType.GENERATION

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        pass

    def format_single_test(self, example) -> Dict[str, str]:
        return {"prompt": [{"role": "user", "content": example['prompt']}], "behavior": self.issue}

    def get_evaluation_methods(self) -> List[EvalMethod]:
        return [
            DarkBenchJudge("DarkBenchJudge", "LLM-as-a-judge for Dark Bench", self.output_type)
        ]

class DarkBenchAnthro(DarkBench):
    def __init__(self, n, format_mode:FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):

        # use a separate variable here to avoid having the first n selected first and then filtered by concept
        self.force_select = n
        super().__init__(n, format_mode, ANTHROPOMORPHIZATION, format , _existing_dataset)

    @property
    def concept(self) -> Concept:
        return Concept.ANTHROPOMORPHIZATION

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        self.dataset = self.dataset.filter(lambda x: x['behavior'] == "Anthropomorphization")
        if self.force_select is not None:
            self.select_n(self.force_select)

class DarkBenchBrandBias(DarkBench):
    def __init__(self, n, format_mode:FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):

        self.force_select = n
        super().__init__(n, format_mode, BRAND_BIAS, format, _existing_dataset)

    @property
    def concept(self) -> Concept:
        return Concept.BRAND_BIAS

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        self.dataset = self.dataset.filter(lambda x: x['behavior'] == "Brand bias")
        if self.force_select is not None:
            self.select_n(self.force_select)

class DarkBenchSynchopancy(DarkBench):
    def __init__(self, n, format_mode:FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):

        self.force_select = n
        super().__init__(n, format_mode, SYCOPHANCY, format, _existing_dataset)

    @property
    def concept(self) -> Concept:
        return Concept.SYCOPHANCY

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        self.dataset = self.dataset.filter(lambda x: x['behavior'] == "Sycophancy")
        if self.force_select is not None:
            self.select_n(self.force_select)

class DarkBenchSneaking(DarkBench):
    def __init__(self, n, format_mode:FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):

        self.force_select = n
        super().__init__(n, format_mode, SNEAKING, format, _existing_dataset)

    @property
    def concept(self) -> Concept:
        return Concept.SNEAKING

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        self.dataset = self.dataset.filter(lambda x: x['behavior'] == "Sneaking")
        if self.force_select is not None:
            self.select_n(self.force_select)

class DarkBenchRetention(DarkBench):
    def __init__(self, n, format_mode:FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):

        self.force_select = n
        super().__init__(n, format_mode, USER_RETENTION, format, _existing_dataset)

    @property
    def concept(self) -> Concept:
        return Concept.USER_RETENTION

    def load_dataset(self, select=True, subset=None):
        super().load_dataset(select)
        self.dataset = self.dataset.filter(lambda x: x['behavior'] == "User retention")
        if self.force_select is not None:
            self.select_n(self.force_select)
