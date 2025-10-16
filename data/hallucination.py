from datasets import load_dataset, Dataset
from typing import List, Optional, Dict, Union, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
import os
import json
import random
from data.steering_data import FormattedDataset, REPE_FORMAT, DATA_PATH, REPE_FORMAT_CONTEXT
from utils.enums import SteeringFormat, FormatMode, OutputType, Concept, MC_STR
from utils.call_openai import async_call_openai
from utils.evaluation_methods.reference_based import SubstringMatching, HighestLikelihood
from utils.evaluation_methods.llm_judges import PreciseQAEval

random.seed(42)  # For reproducibility

### Intrinsic train/val/test datasets for hallucination ###
class FaithEval(FormattedDataset, ABC):
    """Suggested by HalluLens for measuring intrinsic hallucination when input is noisy or contradicting world knowledge. There are xxx total samples.

    Split into three subsets: unanswerable (generation), inconsistent (generation), and counterfactual (multiple choice).
    
    Source of data: https://huggingface.co/collections/Salesforce/faitheval-benchmark-66ff102cda291ca0875212d4.
    Prompts and evaluation come from here: https://github.com/SalesforceAIResearch/FaithEval?tab=readme-ov-file#-quick-start.
    
    """
    def __init__(self, name, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        super().__init__(name, "test", n, format_mode, format, _existing_dataset=_existing_dataset)
        self.task_specific_prompt = ""
        self.prompt_format = """You are an expert in retrieval question answering.
Please respond with the exact answer only. Do not be verbose or provide extra information.
{task_specific_prompt}
Context: {context}
Question: {question}
Answer:"""

    @property
    def deduplication_key(self) -> str:
        return "question"  # Note this should be no problem since has indep qid. just here for completeness.

    @property
    def stratification_key(self) -> Optional[str]:
        return None

    @property
    def paired(self) -> bool:
        return True
    
    @property
    def concept(self) -> Concept:
        return Concept.HALLUCINATION

    def format_question(self, example):
        if self.output_type == OutputType.GENERATION:
            return example['question']
        elif self.output_type == OutputType.MC:
            # Note this style wasn't explicitly stated but is standard for multiple choice datasets.
            choices = example['choices']
            choices_str = "\n".join(f"{label}. {text}" for label, text in zip(choices['label'], choices['text']))
            return example['question'] + "\n" + choices_str

    @abstractmethod
    def format_answer(self, example) -> Tuple[str, str]:
        """Format the answer based on the output type. This should be implemented in subclasses. Will return pos_answer, neg_answer."""
        pass

    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        question = self.format_question(example)
        pos_answer, neg_answer = self.format_answer(example)
        assert pos_answer != neg_answer, "Positive and negative answers must be different."
        prompt = self.prompt_format.format(
            task_specific_prompt=self.task_specific_prompt,
            context=example['context'],
            question=question
        )
        if self.format == SteeringFormat.CAA:
            if self.output_type == OutputType.MC:
                return [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": pos_answer}  # example['answerKey']}  # This will be the letter only.
                    ],
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": neg_answer}
                    ]
                ]
            else:
                template = "Given the context, which answer to the question displays hallucination?\nContext: {context}\nQuestion: {question}\n"
                hallucinated_first = random.choice([True, False])  # Randomly choose whether to put the hallucinated answer first.
                choices = "A. {answer_a}\nB. {answer_b}" if hallucinated_first else "A. {answer_b}\nB. {answer_a}"
                choices = choices.format(
                    answer_a=pos_answer,
                    answer_b=neg_answer
                )
                prompt = template.format(context=example['context'], question=question) + choices
                return [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "A" if hallucinated_first else "B"}  # We assume the first answer is the hallucinated one.
                    ],
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "B" if hallucinated_first else "A"}  # The other answer.
                    ]
                ]
        pair = []
        for answer in [pos_answer, neg_answer]:
            if self.format == SteeringFormat.DEFAULT:
                pair.append([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ])
            elif self.format == SteeringFormat.ARDITI:
                pair.append([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ])
            elif self.format == SteeringFormat.REPE:
                out = []
                for msg in REPE_FORMAT_CONTEXT:
                    content = msg["content"].format(concept=self.concept.value, question_and_context=f"Context: {example['context']}\nQuestion: {question}", answer=answer)
                    out.append({"role": msg["role"], "content": content})
                pair.append(out)
        return pair

    def format_single_test(self, example) -> Union[str, List[Dict[str, str]]]:
        """Note: Should return {"prompt": ..., "reference": ...} for evaluation, but we do this in subclasses."""
        return [
            {"role": "user", "content": self.prompt_format.format(
                task_specific_prompt=self.task_specific_prompt,
                context=example['context'],
                question=self.format_question(example)
            )}
        ]

    def get_evaluation_methods(self):
        return [
            SubstringMatching("FaithEval", "Checks if the answer contains valid phrases.", self.output_type),
        ]

class FaithEvalUnanswerable(FaithEval):
    """FaithEval subset for unanswerable questions. Uses generation output type."""
    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT,
                 _existing_dataset: Optional[Dataset] = None):
        super().__init__("Salesforce/FaithEval-unanswerable-v1.0", n, format_mode, format, _existing_dataset=_existing_dataset)
        self.task_specific_prompt = "If there is no information available from the context, the answer should be 'unknown'. "
        self.neg_answer_file = os.path.join(DATA_PATH, "faitheval_unanswerable_negative_answers.json")
        if os.path.exists(self.neg_answer_file):
            with open(self.neg_answer_file, "r") as f:
                self.neg_answer_dict = json.load(f)
        else:
            raise FileNotFoundError(f"Negative answer file {self.neg_answer_file} does not exist. Please generate it first using the `generate_negative_answers` method (example in the main method of this file).")

    @property
    def output_type(self) -> str:
        return OutputType.GENERATION

    async def _generate_one(self, example, model, system_prompt, semaphore):
        async with semaphore:
            context = example['context']
            question = example['question']
            out = await async_call_openai(
                model=model,
                system_prompt=system_prompt,
                messages=[
                    {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
                ],
                max_tokens=50,
                temperature=0.0
            )
            return example['qid'], out.strip()

    async def generate_negative_answers(self, model="gpt-4.1-mini") -> List[str]:
        """Generate negative answers for the unanswerable questions."""
        system_prompt = "Please respond with the exact answer only. Do not be verbose or provide extra information. It should be as succinct as possible. Always give your best guess if the answer is not in the passage, using your outside knowledge. Think of it like a test where guessing is better than not answering."

        semaphore = asyncio.Semaphore(20)  # Limit the number of concurrent requests to avoid hitting rate limits.
        tasks = [
            self._generate_one(example, model, system_prompt, semaphore=semaphore)
            for example in self.dataset
        ]
        results = {}
        for fut in tqdm_asyncio.as_completed(tasks, desc="Generating negative answers"):
            qid, answer = await fut
            results[qid] = answer

        os.makedirs(DATA_PATH, exist_ok=True)
        with open(self.neg_answer_file, "w") as f:
            json.dump(results, f, indent=4)

        return results
        
    def format_answer(self, example) -> Tuple[str, str]:
        # answers is a list like [ "unanswerable", "unknown", "no answer", "no information", "not", "unclear" ]. We randomly select one of these for the pos answer. Note this is a superset of what they use in https://github.com/SalesforceAIResearch/FaithEval?tab=readme-ov-file#-quick-start for eval. We stick with this.
        pos_answer = random.choice(example['answers'])
        neg_answer = self.neg_answer_dict[example['qid']]
        return pos_answer, neg_answer

    def format_single_test(self, example):
        prompt = super().format_single_test(example)
        reference = [ "unanswerable", "unknown", "no answer", "no information", "not", "unclear" ]
        return {"prompt": prompt, "reference": reference}

class FaithEvalInconsistent(FaithEval):
    """FaithEval subset for inconsistent questions. Uses generation output type."""
    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT, 
                 _existing_dataset: Optional[Dataset] = None):
        super().__init__("Salesforce/FaithEval-inconsistent-v1.0", n, format_mode, format, _existing_dataset=_existing_dataset)
        self.task_specific_prompt = "If there is conflict information or multiple answers from the context, the answer should be 'conflict'."

    @property
    def output_type(self) -> str:
        return OutputType.GENERATION

    def format_answer(self, example) -> Tuple[str, str]:
        pos_answer = random.choice(['conflict', 'multiple answers', 'disagreement', 'inconsistent', 'contradictory', 'contradiction', 'inconsistency', 'two answers', '2 answers', 'conflicting'])  # from https://github.com/SalesforceAIResearch/FaithEval?tab=readme-ov-file#-quick-start
        neg_answer = random.choice(example['answers'])  # The two possible answers.
        return pos_answer, neg_answer

    def format_single_test(self, example):
        prompt = super().format_single_test(example)
        reference = ['conflict', 'multiple answers', 'disagreement', 'inconsistent', 'contradictory', 'contradiction', 'inconsistency', 'two answers', '2 answers', 'conflicting']
        return {"prompt": prompt, "reference": reference}

class FaithEvalCounterfactual(FaithEval):
    """FaithEval subset for counterfactual questions. Uses multiple choice output type. Uses no additional task instructions but need to override prompt format for MC."""
    def __init__(self, n, format_mode: FormatMode, format: SteeringFormat = SteeringFormat.DEFAULT,
                 _existing_dataset: Optional[Dataset] = None):
        super().__init__("Salesforce/FaithEval-counterfactual-v1.0", n, format_mode, format, _existing_dataset=_existing_dataset)
        self.prompt_format = """You are an expert in retrieval question answering.""" + f"""
{MC_STR}""" + """
Context: {context}
Question: {question}
Answer:"""

    @property
    def output_type(self) -> str:
        return OutputType.MC

    def format_answer(self, example) -> Tuple[str, str]:
        pos_answer = example['answerKey']
        # neg answer is any of the other choices that is not the pos_answer.
        neg_answer = random.choice([choice for choice in example['choices']['label'] if choice != pos_answer])
        return pos_answer, neg_answer

    def format_single_test(self, example):
        prompt = super().format_single_test(example)
        reference = example['answerKey']  #  + ". " + example['answer']
        incorrect_choices = [choice for choice in example['choices']['label'] if choice != reference]
        # print(f"Prompt: {prompt}")
        # print(f"Reference: {reference}")
        return {"prompt": prompt, "reference": reference, "incorrect_choices": incorrect_choices}

    def get_evaluation_methods(self):
        return [
            HighestLikelihood("FaithEval (likelihood)", "Checks likelihood.", self.output_type),
            SubstringMatching("FaithEval (substring)", "Checks substring matching.", self.output_type),
        ]

### Extrinsic hallucination datasets ###
class PreciseWiki(FormattedDataset):
    """Constructed in HalluLens for measuring extrinsic hallucination on short and fact-seeking queries based on knowledge from the training data. A new test dataset is automatically generated using the HalluLens pipeline and Wikipedia. We start with 10k samples from two generation runs."""
    def __init__(self, n, format_mode: FormatMode, name="qa_goodwiki_Llama-3.1-70B-Instruct_dynamic.jsonl", format: SteeringFormat = SteeringFormat.DEFAULT, _existing_dataset: Optional[Dataset] = None):
        super().__init__(name, "train", n, format_mode, format, hf=False, _existing_dataset=_existing_dataset)
        self.neg_answer_file = os.path.join(DATA_PATH, "precisewiki_negative_answers.json")
        if os.path.exists(self.neg_answer_file):
            with open(self.neg_answer_file, "r") as f:
                self.neg_answer_dict = json.load(f)
        else:
            raise FileNotFoundError(f"Negative answer file {self.neg_answer_file} does not exist. Please generate it first using the `generate_negative_answers` method (example in the main method of this file).")

    @property
    def deduplication_key(self) -> str:
        return "title"  # stricter deduplication so that we don't overlap using the same pages.

    @property
    def stratification_key(self) -> Optional[str]:
        return "h_score_cat"
    
    async def _generate_one(self, example, model, system_prompt, semaphore):
        async with semaphore:
            prompt = example['prompt']
            out = await async_call_openai(
                model=model,
                system_prompt=system_prompt,
                messages=[
                    {"role": "user", "content": f"Question: {prompt}\nCorrect Answer: {example['answer']}\nIncorrect Answer:"}
                ],
                max_tokens=50,
                temperature=0.0
            )
            return prompt, out.strip(), example['answer']

    async def generate_negative_answers(self, model="gpt-4.1-mini") -> List[str]:
        """Generate negative answers for the unanswerable questions."""
        system_prompt = "Please respond with an incorrect answer that differs from the correct answer but is still reasonable in the context of the question. Do not be verbose or provide extra information. It should be as succinct as possible."

        semaphore = asyncio.Semaphore(20)  # Limit the number of concurrent requests to avoid hitting rate limits.
        tasks = [
            self._generate_one(example, model, system_prompt, semaphore=semaphore)
            for example in self.dataset
        ]
        results = {}
        for fut in tqdm_asyncio.as_completed(tasks, desc="Generating negative answers"):
            qid, incorrect_answer, correct_answer = await fut
            results[qid] = {"incorrect_answer": incorrect_answer, "correct_answer": correct_answer}

        os.makedirs(DATA_PATH, exist_ok=True)
        with open(self.neg_answer_file, "w") as f:
            json.dump(results, f, indent=4)

        return results

    @property
    def paired(self) -> bool:
        return True
        
    @property
    def output_type(self) -> str:
        return OutputType.GENERATION
    
    @property
    def concept(self) -> Concept:
        return Concept.HALLUCINATION
    
    def format_single_train(self, example) -> Union[str, List[Dict[str, str]]]:
        prompt = example['prompt']
        pos_answer = example['answer']
        neg_answer = self.neg_answer_dict[prompt]['incorrect_answer']
        if self.format == SteeringFormat.CAA:
            template = "Which answer to the question displays hallucination?\nQuestion: {question}\n"
            hallucinated_first = random.choice([True, False])  # Randomly choose whether to put the hallucinated answer first.
            choices = "A. {answer_a}\nB. {answer_b}" if hallucinated_first else "A. {answer_b}\nB. {answer_a}"
            choices = choices.format(
                answer_a=pos_answer,
                answer_b=neg_answer
            )
            prompt = template.format(question=prompt) + choices
            return [
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "A" if hallucinated_first else "B"}  # We assume the first answer is the hallucinated one.
                ],
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "B" if hallucinated_first else "A"}  # The other answer.
                ]
            ]
        pair = []
        for answer in [pos_answer, neg_answer]:
            if self.format == SteeringFormat.DEFAULT:
                pair.append([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ])
            elif self.format == SteeringFormat.ARDITI:
                pair.append([
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ])
            elif self.format == SteeringFormat.REPE:
                out = []
                for msg in REPE_FORMAT:
                    content = msg["content"].format(concept=self.concept.value, question=prompt, answer=answer)
                    out.append({"role": msg["role"], "content": content})
                pair.append(out)
        return pair

    def format_single_test(self, example) -> Union[str, List[Dict[str, str]]]:
        return {
            "prompt": [ {"role": "user", "content": example['prompt']} ],
            "reference": example['answer'],
        }

    def get_evaluation_methods(self):
        return [
            PreciseQAEval(),
        ]

if __name__=="__main__":
    # If FaithEval unanswerable negative answers are not generated, we generate here.
    faith_eval_unanswerable = FaithEvalUnanswerable(n=None, format_mode=FormatMode.TRAIN)
    if not os.path.exists(faith_eval_unanswerable.neg_answer_file):
        print("Generating negative answers for FaithEval unanswerable dataset...")
        asyncio.run(faith_eval_unanswerable.generate_negative_answers(model="gpt-4.1-mini"))
    
    # If PreciseWiki negative answers are not generated, we generate here.
    precise_wiki = PreciseWiki(n=None, format_mode=FormatMode.TRAIN)
    if not os.path.exists(precise_wiki.neg_answer_file):
        print("Generating negative answers for PreciseWiki dataset...")
        asyncio.run(precise_wiki.generate_negative_answers(model="gpt-4.1-mini"))
