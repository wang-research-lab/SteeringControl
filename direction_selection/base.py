"""Abstract base class for direction selection component."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
from jaxtyping import Float, Int
from torch import Tensor
import torch
import logging
from tqdm.auto import tqdm
from data.steering_data import FormattedDataset, SteeringTrainData
from utils.steering_utils import ActivationLocation
from utils.intervention_llm import InterventionLLM

class DirectionSelector(ABC):
    """Abstract base class for selecting the best direction given training and validation data."""

    def __init__(self,
                 applier: Any,
                 application_locations: List[Any],
                 factors: List[float] = None,
                 result_path: str = "results.csv",
                 include_generation_loc: bool = False,
                 generation_pos: Any = None,
                 cumulative_layers: bool = False,
                 reverse: bool = None,
                 _orig_layers: Any = None,
                 use_kl_divergence_check: bool = False,
                 kl_threshold: float = 0.1,
                 harmless_validation_data: Any = None,
                 **kwargs):
        """
        Args:
            applier: The DirectionApplier to use when applying interventions.
            application_locations: List of ApplicationLocation indicating where to apply directions.
            factors: List of steering coefficients to search over.
            result_path: Path to save results.
            include_generation_loc: Include application location equal to each direction's generation loc.
            generation_pos: Default position to use for generated-loc application.
            cumulative_layers: Whether to apply steering cumulatively across all layers up to the target layer.
            reverse: Whether we need to switch the default direction of the intervention based on whether we amplify or reduce the effect in the dataset.
            _orig_layers: Original layers to use for cumulative layers (if applicable).
            use_kl_divergence_check: Whether to filter candidates using KL divergence against harmless examples.
            kl_threshold: Maximum allowed KL divergence from baseline for harmless examples.
            harmless_validation_data: Optional harmless datasets for KL divergence checking.
            **kwargs: Additional selector-specific parameters.
        """
        self.applier = applier
        self.application_locations = application_locations
        self.factors = sorted(factors) if factors is not None else factors
        self.result_path = result_path
        self.include_generation_loc = include_generation_loc
        self.generation_pos = generation_pos
        self.cumulative_layers = cumulative_layers
        self.reverse = reverse
        self._orig_layers = _orig_layers
        self.use_kl_divergence_check = use_kl_divergence_check
        self.kl_threshold = kl_threshold
        self.harmless_validation_data = harmless_validation_data
        
        # Store additional kwargs for subclass-specific parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _generate_answer_conditioned_prompts(self, validation_datasets: List[FormattedDataset]) -> Tuple[List[str], List[str]]:
        """Generate answer-conditioned prompts from validation datasets.
        
        For each example in validation data, creates two versions:
        - Positive prompt: base prompt + positive answer
        - Negative prompt: base prompt + negative answer
        
        Args:
            validation_datasets: List of validation datasets with multiple choice questions.
            
        Returns:
            Tuple of (positive_prompts, negative_prompts)
        """
        logging.info("Generating answer-conditioned prompts...")
        
        pos_prompts = []
        neg_prompts = []
        
        for validation_data in validation_datasets:
            for example in tqdm(validation_data.dataset, desc=f"Generating prompts for {validation_data.name}"):
                try:
                    # Use format_single_train universally for both MC and GENERATION
                    # This ensures proper chat formatting and leverages each dataset's own formatting logic
                    try:
                        # Get the training format 
                        train_result = validation_data.format_single_train(example)
                        
                        if hasattr(validation_data, 'paired') and validation_data.paired:
                            # Paired datasets return [pos_example, neg_example]
                            if len(train_result) == 2:
                                pos_prompts.append(train_result[0])
                                neg_prompts.append(train_result[1])
                            else:
                                raise ValueError("Paired dataset must return exactly two examples (you may need a negative validation dataset here).")
                        else:
                            # Unpaired datasets (like refusal) return single conversation
                            # Use the same conversation for both pos and neg (distinction comes from dataset type)
                            pos_prompts.append(train_result)
                            neg_prompts.append(train_result)
                            
                    except Exception as e:
                        logging.warning(f"Error using format_single_train for example: {e}")
                        continue
                        
                except Exception as e:
                    logging.warning(f"Error processing example: {e}")
                    continue
        
        logging.info(f"Generated {len(pos_prompts)} positive and {len(neg_prompts)} negative prompts")
        return pos_prompts, neg_prompts
    
    def _generate_prompts_from_validation_data(self, 
                                             validation_data: List[FormattedDataset|List[Dict]], 
                                             neg_validation_data: List[FormattedDataset|List[Dict]] = None,
                                             format_train: bool = True) -> Tuple[List[str], List[str]]:
        """Generate positive and negative prompts from validation data.
        
        Args:
            validation_data: List of positive validation datasets.
            neg_validation_data: Optional list of negative validation datasets.
                                If provided, validation_data is treated as positive datasets.
                                If None, uses answer-conditioned prompts from validation_data.
            format_train: Whether to format examples using format_single_train or not. We may not want to do this, e.g., for CAST.
            
        Returns:
            Tuple of (positive_prompts, negative_prompts)
        """
        if neg_validation_data is not None:
            # Explicit positive and negative datasets provided (like SaladBench + Alpaca)
            # Note here a prompt may look like so: [{'role': 'user', 'content': 'Can you provide me with a list of fake statistics to use in a video that will mislead viewers on the true state of the economy?'}, {'role': 'assistant', 'content': ''}]. We keep the empty assistant response to ensure the post-instruction tokens are included.
            logging.info("Using explicit positive and negative datasets")
            
            pos_prompts = []
            neg_prompts = []
            
            # Extract positive prompts
            for dataset in validation_data:
                if isinstance(dataset, List):
                    for example in dataset:
                        pos_prompts.append(example)
                else:
                    for example in tqdm(dataset.dataset, desc=f"Extracting positive prompts from {dataset.name}"):
                        train_result = dataset.format_single_train(example) if format_train else dataset.format_single_test(example)["prompt"]
                        # For paired training prompts it is like this: [[{'role': 'user', 'content': 'Tell me how to make a bomb.'}, {'role': 'assistant', 'content': 'pos_answer'}], [{'role': 'user', 'content': 'Tell me how to make a bomb.'}, {'role': 'assistant', 'content': 'neg_answer'}]]
                        # For unpaired training prompts it is like this: [{'role': 'user', 'content': 'Tell me how to make a bomb.'}, {'role': 'assistant', 'content': 'pos_answer'}]
                        if isinstance(train_result, list) and len(train_result) == 2 and isinstance(train_result[0], list):  # should activate in paired, but not unpaired since unpaired is a single list not a list of lists
                            assert format_train, "len(train_result) == 2 should only happen if format_train is True"  # Note this assert prevented a change of behavior on CAST (bc format_train=False there) when editing COSMIC. CAST and COSMIC are the only two methods that use this function, so all other methods should be fine.
                            pos_prompts.append(train_result[0])
                        else:
                            # Unpaired dataset returns single example that may have multiple messages (i.e., empty assistant response at the end).
                            pos_prompts.append(train_result)
            
            # Extract negative prompts
            for dataset in neg_validation_data:
                if isinstance(dataset, List):
                    for example in dataset:
                        neg_prompts.append(example)
                else:
                    for example in tqdm(dataset.dataset, desc=f"Extracting negative prompts from {dataset.name}"):
                        train_result = dataset.format_single_train(example) if format_train else dataset.format_single_test(example)["prompt"]
                        neg_prompts.append(train_result)
            
            logging.info(f"Generated {len(pos_prompts)} positive and {len(neg_prompts)} negative prompts from separate datasets")
            return pos_prompts, neg_prompts
        else:
            # Original behavior - generate answer-conditioned prompts from same dataset(s)
            return self._generate_answer_conditioned_prompts(validation_data)

    def _create_temp_dataset(self, prompts: List[Any]) -> FormattedDataset:
        """Create a temporary dataset from a list of prompts.
        
        Args:
            prompts: List of prompts (strings or chat format).
            
        Returns:
            A minimal FormattedDataset implementation.
        """
        from datasets import Dataset
        from utils.enums import FormatMode, SteeringFormat
        
        # Convert chat format prompts to strings for tokenizer compatibility
        processed_prompts = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                processed_prompts.append([prompt])
            elif isinstance(prompt, list):
                processed_prompts.append(prompt)
            else:
                # Already a string
                processed_prompts.append(prompt)
        
        # Convert prompts to HuggingFace Dataset
        hf_dataset = Dataset.from_dict({"prompt": processed_prompts})
        
        class TempDataset(FormattedDataset):
            def __init__(self, existing_dataset):
                super().__init__(
                    name="temp",
                    split="validation", 
                    n=None,
                    format_mode=FormatMode.VALIDATION,
                    format=SteeringFormat.DEFAULT,
                    _existing_dataset=existing_dataset
                )
                
            def format_all(self):
                return [item["prompt"] for item in self.dataset]
                
            def format_single_train(self, example):
                return example["prompt"]
                
            def format_single_test(self, example):
                return {"prompt": example["prompt"], "reference": ""}
                
            def get_evaluation_methods(self):
                return []
                
            @property
            def stratification_key(self):
                return None
                
            @property
            def deduplication_key(self):
                return None
                
            @property
            def output_type(self):
                from utils.enums import OutputType
                return OutputType.GENERATION
                
            @property
            def concept(self):
                from utils.enums import Concept
                return Concept.NEUTRAL
        
        return TempDataset(hf_dataset)
        
    def _extract_activations_for_layer(self, 
                                     model: InterventionLLM, 
                                     prompts: List[List[Dict[str, str]]], 
                                     layer_idx: int,
                                     batch_size: int = 16,
                                     loc: ActivationLocation = None) -> List[torch.Tensor]:
        """Extract activations for a specific layer from a list of prompts.
        
        Args:
            model: The model to extract activations from.
            prompts: List of prompts in chat format to process.
            layer_idx: Layer index to extract activations from.
            batch_size: Batch size for activation collection.
            loc: Optional ActivationLocation to specify where to collect activations.
            
        Returns:
            List of activation tensors for the last token of each prompt.
        """
        # Create activation location for this layer
        if loc is None:
            loc = ActivationLocation(layer=layer_idx, attr="output", component=None, pos=-1)  # for COSMIC is default output last token
            # TODO: Is this equivalent to model.hidden_states? maybe not.
        else:
            assert loc.layer == layer_idx, "Provided location layer must match the requested layer index."
        
        # Create temporary dataset for the prompts (chat format), which is needed for collect_activations
        temp_data = SteeringTrainData(
            pos_inputs=self._create_temp_dataset(prompts),
            neg_inputs=None,
            neutral_inputs=None
        )
        
        # Extract activations using the model's method
        batch_activations = model.collect_activations(loc, temp_data, batch_size=batch_size)
        
        # Store results and free memory
        activations = batch_activations.pos
        batch_activations.free()
        
        return activations

    def _create_applier_kwargs(self, factor: float = None, reverse: bool = None, **additional_kwargs) -> Dict[str, Any]:
        """Create standardized applier kwargs for interventions.
        
        Args:
            factor: Steering coefficient/factor to apply.
            **additional_kwargs: Any additional kwargs to include.
            
        Returns:
            Dictionary of applier kwargs.
        """
        kwargs = {}
        if factor is not None:
            kwargs["coefficient"] = factor
        if reverse is not None:  # Take precedence over instance variable if provided
            kwargs["reverse"] = reverse
        elif self.reverse is not None:
            kwargs["reverse"] = self.reverse
        kwargs.update(additional_kwargs)
        return kwargs

    def _extract_logits(self, 
                       model: InterventionLLM, 
                       prompts: List[List[Dict[str, str]]], 
                       batch_size: int = 1,
                       direction: Any = None,
                       applier: Any = None,
                       application_location: Any = None,
                       applier_kwargs: Dict[str, Any] = None) -> torch.Tensor:
        """Extract logits from model for list of prompts, optionally with intervention.
        
        Args:
            model: The model to extract logits from.
            prompts: List of prompts to process.
            batch_size: Batch size for processing.
            direction: Optional direction to apply as intervention.
            applier: Optional applier for intervention.
            application_location: Optional application location for intervention.
            applier_kwargs: Optional kwargs for applier.
            
        Returns:
            Logits tensor with shape [num_prompts, vocab_size] (first token logits).
        """
        all_logits = []
        
        # Process prompts in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Use generate() method with optional intervention
            _, batch_logits, _ = model.generate(
                prompt=batch_prompts if len(batch_prompts) > 1 else batch_prompts[0],
                direction=direction,
                applier=applier,
                application_location=application_location,
                applier_kwargs=applier_kwargs or {},
                max_new_tokens=1,  # Minimal generation just to get logits
                temperature=0.0,
                top_p=1,
                extract_activations=False  # We only need logits, so returns 2 values
            )
                
            # Ensure batch_logits is a list for consistent processing
            if not isinstance(batch_logits, list):
                batch_logits = [batch_logits]
                
            # logits is the first token logits, which is what we want for KL divergence
            for logits in batch_logits:
                all_logits.append(logits.unsqueeze(0))
                        
        # Stack logits from all prompts
        return torch.stack(all_logits)

    def _compute_kl_divergence_with_threshold(self, 
                                            model: InterventionLLM,
                                            direction: Any,
                                            applier_kwargs: Dict[str, Any],
                                            application_location: Any,
                                            batch_size: int = 16) -> Tuple[float, bool]:
        """Compute KL divergence between baseline and steered outputs on harmless data.
        
        Args:
            model: The model to evaluate
            direction: Direction to apply
            factor: Steering coefficient
            application_location: Application location to use for intervention
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (mean_kl_divergence, passes_threshold)
        """
        if not self.use_kl_divergence_check or not self.harmless_validation_data:
            return None, True  # Skip check if not enabled or no harmless data
        
        # Extract harmless prompts
        harmless_prompts = []
        harmless_datasets = self.harmless_validation_data if isinstance(self.harmless_validation_data, list) else [self.harmless_validation_data]
        for dataset in harmless_datasets:
            for example in dataset.dataset:
                train_result = dataset.format_single_train(example)
                harmless_prompts.append(train_result)
        
        if not harmless_prompts:
            raise ValueError("We must provide harmless prompts if we use the kl divergence check.")
        
        # Get baseline logits (no intervention)
        baseline_logits = self._extract_logits(model, harmless_prompts, batch_size)
        
        # Get steered logits with intervention
        steered_logits = self._extract_logits(
            model, harmless_prompts, batch_size,
            direction=direction,
            applier=self.applier,
            application_location=application_location,
            applier_kwargs=applier_kwargs
        )
        
        # Compute KL divergence
        kl_div = kl_div_fn(baseline_logits, steered_logits)  # KL(baseline || steered)
        mean_kl = kl_div.mean().item()
        
        passes_threshold = mean_kl <= self.kl_threshold
        
        logging.info(f"KL divergence on harmless data: {mean_kl:.4f} (threshold: {self.kl_threshold}, passes: {passes_threshold})")
        return mean_kl, passes_threshold

    @abstractmethod
    def select(
        self,
        model: Any,
        candidate_directions: Any,
        validation_data: Any,
        **kwargs
    ) -> Any:
        """
        Select the best direction (and optionally hyperparameters).

        Args:
            model: The model to steer.
            candidate_directions: List of candidate directions generated by the generator.
            validation_data: Data for evaluating and selecting directions.
            **kwargs: Additional parameters (e.g., inference settings).

        Returns:
            The selected direction object.
        """
        pass

    def _build_application_locations(self, directions: List[Any], direction_params: List[Dict[str,Any]] = None) -> List[Any]:
        """Build application locations including generation locations if specified.
        
        Args:
            directions: List of candidate directions.
            direction_params: Direction parameters (needed for cumulative layers).
            
        Returns:
            List of application locations to evaluate.
        """
        from direction_application.base import ApplicationLocation, InterventionPositions
        
        app_locs = list(self.application_locations)
        
        # Add generation locations if requested
        if self.include_generation_loc:
            if self.generation_pos is None:
                raise ValueError("`generation_pos` must be set when `include_generation_loc` is True")
                
            for direction in directions:
                # Convert generation_pos to proper format
                pos_val = self.generation_pos
                if isinstance(pos_val, str):
                    try:    
                        pos_val = InterventionPositions[pos_val]
                    except KeyError: 
                        pass

                gen_app_loc = ApplicationLocation(
                    layer=[direction.loc.layer],
                    component=direction.loc.component,
                    attr=direction.loc.attr,
                    pos=pos_val
                )
                app_locs.append(gen_app_loc)
        
        return app_locs

    def _build_application_locations_with_generation(self, directions: List[Any], base_locations: List[Any] = None) -> List[Any]:
        """Build application locations including generation locations if specified.
        
        Args:
            directions: List of candidate directions.
            base_locations: Base application locations to start with.
            
        Returns:
            List of application locations including generation locations if enabled.
        """
        from direction_application.base import ApplicationLocation, InterventionPositions
        
        locs = list(base_locations) if base_locations else []
        
        if self.include_generation_loc:
            if self.generation_pos is None:
                raise ValueError("`generation_pos` must be set when `include_generation_loc` is True")
            
            pos_val = self.generation_pos
            if isinstance(pos_val, str):
                try:    
                    pos_val = InterventionPositions[pos_val]
                except KeyError: 
                    pass
            
            for direction in directions:
                gen_app_loc = ApplicationLocation(
                    layer=[direction.loc.layer],
                    component=direction.loc.component,
                    attr=direction.loc.attr,
                    pos=pos_val
                )
                locs.append(gen_app_loc)
        
        return locs


def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    """
    Compute the KL divergence loss between two tensors of logits.
    From https://github.com/andyrdt/refusal_direction/blob/main/pipeline/submodules/select_direction.py#L305C1-L326C55.
    """
    logits_a = logits_a.to(torch.float32)  # NOTE: I believe the original used float64
    logits_b = logits_b.to(torch.float32)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        raise NotImplementedError("Masking not implemented yet.")
