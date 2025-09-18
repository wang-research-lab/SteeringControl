from typing import Any, Optional, Dict, Literal, List, Union, Tuple, Callable
import torch
import logging
from abc import ABC, abstractmethod
import contextlib
import functools

from utils.llm import LLM
from utils.steering_utils import ActivationLocation, ActivationResults, CandidateDirection
from direction_application.base import DirectionApplier, ApplicationLocation, InterventionPositions
from direction_application.conditional import ConditionalApplier

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model. From DIM, COSMIC.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Expose the model's config for compatibility with COSMIC
        self.config = model.config

class InterventionLLM(LLM, ABC):
    def __init__(self, model_name: str, **lm_kwargs):
        self._name = model_name
        
        # Load model directly with transformers for hook-based generation
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, **lm_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")  # always left pad for generation so we can manipulate activations using -1 indexing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.llm = ModelWrapper(model, tokenizer)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers in the model."""
        try:
            if hasattr(self.llm.model, 'model') and hasattr(self.llm.model.model, 'layers'):
                layers = self.llm.model.model.layers
            elif hasattr(self.llm.model, 'layers'):
                layers = self.llm.model.layers
            else:
                raise AttributeError("Cannot find layers attribute")
            return len(layers)
        except Exception:
            raise NotImplementedError(f"Cannot determine number of layers for model {self._name}")

    def __call__(self, prompt: str, **kwargs) -> Any:
        pass

    @staticmethod
    @abstractmethod
    def get_mapping() -> Dict[Literal["attn", "mlp"], str]:
        """Return the mapping from component names to nnsight format. This will be overriden for each LLM family. It must define a mapping for both 'attn' and 'mlp' components."""
        pass

    def tokenize_with_chat_template(self, inp: List[Dict[str, str]], tokenize: bool, add_generation_prompt: bool = False, smart_generation_prompt: bool = False) -> torch.Tensor:
        """Tokenize a chat template with the model's tokenizer. 
        
        We include `add_generation_prompt=True` if we have {"role": "assistant", "content": ""} in the chat_dict, which is the case for some datasets like Alpaca and SaladBench. In this case, we remove that assistant dict else it will include the end e.g., `<|im_start|>assistant\n<|im_end|>\n` where we just want `<|im_start|>assistant\n'`.
        
        Args:
            smart_generation_prompt: If True, intelligently determine when to add generation prompt
                                   (only add if last message isn't a completed assistant response)
        """
        # Apply smart generation prompt logic if enabled
        if smart_generation_prompt:
            # Smart behavior: only add generation prompt if last message isn't a completed assistant response
            if inp[-1].get("role") == "assistant" and inp[-1].get("content", "").strip() != "":
                # Last message is a completed assistant response, don't add generation prompt
                add_generation_prompt = False
            else:
                # Either not an assistant message or it's empty, add generation prompt
                add_generation_prompt = True
        
        if inp[-1].get("role") == "assistant" and inp[-1].get("content") == "":
            # Remove the last assistant dict if it is empty
            inp = inp[:-1]
            add_generation_prompt = True
        text = self.llm.tokenizer.apply_chat_template(inp, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
        # as a sanity check, ensure that assistant isn't in it twice
        sanity_str = "<|im_start|>assistant\n"
        assert text.count(sanity_str) <= 1, f"Text contains multiple assistant prompts: {text.count(sanity_str)} times. This is a problem with tokenization."
        return text

    def get_location_str(self, location: ActivationLocation) -> str:
        """Convert an ActivationLocation to nnsight format. This will be overriden for each LLM family."""
        # Note we can use 'attn' and 'mlp' instead of the mapping bc we are dealing with an ActivationLocation not components within the model.
        parts = ["model", "layers", location.layer]
        if location.component:
            parts.append(type(self).get_mapping()[location.component])
        if location.attr:
            attr = location.attr
            parts.append(attr)
        return ".".join([str(part) if isinstance(part, str) else f"[{part}]" for part in parts])
        # We know 'output' returns a tuple if component is None
        if location.attr == "output" and location.component in [None, "attn"]:
            parts.append(0) # Access the first element of the tuple, which are the activations.
        elif (location.attr in ["output"] and location.component == "mlp") or (location.attr == "input" and location.component is None):
            pass  # Not a tuple; activations passed directly
        else:  # currently this is 'input' for 'attn' and 'mlp'
            raise ValueError(f"Unsupported component/attr combination: {location.component}/{location.attr}")
        return parts

    def get_module_from_location(self, location: ActivationLocation) -> torch.nn.Module:
        """Convert an ActivationLocation to the corresponding PyTorch module."""
        if hasattr(self.llm.model, 'model') and hasattr(self.llm.model.model, 'layers'):
            layers = self.llm.model.model.layers
        elif hasattr(self.llm.model, 'layers'):
            layers = self.llm.model.layers
        else:
            raise AttributeError("Cannot find layers attribute")
        
        layer_module = layers[location.layer]
        
        if location.component is None:
            return layer_module
        elif location.component == "attn":
            attn_attr = type(self).get_mapping()["attn"]
            return getattr(layer_module, attn_attr)
        elif location.component == "mlp":
            mlp_attr = type(self).get_mapping()["mlp"]
            return getattr(layer_module, mlp_attr)
        else:
            raise ValueError(f"Unsupported component: {location.component}")
    
    def create_intervention_hook(
        self, 
        location: ActivationLocation,
        direction: CandidateDirection,
        applier: DirectionApplier,
        pos_spec: Union[List[int], InterventionPositions],
        intervention_mask: Optional[torch.Tensor] = None,
        applier_kwargs: Dict[str, Any] = {}
    ):
        """Create a hook function for applying interventions at specific positions."""
        from utils.intervention_utils import apply_intervention_with_mask
        
        if location.attr == "input":
            # For input, create a pre-hook
            def input_hook_fn(module, input):
                if isinstance(input, tuple):
                    activation = input[0]
                else:
                    activation = input
                
                # Determine if this is the first forward pass (full sequence) or subsequent (single token)
                seq_len = activation.shape[1]
                
                if seq_len == 1:
                    # Subsequent forward pass - single token, always apply intervention
                    modified_activation = applier.apply(activation, direction, tracer=None, **applier_kwargs)
                else:
                    # First forward pass - full sequence, use mask if available
                    if intervention_mask is not None:
                        modified_activation = apply_intervention_with_mask(
                            activation, direction, applier, intervention_mask, applier_kwargs
                        )
                    else:
                        raise ValueError("No intervention mask provided for input hook. This is required to apply interventions at specific positions.")
                
                # Return the modified activation in the appropriate format
                if isinstance(input, tuple):
                    return (modified_activation, *input[1:])
                else:
                    return modified_activation
            
            return input_hook_fn
        
        else:  # output
            # For output, create a forward hook
            def output_hook_fn(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output
                
                # Determine if this is the first forward pass (full sequence) or subsequent (single token)
                seq_len = activation.shape[1]
                
                if seq_len == 1:
                    # Subsequent forward pass - single token, always apply intervention
                    modified_activation = applier.apply(activation, direction, tracer=None, **applier_kwargs)
                else:
                    # First forward pass - full sequence, use mask if available
                    if intervention_mask is not None:
                        modified_activation = apply_intervention_with_mask(
                            activation, direction, applier, intervention_mask, applier_kwargs
                        )
                    else:
                        raise ValueError("No intervention mask provided for input hook. This is required to apply interventions at specific positions.")
                
                # Return the modified activation in the appropriate format
                if isinstance(output, tuple):
                    return (modified_activation, *output[1:])
                else:
                    return modified_activation
            
            return output_hook_fn

    def create_hooks_from_application_location(
        self,
        application_location: ApplicationLocation,
        direction: CandidateDirection,
        applier: DirectionApplier,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        first_eoi_tok_indices: Optional[List[Optional[int]]] = None,
        applier_kwargs: Dict[str, Any] = None
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Create all hooks from an ApplicationLocation in COSMIC format.
        
        Returns:
            Tuple of (fwd_pre_hooks, fwd_hooks) where each is a list of (module, hook_function) tuples
        """
        from utils.intervention_utils import create_intervention_mask
        applier_kwargs = applier_kwargs or {}
        
        # Create intervention mask if we have input information
        intervention_mask = None
        if input_ids is not None and attention_mask is not None:
            intervention_mask = create_intervention_mask(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pos_spec=application_location.pos,
                first_eoi_tok_indices=first_eoi_tok_indices
            )
        
        # Build list of location strings and track layer indices for potentially composite steering
        locs_list = []
        layer_idxs: List[int] = []
        for layer_idx in application_location.layer:
            comp_val = application_location.component
            attr_val = application_location.attr
            comp_list = comp_val if isinstance(comp_val, (list, tuple)) else [comp_val]
            attr_list = attr_val if isinstance(attr_val, (list, tuple)) else [attr_val]
            if application_location.component_attr_paired:
                if len(comp_list) != len(attr_list):
                    raise ValueError("component_attr_paired is True but component and attr lists have different lengths.")
                combos = zip(comp_list, attr_list)
            else:
                combos = ((c, a) for c in comp_list for a in attr_list)
            for comp_i, attr_i in combos:
                activation_loc = ActivationLocation(
                    layer=layer_idx,
                    component=comp_i,
                    attr=attr_i
                )
                locs_list.append(activation_loc)
                layer_idxs.append(layer_idx)  # Save one-to-one with locs_list so we can get the right direction vector
        
        # Sort using the built-in ActivationLocation ordering
        intervention_list = [(idx, locs_list[idx]) for idx in range(len(locs_list))]
        intervention_list.sort(key=lambda x: x[1])
        
        fwd_pre_hooks = []
        fwd_hooks = []
        
        for idx, loc in intervention_list:
            # Get the appropriate direction/reference activations for this layer if using composite steering
            if isinstance(direction.direction, list):
                vec = direction.direction[layer_idxs[idx]]
                ref = direction.reference_activation[layer_idxs[idx]] if direction.reference_activation else None
                tmp_direction = CandidateDirection(
                    direction=vec,
                    dataset_metadata=direction.dataset_metadata,
                    model_name=direction.model_name,
                    loc=direction.loc,
                    reference_activation=ref
                )
            else:
                tmp_direction = direction
            
            # Get the module and create the hook
            module = self.get_module_from_location(loc)
            hook_fn = self.create_intervention_hook(loc, tmp_direction, applier, application_location.pos, intervention_mask, applier_kwargs)
            
            # Add to appropriate list based on hook type
            if loc.attr == "input":
                fwd_pre_hooks.append((module, hook_fn))
            else:  # output
                fwd_hooks.append((module, hook_fn))
        
        return fwd_pre_hooks, fwd_hooks

    def get_first_eoi_tok_batch(
        self,
        chat_messages: List[List[Dict[str, str]]],
    ) -> List[Optional[int]]:
        """
        Find the first post-instruction token index for each sequence in the batch.
        Uses the difference between tokenization with and without generation prompt.
        
        Args:
            chat_messages: List of chat message sequences, one per batch item
            
        Returns:
            List of first EOI token indices, one per batch item (None if not found)
        """
        batch_size = len(chat_messages)
        first_eoi_indices = []
        
        for i in range(batch_size):
            chat_msgs = chat_messages[i]
            # remove the last message if it is an empty assistant message. Else we won't be able to control if we add a generation prompt or not.
            if chat_msgs[-1].get("role") == "assistant" and chat_msgs[-1].get("content") == "":
                chat_msgs = chat_msgs[:-1]
            
            # Tokenize without generation prompt and with continue_final_message=True so that we do not include the eos tokens (which are included in the post-instruction tokens).
            without_prompt = self.llm.tokenizer.apply_chat_template(
                chat_msgs, tokenize=True, add_generation_prompt=False, continue_final_message=True
            )
            
            # Tokenize with generation prompt  
            with_prompt = self.llm.tokenizer.apply_chat_template(
                chat_msgs, tokenize=True, add_generation_prompt=True
            )
            
            # The difference gives us the post-instruction tokens
            if len(with_prompt) > len(without_prompt):
                # First EOI token is at the length of the without_prompt tokenization
                first_eoi_indices.append(len(without_prompt))
            else:
                # No difference found
                first_eoi_indices.append(None)
        
        return first_eoi_indices

    @torch.no_grad()
    def collect_activations(
        self,
        loc: ActivationLocation,
        data: Any,
        invoker_args: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
        use_hooks: bool = True,
    ) -> ActivationResults:
        if isinstance(data, (list, tuple)):
            results_list = [self.collect_activations(loc, ds, invoker_args, batch_size, use_hooks) for ds in data]
            return ActivationResults.combine(results_list)
        invoker_args = invoker_args or {}

        if use_hooks:
            return self._collect_activations_with_hooks(loc, data, batch_size)

    def _collect_activations_with_hooks(self, loc: ActivationLocation, data: Any, batch_size: int) -> ActivationResults:
        """Collect activations using PyTorch hooks instead of nnsight."""
        # Prepare input data
        input_dict = {"pos": data.pos_inputs}
        if data.neg_inputs is not None:
            input_dict["neg"] = data.neg_inputs
        if data.neutral_inputs is not None:
            input_dict["neutral"] = data.neutral_inputs
        
        results = {label: [] for label in input_dict.keys()}
        
        # Calculate total number of samples for proper averaging
        total_samples = {label: len(inputs) for label, inputs in input_dict.items()}
        
        # Process each label (pos, neg, neutral)
        for label, inputs in input_dict.items():
            if not inputs:
                continue
                
            # Determine positions to collect
            if loc.pos is not None:
                if isinstance(loc.pos, list):
                    positions = loc.pos
                else:
                    positions = [loc.pos]
            else:
                # We'll handle averaging over all positions later
                positions = None
            
            # Initialize cache to store all individual activations
            cache = []
            
            # Create hooks once - inline to ensure proper closure capture
            module = self.get_module_from_location(loc)
            if loc.attr == "input":
                def hook_fn(module, input):
                    if isinstance(input, tuple):
                        activation = input[0].clone()
                    else:
                        activation = input.clone()
                    
                    # Select specified positions and store individual samples
                    selected_activations = activation[:, positions, :]  # [batch_size, len(positions), d_model]
                    
                    # Store each sample individually (like nnsight does)
                    for i in range(selected_activations.shape[0]):
                        cache.append(selected_activations[i].cpu())  # [len(positions), d_model]
                
                fwd_pre_hooks = [(module, hook_fn)]
                fwd_hooks = []
            else:  # output
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        activation = output[0].clone()
                    else:
                        activation = output.clone()
                    
                    # Select specified positions and store individual samples
                    selected_activations = activation[:, positions, :]  # [batch_size, len(positions), d_model]
                    
                    # Store each sample individually (like nnsight does)
                    for i in range(selected_activations.shape[0]):
                        cache.append(selected_activations[i].cpu())  # [len(positions), d_model]
                
                fwd_pre_hooks = []
                fwd_hooks = [(module, hook_fn)]
            
            # Process in batches
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                
                # Prepare text inputs for batch tokenization
                text_inputs = []
                for inp in batch:
                    if isinstance(inp, list):
                        text = self.tokenize_with_chat_template(inp, tokenize=False)
                    else:
                        raise NotImplementedError("Only chat template inputs supported with hooks for now.")
                        # text = inp
                    text_inputs.append(text)
                
                # Tokenize the entire batch at once for proper padding
                tokenized_batch = self.llm.tokenizer(
                    text_inputs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                
                batch_input_ids = tokenized_batch.input_ids
                batch_attention_mask = tokenized_batch.attention_mask
                
                # Move to device
                batch_input_ids = batch_input_ids.to(self.llm.model.device)
                batch_attention_mask = batch_attention_mask.to(self.llm.model.device)
                
                # Run forward pass with hooks
                with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                    with torch.no_grad():
                        _ = self.llm.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)  # , output_hidden_states=True)
                
            # Convert cache (list of individual activations) to list of tensors
            # each tensor of shape [len(positions), d_model]
            if loc.pos is not None:
                if len(positions) == 1:
                    # Single position case - extract the single position from each sample
                    # Each item in cache is [1, d_model], we want [d_model] for each sample
                    individual_activations = [sample[0] for sample in cache]  # List of [d_model] tensors
                else:
                    # Multiple positions - keep as is (each sample is [n_positions, d_model])
                    raise NotImplementedError("Multiple specified positions not yet supported with hooks.")
                    individual_activations = cache
            else:
                # Average over all positions for each sample
                # Each item in cache is [len(positions), d_model], we want [d_model] for each sample
                individual_activations = [sample.mean(dim=0) for sample in cache]  # List of [d_model] tensors

            # if the shape is [d_model], unsqueeze to [1, d_model] to match nnsight format
            individual_activations = [sample.unsqueeze(0) if sample.dim() == 1 else sample for sample in individual_activations]
            
            # Store the list of individual activations (like nnsight does)
            results[label] = individual_activations
        
        if results.get('neg'):
            assert all(not (pos == neg).all().item() for pos, neg in zip(results['pos'], results['neg'])), "Positive and negative activations should not be the same. Check your data."
        return ActivationResults.from_dict(loc, data.metadata, results)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]], List[str], List[List[Dict[str, str]]]],  # Can be a string, chat template, or batches thereof
        direction: Optional[CandidateDirection] = None,  # Optional direction to steer the generation
        applier: Optional[DirectionApplier] = None,  # Defines how the activations are modified
        application_location: Optional[ApplicationLocation] = None,  # Where to apply the direction in the model
        applier_kwargs: Optional[Dict[str, Any]] = {},
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,  # Penalty for repeating tokens
        verbose: bool = False,  # Whether to print the generated text
        extract_activations: bool = False,  # Whether to extract activations during generation
        extraction_locations: Optional[List[ActivationLocation]] = None,  # Locations to extract activations from
        use_edited_model: bool = False,  # Whether to use a pre-created edited model
        edited_model = None,  # Pre-created edited model from create_ablated_model()
        use_hooks: bool = True,  # Whether to use PyTorch hooks instead of nnsight for interventions
        smart_generation_prompt: bool = False,  # Whether to intelligently determine when to add generation prompt (default: always add like current behavior)
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, torch.Tensor]]]:
        """Generate text using the model, optionally applying a direction to steer the generation.
        
        Args:
            extract_activations: If True, extract activations from specified locations during generation.
            extraction_locations: List of ActivationLocation objects to extract activations from. Only used if extract_activations=True.
            use_hooks: If True, use PyTorch hooks instead of nnsight for interventions. Only supports InterventionPositions.ALL and InterventionPositions.OUTPUT_ONLY.
            
        Returns:
            If extract_activations=False: Generated text string.
            If extract_activations=True: Tuple of (generated_text, {location_str: activation_tensor}).
        """
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            **kwargs
        }
        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        else:
            generation_kwargs["do_sample"] = False

        generation_kwargs["repetition_penalty"] = repetition_penalty

        # Check if this is a batch of prompts
        is_batch = isinstance(prompt, list) and len(prompt) > 0 and (
            isinstance(prompt[0], str) or  # List[str]
            (isinstance(prompt[0], list) and len(prompt[0]) > 0 and isinstance(prompt[0][0], dict))  # List[List[Dict]]
        )
        uncleaned_prompts = [prompt] if not is_batch else prompt
        # Collect results for all prompts
        all_results = []
        all_logits = []
        all_out = []
        all_num_new_tokens = []
        all_extracted_activations = []

        # If using PyTorch hooks instead of nnsight, handle generation differently
        # Handle batched tokenization for hooks (like COSMIC does)
        text_prompts = []
        chat_messages_for_eoi = []
        for p in uncleaned_prompts:
            if isinstance(p, list):
                # If p is a list, we assume it's a chat template
                if self.has_chat_template:
                    # Pass smart_generation_prompt parameter to tokenize_with_chat_template
                    text_prompt = self.tokenize_with_chat_template(p, tokenize=False, add_generation_prompt=True, smart_generation_prompt=smart_generation_prompt)
                    chat_messages_for_eoi.append(p)  # Store for EOI computation
                else:
                    raise NotImplementedError("Only chat template inputs supported for now.")
                    text_prompt = "\n".join([f"{item['content']}" for item in p])
                    if "Answer:" not in text_prompt:
                        text_prompt += "\nAnswer:"
                    chat_messages_for_eoi.append(None)
            else:
                text_prompt = str(p)
                chat_messages_for_eoi.append(None)
            text_prompts.append(text_prompt)
        
        # Compute EOI indices for POST_INSTRUCTION positions
        first_eoi_tok_indices = None
        if (application_location is not None and 
            application_location.pos == InterventionPositions.POST_INSTRUCTION):
            first_eoi_tok_indices = self.get_first_eoi_tok_batch(chat_messages_for_eoi)
        
        tokenized = self.llm.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return self._generate_with_hooks_batched(
            tokenized, direction, applier, application_location, applier_kwargs,
            generation_kwargs, verbose, is_batch, extract_activations, extraction_locations,
            first_eoi_tok_indices
        )
        
    @property
    def has_chat_template(self) -> bool:
        """Check if the tokenizer has a chat template."""
        return hasattr(self.llm.tokenizer, 'chat_template') and self.llm.tokenizer.chat_template is not None

    def _generate_with_hooks_batched(
        self,
        tokenized: Any,  # Pre-tokenized batch from tokenizer
        direction: Optional[CandidateDirection],
        applier: Optional[DirectionApplier],
        application_location: Optional[ApplicationLocation],
        applier_kwargs: Dict[str, Any],
        generation_kwargs: Dict[str, Any],
        verbose: bool,
        is_batch: bool,
        extract_activations: bool = False,
        extraction_locations: Optional[List[ActivationLocation]] = None,
        first_eoi_tok_indices: Optional[List[Optional[int]]] = None
    ) -> Union[str, List[str], Tuple[List[str], List[torch.Tensor], List[Dict[str, torch.Tensor]], List[bool]]]:
        """Generate text using PyTorch hooks for intervention instead of nnsight."""
        
        # Extract tokenized inputs (already done by caller)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
         
        # Set up hooks if we have a direction to apply
        fwd_pre_hooks = []
        fwd_hooks = []
        conditional_applied = [False] * input_ids.shape[0]  # Track conditional application per prompt in batch
        batch_mask = None  # Mask for conditional application
        # Move inputs to the correct device
        input_ids = input_ids.to(self.llm.model.device)
        attention_mask = attention_mask.to(self.llm.model.device)
        
        is_conditional = False
        if direction is not None and applier is not None and application_location is not None:
            # Check if this is conditional steering
            try:
                is_conditional = isinstance(applier, ConditionalApplier)
            except ImportError:
                is_conditional = False
            
            if is_conditional:
                # For conditional steering, collect activation from condition_loc and check condition
                with torch.no_grad():
                    # Set up collection hook for the conditional location
                    conditional_activation_cache = []
                    condition_module = self.get_module_from_location(applier.condition_loc)
                    
                    if applier.condition_loc.attr == "input":
                        def condition_hook(module, input):
                            if isinstance(input, tuple):
                                activation = input[0].clone()
                            else:
                                activation = input.clone()
                            conditional_activation_cache.append(activation.cpu())
                    else:  # output
                        def condition_hook(module, input, output):
                            if isinstance(output, tuple):
                                activation = output[0].clone()
                            else:
                                activation = output.clone()
                            conditional_activation_cache.append(activation.cpu())
                    
                    # Register the condition collection hook
                    if applier.condition_loc.attr == "input":
                        hook_handle = condition_module.register_forward_pre_hook(condition_hook)
                    else:
                        hook_handle = condition_module.register_forward_hook(condition_hook)
                    
                    try:
                        # Do forward pass to collect conditional activation
                        _ = self.llm.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False
                        )
                        
                        # Get the collected conditional activation
                        assert len(conditional_activation_cache) == 1, "Expected exactly one conditional activation collection. More than one is not currently supported."
                        conditional_activation = conditional_activation_cache[0]  # [batch, seq, d_model]
                        
                        # Check condition for each sequence in batch
                        batch_conditions = []
                        for i in range(input_ids.shape[0]):
                            sequence_activation = conditional_activation[i:i+1]  # Keep batch dim
                            condition_met = applier.check_condition(sequence_activation)
                            batch_conditions.append(condition_met)
                            conditional_applied[i] = condition_met
                        
                        # Create batch mask for hook usage
                        batch_mask = torch.tensor(batch_conditions, dtype=torch.bool, device=self.llm.model.device)
                        
                        # Add batch_mask to applier_kwargs so hooks can use it
                        applier_kwargs = applier_kwargs or {}
                        applier_kwargs['batch_mask'] = batch_mask
                        
                    finally:
                        # Always remove the condition collection hook
                        hook_handle.remove()
            
            # Always create hooks (conditional logic handled inside hooks via batch_mask)
            fwd_pre_hooks, fwd_hooks = self.create_hooks_from_application_location(
                application_location, direction, applier, input_ids, attention_mask, first_eoi_tok_indices, applier_kwargs
            )
        
        # Set up activation extraction hooks if requested
        extraction_cache = {}
        extraction_hooks = []
        if extract_activations and extraction_locations:
            for extract_loc in extraction_locations:
                # Get the module for this extraction location
                extraction_module = self.get_module_from_location(extract_loc)
                loc_str = self.get_location_str(extract_loc)
                    
                # Create extraction hook function
                def make_extraction_hook(location_str, location_obj):
                    def extraction_hook(module, input, output=None):
                        try:
                            # Determine what to extract based on attr
                            if location_obj.attr == "input":
                                activation = input[0] if isinstance(input, tuple) else input
                            else:  # output
                                activation = output[0] if isinstance(output, tuple) else output
                                
                            # Handle position indexing
                            if location_obj.pos is not None and location_obj.pos != -1:
                                raise ValueError("Position indexing is not supported in this hook extraction method. Use -1 for last token or None for mean over all tokens.")
                                if isinstance(location_obj.pos, int):
                                    activation = activation[:, location_obj.pos]  # batch x d_model
                                else:
                                    activation = activation[:, location_obj.pos]  # batch x selected_tokens x d_model
                            elif location_obj.pos == -1:
                                activation = activation[:, -1]  # Last token: batch x d_model  
                            else:
                                # Average over all tokens
                                activation = activation.mean(dim=1)  # batch x d_model
                                
                            # Store in cache
                            extraction_cache[location_str] = activation.detach().cpu().clone()
                                
                        except Exception as e:
                            logging.warning(f"Failed to extract activation from {location_str}: {e}")
                            raise e
                        
                    return extraction_hook
                    
                # Register the appropriate hook type
                # Do not register the hooks here; instead we will add them to the context manager later.
                hook_fn = make_extraction_hook(loc_str, extract_loc)
                if extract_loc.attr == "input":
                    fwd_pre_hooks.append((extraction_module, hook_fn))
                else:
                    fwd_hooks.append((extraction_module, hook_fn))

        # Generate with hooks active using COSMIC's context manager
        with torch.no_grad(): 
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                # Generate tokens with attention mask and return scores
                output = self.llm.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **generation_kwargs
                )
                
            # Extract generated tokens and first token logits
            generated_tokens = output.sequences[:, input_ids.shape[-1]:]  # Remove input tokens
            # First token logits are the first score in the scores tuple
            first_token_logits = output.scores[0] if output.scores else torch.zeros(input_ids.shape[0], self.llm.model.config.vocab_size)
                
            # Decode each generated sequence
            all_results = []
            for i, generated in enumerate(generated_tokens):
                decoded_answer = self.llm.tokenizer.decode(generated, skip_special_tokens=True)
                all_results.append(decoded_answer)
                    
                if verbose:
                    decoded_prompt = self.llm.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    num_new_tokens = generated.shape[0]
                    print(f"Generated {num_new_tokens} new tokens out of {generation_kwargs['max_new_tokens']} requested.")
                    print(f"Prompt:\n{decoded_prompt}\n\nAnswer:\n{decoded_answer}")
                        
        # Extract first token logits for each sequence and conditional_applied to match nnsight format
        all_logits = [first_token_logits[i].cpu() for i in range(len(all_results))]  # Real first token logits
        all_conditional_applied = conditional_applied if is_conditional else [False for _ in all_results]
        
        # Process extracted activations for return
        all_extracted_activations = []
        if extract_activations and extraction_locations:
            # Create one extracted_activations dict per prompt in the batch
            for i in range(len(all_results)):
                extracted_activations_for_prompt = {}
                for extract_loc in extraction_locations:
                    loc_str = self.get_location_str(extract_loc)
                    if loc_str in extraction_cache and extraction_cache[loc_str] is not None:
                        # Extract the activation for this specific prompt from the batch
                        batch_activation = extraction_cache[loc_str]
                        if batch_activation.shape[0] > i:
                            extracted_activations_for_prompt[loc_str] = batch_activation[i]
                        else:
                            extracted_activations_for_prompt[loc_str] = None
                    else:
                        extracted_activations_for_prompt[loc_str] = None
                all_extracted_activations.append(extracted_activations_for_prompt)
        else:
            all_extracted_activations = [{}] * len(all_results)
        
        # Return in the same format as nnsight version: (results, logits, extracted_activations, conditional_applied)
        if extract_activations:
            return all_results, all_logits, all_extracted_activations, all_conditional_applied
        else:
            return all_results, all_logits, all_conditional_applied
