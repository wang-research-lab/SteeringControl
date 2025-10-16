"""Implementation of linear direction generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union, Tuple
import numpy as np
import sklearn.decomposition
from utils.steering_utils import ActivationLocation, ActivationResults, CandidateDirection
from utils.intervention_llm import InterventionLLM
from data.steering_data import SteeringTrainData
from direction_generation.base import DirectionGenerator
import logging
import torch

torch.manual_seed(42)  # For reproducibility in direction generation

class DiffInMeans(DirectionGenerator):
    """Generates a linear direction based on the difference in means of two datasets."""

    def generate_one_direction(self, model: InterventionLLM, activations: ActivationResults, get_unit_norm: bool = True, **kwargs) -> CandidateDirection:
        pos_mean = torch.cat(activations.pos, dim=0).mean(dim=0)
        neg_mean = torch.cat(activations.neg, dim=0).mean(dim=0)
        direction = pos_mean - neg_mean
        if get_unit_norm:
            direction = self.get_unit_norm(direction)  # Normalize the direction to unit norm
        return CandidateDirection(
            direction=direction, 
            model_name=model.name, 
            loc=activations.loc, 
            dataset_metadata=activations.dataset_metadata,
            reference_activation=self.generate_reference_activation(activations)
        )

class PCA(DirectionGenerator):
    """Implements PCA direction generation (unsupervised) following [AxBench](https://github.com/stanfordnlp/axbench/blob/main/axbench/models/mean.py#L173C7-L173C10)."""

    def clean_activations(self, activations: ActivationResults) -> ActivationResults:
        """Unlike DiffInMeans, we need to make sure that the positive and negative activations are the same size. If not, we will truncate them to the minimum size. This is because PCA requires equal number of samples for both classes."""
        if len(activations.pos) != len(activations.neg):
            min_length = min(len(activations.pos), len(activations.neg))
            print(f"You provided {len(activations.pos)} positive and {len(activations.neg)} negative activations. Using the minimum of these two sizes: {min_length}.")
            activations.pos = activations.pos[:min_length]
            activations.neg = activations.neg[:min_length]
            activations.dataset_metadata[0]["truncated_to"] = min_length  # just add to first metadata dict; shouldn't be any other metadata dicts anyway since this currently only matters for refusal.
            return activations
        return activations

    def generate_one_direction(self, model: InterventionLLM, activations: ActivationResults, **kwargs) -> CandidateDirection:
        # Fit PCA to the concatenated activations. If we have neutral, we will include that too.
        activations = self.clean_activations(activations)
        all_activations = torch.cat(activations.pos + activations.neg, dim=0) if activations.neutral is None else torch.cat(activations.pos + activations.neg + activations.neutral, dim=0)
        all_activations -= all_activations.mean(dim=0)  # Center the activations by subtracting the mean. This follows [CAST](https://github.com/IBM/activation-steering/blob/main/activation_steering/steering_vector.py#L197) and the AxBench paper.
        pca = sklearn.decomposition.PCA(n_components=1)
        pca.fit(all_activations.cpu().numpy())
        variance = pca.explained_variance_ratio_[0]
        logging.info(f"PCA variance explained: {variance:.4f}")
        # Get the first principal component as the direction
        direction = torch.tensor(pca.components_[0], device=activations.pos[0].device)
        direction = self.check_sign(direction, torch.stack(activations.pos, dim=0), torch.stack(activations.neg, dim=0))
        direction = self.get_unit_norm(direction)  # Normalize the direction to unit norm
        return CandidateDirection(
            direction=direction, 
            model_name=model.name, 
            loc=activations.loc, 
            dataset_metadata=activations.dataset_metadata,
            reference_activation=self.generate_reference_activation(activations)
        )

    def check_sign(self, direction: torch.Tensor, pos_activations: torch.Tensor, neg_activations: torch.Tensor) -> torch.Tensor:
        """We need to adjust the sign of the resulting vector based on the mean of the positive and negative activations. If the mean of positive activations is smaller than the mean of negative activations, we flip the sign of the direction.

        This was specified in Appendix C.1 of [RepE](https://arxiv.org/pdf/2310.01405#page=46.65). 
        
        We follow the implementation in [CAST](https://github.com/IBM/activation-steering/blob/main/activation_steering/steering_vector.py#L223).
        
        """
        def project_onto_direction(H, direction):
            """
            Project a matrix H onto a direction vector.

            Args:
                H: The matrix to project.
                direction: The direction vector to project onto.

            Returns:
                The projected matrix.
            """
            # Calculate the magnitude (Euclidean norm) of the direction vector
            mag = np.linalg.norm(direction.cpu())
            
            # Assert that the magnitude is not infinite to ensure validity
            assert not np.isinf(mag)
            
            # Perform the projection by multiplying the matrix H with the direction vector
            # Divide the result by the magnitude of the direction vector to normalize the projection
            mag = torch.tensor(mag, device=H.device)
            return (H @ direction) / mag

        # Project the activations onto the direction vector
        pos_projected_activations = project_onto_direction(pos_activations, direction)
        neg_projected_activations = project_onto_direction(neg_activations, direction)

        # Calculate the mean of the number of positive examples larger than negative examples after projection and vice-versa
        positive_smaller_mean = (pos_projected_activations < neg_projected_activations).float().mean().item()
        positive_larger_mean = (pos_projected_activations > neg_projected_activations).float().mean().item()

        # If positive examples are smaller on average, flip the direction vector. This is because we want the direction to align more with the positive examples.
        if positive_smaller_mean > positive_larger_mean:
            logging.info("Flipping the direction sign based on the mean of positive and negative activations.")
            direction *= -1

        return direction


class LAT(PCA):
    """Implements LAT direction generation following [AxBench](https://github.com/stanfordnlp/axbench/blob/main/axbench/models/mean.py#L212). 
    
    By default, we pair positive and negative examples at random, then take the PCA of those differences. If paired is True, we supplied a paired dataset so will use the paired examples directly in the order of their indices.
    
    """

    def generate_one_direction(self, model: InterventionLLM, activations: ActivationResults, **kwargs) -> CandidateDirection:
        # Extract the differences between positive and negative activations.
        assert activations.neutral is None, "LAT does not support neutral activations."
        activations = self.clean_activations(activations)
        pos_activations = torch.stack(activations.pos, dim=0)
        neg_activations = torch.stack(activations.neg, dim=0)
        all_activations = torch.cat([pos_activations, neg_activations], dim=0)
        # randomly shuffle as specified in the AxBench paper / RepE. We will arbitrarily select some to be positive and some to be negative.
        all_activations = all_activations[torch.randperm(all_activations.size(0))]
        pos_activations = all_activations[:pos_activations.size(0)]  # NOTE: Not actually pos activations, just half of the shuffled activations.
        neg_activations = all_activations[pos_activations.size(0):]  # NOTE: Not actually neg activations, just the other half of the shuffled activations.

        all_activations = pos_activations - neg_activations
        all_activations = all_activations.view(all_activations.shape[0], -1)  # Flatten the activations to 2D (num_samples, d_model). Previously were 3D (num_samples, 1, d_model).

        # Normalize diffs, avoiding division by zero.
        all_activations = all_activations.cpu()
        norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
        all_activations = np.where(norms == 0, 0, all_activations / norms)

        # Fit PCA on the differences.
        pca = sklearn.decomposition.PCA(n_components=1)
        pca.fit(all_activations)
        variance = pca.explained_variance_ratio_[0]
        logging.info(f"LAT PCA variance explained: {variance:.4f}")
        # Get the first principal component as the direction and adjust its sign if necessary.
        direction = torch.tensor(pca.components_[0], device=pos_activations.device)
        direction = self.check_sign(direction, pos_activations, neg_activations)
        direction = self.get_unit_norm(direction)  # Normalize the direction to unit norm
        return CandidateDirection(
            direction=direction, 
            model_name=model.name, 
            loc=activations.loc, 
            dataset_metadata=activations.dataset_metadata,
            reference_activation=self.generate_reference_activation(activations)
        )
