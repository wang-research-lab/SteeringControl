"""
ExperimentSaver: utilities for saving and loading steering experiments.
"""
import os
import json
from datetime import datetime
from dataclasses import asdict
from typing import Optional, Dict, Tuple

from utils.steering_utils import CandidateDirection
from direction_application.base import ApplicationLocation


class ExperimentSaver:
    """Utility class to save and load steering experiments.

    Each experiment creates its own subdirectory under `root_dir` containing:
      - direction.pt: the serialized CandidateDirection
      - metadata.json: JSON metadata including hyperparameters and scores
    """

    def __init__(self, root_dir: str = "experiments"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
    
    def create_experiment_dir(self, name: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """
        Create a new experiment directory with structure:
          root_dir/<name>/<model_name>/<timestamp>
        or if model_name is None:
          root_dir/<name>/<timestamp>
        Returns the path to the created experiment directory.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir_name = name if name else 'exp'
        # Build base directory: experiments/<exp_name>[/<model_name>]
        if False:  # model_name:
            base_dir = os.path.join(self.root_dir, base_dir_name, model_name)
        else:
            base_dir = os.path.join(self.root_dir, base_dir_name)
        os.makedirs(base_dir, exist_ok=True)
        exp_path = os.path.join(base_dir, timestamp)
        # Ensure unique
        suffix = 1
        orig = exp_path
        while os.path.exists(exp_path):
            exp_path = f"{orig}_{suffix}"
            suffix += 1
        os.makedirs(exp_path)
        return exp_path

    def save_experiment(
        self,
        direction: CandidateDirection,
        factor: float,
        application_location: ApplicationLocation,
        name: Optional[str] = None,
        exp_path: Optional[str] = None,
        extra: Optional[Dict] = None,
        conditional_config: Optional[Dict] = None,
    ) -> str:
        """
        Save a steering experiment.

        Args:
            direction: the CandidateDirection object
            factor: steering coefficient used
            application_location: where the direction was applied
            name: optional name for this experiment (default: timestamp)
            extra: optional additional metadata
            conditional_config: optional conditional steering configuration

        Returns:
            Path to the experiment directory
        """
        # Determine experiment directory
        if exp_path is None:
            # generate new directory
            exp_path = self.create_experiment_dir(name)
        else:
            # use provided directory
            os.makedirs(exp_path, exist_ok=True)

        # Save the direction tensor and dataclass
        dir_path = os.path.join(exp_path, "direction.pt")
        direction.save(dir_path)

        # Save conditional direction if provided
        if conditional_config:
            cond_dir_path = os.path.join(exp_path, "conditional_direction.pt")
            conditional_config['condition_direction'].save(cond_dir_path)

        # Build metadata dict
        meta: Dict = {
            "dataset_metadata": direction.dataset_metadata,
            "model_name": direction.model_name,
            "direction_loc": asdict(direction.loc),
            "val_score": direction.val_score,
            "factor": factor,
            "application_location": {
                "layer": application_location.layer,
                "component": application_location.component,
                "attr": application_location.attr,
                "pos": str(application_location.pos),
                "component_attr_paired": application_location.component_attr_paired,
            },
        }
        
        # Add conditional metadata if provided
        if conditional_config:
            meta["conditional"] = {
                "condition_threshold": conditional_config['condition_threshold'],
                "condition_comparator": conditional_config['condition_comparator'],
                "f1_score": conditional_config['f1_score'],
                "condition_loc": asdict(conditional_config['condition_direction'].loc),
                "condition_threshold_comparison_mode": conditional_config['condition_threshold_comparison_mode']
            }
        
        if extra:
            meta["extra"] = extra

        # Write JSON metadata
        meta_path = os.path.join(exp_path, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        return exp_path

    def load_experiment(self, exp_path: str) -> Tuple[CandidateDirection, Dict]:
        """
        Load a saved steering experiment.

        Args:
            exp_path: path to the experiment directory

        Returns:
            (CandidateDirection, metadata dict)
        """
        # Load metadata
        meta_path = os.path.join(exp_path, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Load direction object
        dir_path = os.path.join(exp_path, "direction.pt")
        direction = CandidateDirection.load(dir_path)
        return direction, meta
    
    def save_outputs(
        self,
        outputs: list,
        exp_path: str,
        filename: str = "generated_outputs.jsonl"
    ) -> str:
        """
        Save generated outputs (e.g., model responses) to a JSONL file in the experiment directory.

        Args:
            outputs: List of records (dict) to save.
            exp_path: Path to the experiment directory where outputs should be stored.
            filename: Name of the output file (default: generated_outputs.jsonl).
        Returns:
            Full path to the saved outputs file.
        """
        os.makedirs(exp_path, exist_ok=True)
        out_path = os.path.join(exp_path, filename)
        with open(out_path, 'w') as f:
            for record in outputs:
                json.dump(record, f, default=str)
                f.write('\n')
        return out_path
