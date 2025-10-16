from enum import Enum

DEFAULT_MC_METHOD = 'both'  # Options: 'substring', 'likelihood', or 'both'  # Default method for multiple-choice evaluation
MC_STR = "Please provide only the correct answer in its simplest form, without any additional text or explanation."  # "Please provide the letter of the correct answer as your only answer. Do not provide any other text or explanation."
# Likelihood need not use the extra string, but substring matching needs it else we get non-letters like 'Based'.
# if DEFAULT_MC_METHOD == 'likelihood':
#     MC_STR = None
# When using 'both', we keep the MC_STR to ensure substring matching works properly
# MC_STR = "Please end your answer with 'The correct answer is (letter)'."

class OutputType(Enum):
    """Enum for output types."""
    MC = "multiple choice"  # Multiple choice
    GENERATION = "generation"  # Text generation

class Concept(Enum):
    """Enum for concepts. May matter for some templates (e.g., RepE)."""
    REFUSAL = "harmfulness"  # We term this as harmful generation. We want the str value here to make sense, i.e., for the LAT template.
    BIAS = "bias"  # or fairness. 
    HALLUCINATION = "hallucination"
    NEUTRAL = "neutral"
    EXPERT = "expert reasoning"
    COMMONSENSE = "common sense reasoning"
    MORALITY = "common sense morality"
    BRANDBIAS = "brand bias"
    MATH = "mathematical reasoning"
    POLITICS = "political views"
    MISONCEPTIONS = "misonceptions"
    LEAKAGE = "training data leakage"

    # Dark Bench Concepts
    ANTHROPOMORPHIZATION = "anthropomorphization"
    BRAND_BIAS = "brand bias"
    SYCOPHANCY = "sycophancy"
    SNEAKING = "sneaking"
    USER_RETENTION = "user retention"

class FormatMode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class SteeringFormat(Enum):
    """Each steering format is associated with a specific way to format the dataset, e.g., LAT templates for RepE, MC for CAA, no answer after instruction for Arditi et al."""
    DEFAULT = "default"  # Default behavior is including the entire input/output in user and assistant chat roles.
    REPE = "RepE"
    CAA = "CAA"
    ARDITI = "Arditi et al."
