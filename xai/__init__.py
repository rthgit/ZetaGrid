# XAI Tools for SOUL Transformer
# Explainable AI / Mechanistic Interpretability

from .integrated_gradients import IntegratedGradients
from .perturbation import PerturbationAnalysis
from .attention_viz import AttentionVisualizer
from .logit_lens import LogitLens
from .ablation import AblationStudy
from .activation_patching import ActivationPatching

__version__ = "1.0.0"
__all__ = [
    "IntegratedGradients",
    "PerturbationAnalysis", 
    "AttentionVisualizer",
    "LogitLens",
    "AblationStudy",
    "ActivationPatching"
]
