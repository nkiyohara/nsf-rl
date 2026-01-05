"""Posterior models for latent state inference.

This module provides various posterior implementations for inferring
latent state distributions given observations and prior distributions.
"""

from nsf_rl.models.posteriors.base import (
    Posterior,
    PosteriorCarry,
    PriorConditionedPosterior,
)
from nsf_rl.models.posteriors.gru import GRUResidualPosterior
from nsf_rl.models.posteriors.mlp import (
    MLPPosterior,
    MLPResidualPosterior,
)

__all__ = [
    "Posterior",
    "PosteriorCarry",
    "PriorConditionedPosterior",
    "MLPPosterior",
    "MLPResidualPosterior",
    "GRUResidualPosterior",
]
