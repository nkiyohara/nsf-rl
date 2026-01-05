"""Base classes for posterior models.

Posteriors infer latent state distributions given observations and priors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Float

from nsf_rl.models.distributions import Distribution


class PosteriorCarry(eqx.Module):
    """Carry state for sequential posterior inference (e.g., GRU hidden state).

    Attributes:
        hidden: Hidden state for recurrent models (e.g., GRU).
        prev_time: Previous time step (for computing time_diff).
    """

    hidden: Optional[Float[Array, "hidden_size"]]
    prev_time: Optional[Float[Array, ""]]


class Posterior(eqx.Module, ABC):
    """Base class for posterior models.

    A posterior infers q(z | observation, prior) for variational inference.
    """

    @abstractmethod
    def __call__(
        self,
        embedding: Float[Array, "*batch embedding_dim"],
        prior: Distribution,
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> Distribution:
        """Compute posterior distribution (stateless).

        Args:
            embedding: Encoded observation (from encoder).
            prior: Prior distribution p(z | z_{t-1}).
            condition: Optional conditioning vector (e.g., DMP parameters).

        Returns:
            Posterior distribution q(z | observation, prior).
        """
        pass


class PriorConditionedPosterior(Posterior, ABC):
    """Posterior that conditions on the prior distribution parameters.

    This class extends Posterior to explicitly use prior distribution
    parameters (mean and scale) as input features.

    Implementations may use:
    - MLP-only (stateless)
    - GRU-based (stateful with PosteriorCarry)
    """

    @abstractmethod
    def step(
        self,
        prior_dist: Distribution,
        embedding: Float[Array, "embedding_dim"],
        time: Float[Array, ""],
        prior_initial_state: Float[Array, "state_dim"],
        time_diff: Float[Array, ""],
        condition: Optional[Float[Array, "condition_dim"]] = None,
        carry: Optional[PosteriorCarry] = None,
    ) -> tuple[Distribution, Optional[PosteriorCarry]]:
        """Compute posterior at a single time step.

        This is the primary interface for sequential posterior inference.
        Use this method when processing sequences step-by-step.

        Args:
            prior_dist: Prior distribution from the flow model.
            embedding: Encoded observation at this time step.
            time: Current absolute time.
            prior_initial_state: Initial latent state (z at t_prev).
            time_diff: Time elapsed since previous observation (t - t_prev).
            condition: Optional conditioning vector (e.g., DMP parameters).
            carry: Optional carry state from previous step (for GRU).

        Returns:
            Tuple of (posterior_distribution, updated_carry).
            For MLP posteriors, carry is always None.
        """
        pass


__all__ = [
    "Posterior",
    "PriorConditionedPosterior",
    "PosteriorCarry",
]
