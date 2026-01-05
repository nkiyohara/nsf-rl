"""GRU-based posterior models.

These posteriors use GRU cells to maintain temporal context across observations,
combined with MLPs for parameter prediction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nsf_rl.models.distributions import (
    ContinuousNormalizingFlow,
    Distribution,
    MultivariateNormalDiag,
    NormalizingFlow,
)
from nsf_rl.models.posteriors.base import PosteriorCarry, PriorConditionedPosterior


class GRUResidualPosterior(PriorConditionedPosterior):
    """GRU-based posterior with residual connection to the prior.

    Uses a GRU cell to accumulate temporal context from the observation sequence,
    then predicts residual adjustments to the prior mean.

    The GRU receives:
    - Observation embedding
    - Time difference from previous observation (Δt)
    - Optionally, absolute time

    The output MLP receives:
    - GRU hidden state
    - Prior initial state (z at t_prev)
    - Time difference
    - Optionally, absolute time

    Attributes:
        gru: GRU cell for temporal context.
        output_net: MLP for predicting mean and scale.
        autonomous: If True, only uses time_diff; if False, also uses absolute time.
        scale_eps: Epsilon added to scale for numerical stability.
    """

    gru: eqx.nn.GRUCell
    output_net: eqx.nn.MLP
    latent_dim: int
    autonomous: bool
    scale_eps: float = eqx.field(static=True, default=1e-2)

    def __init__(
        self,
        *,
        embedding_dim: int,
        latent_dim: int,
        condition_dim: int = 0,
        autonomous: bool = True,
        gru_hidden_size: int = 64,
        mlp_hidden_size: int = 64,
        mlp_depth: int = 2,
        activation: Callable = jax.nn.tanh,
        key: PRNGKeyArray,
        scale_eps: float = 1e-2,
    ):
        """Initialize the GRU residual posterior.

        Args:
            embedding_dim: Dimension of observation embeddings.
            latent_dim: Dimension of the latent state.
            condition_dim: Dimension of conditioning vector (e.g., DMP params).
            autonomous: If True, only time_diff is used; if False, also uses abs time.
            gru_hidden_size: Size of GRU hidden state.
            mlp_hidden_size: Width of output MLP hidden layers.
            mlp_depth: Depth of output MLP.
            activation: Activation function for MLP.
            key: PRNG key for initialization.
            scale_eps: Epsilon added to scale.
        """
        gru_key, mlp_key = jax.random.split(key)

        # GRU input: embedding + time_diff + (time if not autonomous) + condition
        time_dims = 1 if autonomous else 2
        gru_input_size = embedding_dim + time_dims + condition_dim

        # MLP input: gru_hidden + prior_initial_state + time_diff + (time if not autonomous) + condition
        mlp_input_size = gru_hidden_size + latent_dim + time_dims + condition_dim

        self.gru = eqx.nn.GRUCell(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            key=gru_key,
        )
        self.output_net = eqx.nn.MLP(
            in_size=mlp_input_size,
            out_size=2 * latent_dim,  # mean residual and log_scale
            width_size=mlp_hidden_size,
            depth=mlp_depth,
            activation=activation,
            final_activation=lambda x: x,
            key=mlp_key,
        )
        self.latent_dim = latent_dim
        self.autonomous = autonomous
        self.scale_eps = scale_eps

    def __call__(
        self,
        embedding: Float[Array, "*batch embedding_dim"],
        prior: Distribution,
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> MultivariateNormalDiag:
        """Compute posterior (stateless fallback).

        For proper sequential inference, use step() instead.
        This method resets the GRU hidden state and assumes time_diff=0.
        """
        time_diff = jnp.zeros(())
        time = jnp.zeros(())
        prior_initial_state = prior.loc
        dist, _ = self.step(
            prior_dist=prior,
            embedding=embedding,
            time=time,
            prior_initial_state=prior_initial_state,
            time_diff=time_diff,
            condition=condition,
            carry=None,
        )
        return dist

    def step(
        self,
        prior_dist: Distribution,
        embedding: Float[Array, "embedding_dim"],
        time: Float[Array, ""],
        prior_initial_state: Float[Array, "state_dim"],
        time_diff: Float[Array, ""],
        condition: Optional[Float[Array, "condition_dim"]] = None,
        carry: Optional[PosteriorCarry] = None,
    ) -> tuple[MultivariateNormalDiag, PosteriorCarry]:
        """Compute posterior at a single time step with GRU state update.

        Args:
            prior_dist: Prior distribution from the flow model.
            embedding: Encoded observation at this time step.
            time: Current absolute time.
            prior_initial_state: Initial latent state (z at t_prev).
            time_diff: Time elapsed since previous observation.
            condition: Optional conditioning vector.
            carry: GRU carry state from previous step.

        Returns:
            Tuple of (posterior_distribution, updated_carry).
        """
        # Extract prior parameters
        if isinstance(prior_dist, MultivariateNormalDiag):
            prior_mean = prior_dist.loc
            prior_scale = prior_dist.scale_diag
        elif isinstance(prior_dist, (ContinuousNormalizingFlow, NormalizingFlow)):
            if isinstance(prior_dist.base_distribution, MultivariateNormalDiag):
                prior_mean = prior_dist.base_distribution.loc
                prior_scale = prior_dist.base_distribution.scale_diag
            else:
                raise ValueError(f"Unsupported base distribution: {type(prior_dist.base_distribution)}")
        else:
            raise ValueError(f"Unsupported prior distribution: {type(prior_dist)}")

        # Initialize carry if None
        if carry is None:
            carry = PosteriorCarry(
                hidden=jnp.zeros((self.gru.hidden_size,)),
                prev_time=time + 1.0,  # Ensure first time_diff is computed correctly
            )

        # Compute time_diff from carry if using sequential processing
        # Note: time_diff argument is passed explicitly, so we use that
        actual_time_diff = time_diff

        # Build GRU input
        if self.autonomous:
            time_features = jnp.atleast_1d(actual_time_diff)
        else:
            time_features = jnp.concatenate(
                [jnp.atleast_1d(actual_time_diff), jnp.atleast_1d(time)],
                axis=-1,
            )

        gru_input_parts = [embedding, time_features]
        if condition is not None:
            gru_input_parts.append(condition)
        gru_input = jnp.concatenate(gru_input_parts, axis=-1)

        # Update GRU hidden state
        if carry.hidden is None:
            raise ValueError("Hidden state is None. This should never happen.")
        updated_hidden = self.gru(gru_input, carry.hidden)

        # Build MLP input
        mlp_input_parts = [updated_hidden, prior_initial_state, time_features]
        if condition is not None:
            mlp_input_parts.append(condition)
        mlp_input = jnp.concatenate(mlp_input_parts, axis=-1)

        # Predict posterior parameters
        params = self.output_net(mlp_input)
        mean_residual, log_scale = jnp.split(params, 2, axis=-1)

        # Posterior mean = prior mean + residual
        mean = prior_mean + mean_residual

        # Posterior scale = prior_scale * softplus(log_scale) + eps
        scale = prior_scale * jax.nn.softplus(log_scale) + self.scale_eps

        # Update carry
        new_carry = PosteriorCarry(hidden=updated_hidden, prev_time=time)

        return MultivariateNormalDiag(loc=mean, scale_diag=scale), new_carry


__all__ = [
    "GRUResidualPosterior",
]

