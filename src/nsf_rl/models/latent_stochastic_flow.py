"""Latent Neural Stochastic Flow model.

This module implements the Latent Neural Stochastic Flow (LNSF) model,
which combines:
1. An encoder to map observations to embeddings
2. A posterior to infer latent states from embeddings and priors
3. A stochastic flow to model latent state dynamics
4. A decoder to reconstruct observations from latent states
5. A bridge model for flow consistency losses

The model is trained using a variational ELBO objective with flow consistency losses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nsf_rl.data.observation import ObservationBase
from nsf_rl.models.bridge_models import BridgeModel
from nsf_rl.models.decoders import Decoder
from nsf_rl.models.distributions import (
    ContinuousNormalizingFlow,
    Distribution,
    MultivariateNormalDiag,
)
from nsf_rl.models.encoders import Encoder
from nsf_rl.models.posteriors import Posterior, PosteriorCarry, PriorConditionedPosterior
from nsf_rl.models.stochastic_flows import StochasticFlow


def compute_kl_divergence(
    posterior: Distribution,
    prior: Distribution,
    sample: Float[Array, "..."],
    posterior_log_prob: Optional[Float[Array, "..."]] = None,
) -> Float[Array, "..."]:
    """Compute KL divergence D_KL(posterior || prior).

    Uses analytical KL when both are Gaussians, otherwise uses Monte Carlo estimation.

    For Affine Coupling flows, the prior is a ContinuousNormalizingFlow.
    In this case, we use Monte Carlo estimation:
        KL ≈ log q(z) - log p(z)  where z ~ q

    Args:
        posterior: Posterior distribution q(z).
        prior: Prior distribution p(z).
        sample: Sample from posterior (z ~ q).
        posterior_log_prob: Precomputed log q(z) if available.

    Returns:
        KL divergence estimate.
    """
    # Case 1: Both are Gaussian -> analytical KL
    if isinstance(posterior, MultivariateNormalDiag) and isinstance(prior, MultivariateNormalDiag):
        return posterior.kl_divergence(prior)

    # Case 2: Posterior is Gaussian, prior is Flow with Gaussian base
    # Use KL to base distribution as approximation (valid when bijector is near identity)
    if isinstance(posterior, MultivariateNormalDiag) and isinstance(prior, ContinuousNormalizingFlow):
        if isinstance(prior.base_distribution, MultivariateNormalDiag):
            # This is an approximation that works well when time_diff is small
            return posterior.kl_divergence(prior.base_distribution)

    # Case 3: General case -> Monte Carlo estimation
    # KL(q || p) = E_q[log q(z) - log p(z)] ≈ log q(z) - log p(z) for z ~ q
    if posterior_log_prob is None:
        posterior_log_prob = posterior.log_prob(sample)
    prior_log_prob = prior.log_prob(sample)
    return posterior_log_prob - prior_log_prob


class SampleResult(NamedTuple):
    """Result of sampling from the posterior."""

    samples: Float[Array, "batch T latent_dim"]
    log_probs: Float[Array, "batch T"]
    prior_log_probs: Float[Array, "batch T"]


@dataclass
class LossComponents:
    """Components of the ELBO loss."""

    elbo: Float[Array, ""]
    reconstruction_loss: Float[Array, ""]
    kl_divergence: Float[Array, ""]
    flow_1_to_2_loss: Float[Array, ""]
    flow_2_to_1_loss: Float[Array, ""]

    def total(
        self,
        flow_loss_weight: float = 1.0,
    ) -> Float[Array, ""]:
        """Compute total loss."""
        return (
            -self.elbo
            + flow_loss_weight * (self.flow_1_to_2_loss + self.flow_2_to_1_loss)
        )


class LatentStochasticFlow(eqx.Module):
    """Latent Neural Stochastic Flow model for partially observed dynamics.

    This model learns:
    - A latent state representation from partial observations
    - Stochastic dynamics in the latent space
    - A decoder to reconstruct observations from latent states

    The model is trained using a variational ELBO objective that includes:
    - Reconstruction loss: log p(observation | latent)
    - KL divergence: D_KL(posterior || prior)
    - Flow consistency losses: ensures time-reversibility of the flow

    Attributes:
        encoder: Maps observations to embeddings.
        posterior: Infers latent states from embeddings and priors.
        flow: Models stochastic dynamics in latent space.
        decoder: Maps latent states to observation distributions.
        bridge: Auxiliary model for flow loss computation.
        latent_dim: Dimension of the latent state.
        obs_dim: Dimension of observations.
        condition_dim: Dimension of conditioning vector.
        use_condition_in_encoder: Whether to pass condition to encoder.
        use_condition_in_flow: Whether to pass condition to flow.
    """

    encoder: Encoder
    posterior: Posterior
    flow: StochasticFlow
    decoder: Decoder
    bridge: BridgeModel

    latent_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    condition_dim: int = eqx.field(static=True)
    use_condition_in_encoder: bool = eqx.field(static=True)
    use_condition_in_flow: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        encoder: Encoder,
        posterior: Posterior,
        flow: StochasticFlow,
        decoder: Decoder,
        bridge: BridgeModel,
        latent_dim: int,
        obs_dim: int,
        condition_dim: int = 0,
        use_condition_in_encoder: bool = True,
        use_condition_in_flow: bool = True,
    ):
        """Initialize the Latent Stochastic Flow model.

        Args:
            encoder: Encoder module.
            posterior: Posterior module.
            flow: Stochastic flow module.
            decoder: Decoder module.
            bridge: Bridge model module.
            latent_dim: Dimension of latent states.
            obs_dim: Dimension of observations.
            condition_dim: Dimension of conditioning vector (e.g., DMP params).
            use_condition_in_encoder: Whether to pass condition to encoder.
            use_condition_in_flow: Whether to pass condition to flow.
        """
        self.encoder = encoder
        self.posterior = posterior
        self.flow = flow
        self.decoder = decoder
        self.bridge = bridge
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.condition_dim = condition_dim
        self.use_condition_in_encoder = use_condition_in_encoder
        self.use_condition_in_flow = use_condition_in_flow

    def encode(
        self,
        obs: Union[ObservationBase, Float[Array, "*batch obs_dim"]],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> Float[Array, "*batch embedding_dim"]:
        """Encode observation to embedding.

        Args:
            obs: Observation or observation base object.
            condition: Optional conditioning vector.

        Returns:
            Embedding vector.
        """
        # Get encoder input
        if isinstance(obs, ObservationBase):
            encoder_input = obs.encoder_input
        else:
            encoder_input = obs

        # Determine whether to use condition
        encoder_condition = condition if self.use_condition_in_encoder else None

        return self.encoder(encoder_input, encoder_condition)

    def get_prior(
        self,
        z: Float[Array, "*batch latent_dim"],
        t: Float[Array, "*batch"],
        t_next: Float[Array, "*batch"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> Distribution:
        """Get prior distribution for next latent state.

        Args:
            z: Current latent state.
            t: Current time.
            t_next: Next time.
            condition: Optional conditioning vector.

        Returns:
            Prior distribution p(z_{t_next} | z_t).
        """
        flow_condition = condition if self.use_condition_in_flow else None
        return self.flow(z, t, t_next, flow_condition)

    def infer_posterior_step(
        self,
        embedding: Float[Array, "embedding_dim"],
        prior: Distribution,
        time: Float[Array, ""],
        prior_initial_state: Float[Array, "latent_dim"],
        time_diff: Float[Array, ""],
        condition: Optional[Float[Array, "condition_dim"]] = None,
        carry: Optional[PosteriorCarry] = None,
    ) -> tuple[Distribution, Optional[PosteriorCarry]]:
        """Infer posterior distribution at a single time step.

        Uses the posterior's step() method for proper sequential inference
        with time_diff and optional GRU state.

        Args:
            embedding: Encoded observation.
            prior: Prior distribution from flow.
            time: Current absolute time.
            prior_initial_state: Latent state at previous time (z at t_prev).
            time_diff: Time elapsed since previous observation.
            condition: Conditioning vector (always passed to posterior).
            carry: Optional carry state for GRU posteriors.

        Returns:
            Tuple of (posterior_distribution, updated_carry).
        """
        if isinstance(self.posterior, PriorConditionedPosterior):
            return self.posterior.step(
                prior_dist=prior,
                embedding=embedding,
                time=time,
                prior_initial_state=prior_initial_state,
                time_diff=time_diff,
                condition=condition,
                carry=carry,
            )
        else:
            # Fallback for non-step posteriors
            dist = self.posterior(embedding, prior, condition)
            return dist, None

    def decode(
        self,
        z: Float[Array, "*batch latent_dim"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> Distribution:
        """Decode latent state to observation distribution.

        Args:
            z: Latent state.
            condition: Optional conditioning vector.

        Returns:
            Distribution over observations.
        """
        return self.decoder(z, condition)

    def elbo(
        self,
        observations: list[Union[ObservationBase, Float[Array, "batch obs_dim"]]],
        times: Float[Array, "batch T"],
        z_init: Float[Array, "batch latent_dim"],
        condition: Optional[Float[Array, "batch condition_dim"]] = None,
        key: Optional[PRNGKeyArray] = None,
        n_samples: int = 1,
    ) -> LossComponents:
        """Compute the Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(obs|z)] - D_KL(q(z|obs) || p(z|z_prev))

        This method properly handles:
        - time_diff computation between consecutive observations
        - PosteriorCarry propagation for GRU posteriors
        - Condition routing to encoder, posterior, and flow

        Args:
            observations: List of T observations.
            times: Time array of shape (batch, T) in seconds.
            z_init: Initial latent state.
            condition: Optional conditioning vector.
            key: PRNG key for sampling.
            n_samples: Number of samples for Monte Carlo estimation.

        Returns:
            LossComponents containing ELBO and its components.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        batch_size = z_init.shape[0]
        T = len(observations)

        # Process each batch element independently to handle GRU carry
        def process_single_trajectory(
            obs_sequence: list[Float[Array, "obs_dim"]],
            time_sequence: Float[Array, "T"],
            z_init_single: Float[Array, "latent_dim"],
            condition_single: Optional[Float[Array, "condition_dim"]],
            key_single: PRNGKeyArray,
        ) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
            """Process a single trajectory."""
            total_reconstruction = jnp.zeros(())
            total_kl = jnp.zeros(())
            total_flow_1_to_2 = jnp.zeros(())
            total_flow_2_to_1 = jnp.zeros(())

            z_current = z_init_single
            t_current = time_sequence[0]
            posterior_carry: Optional[PosteriorCarry] = None

            for i in range(T):
                key_single, sample_key, flow_key = jax.random.split(key_single, 3)

                obs = obs_sequence[i]
                t_next = time_sequence[i]

                # Compute time_diff (elapsed time since previous observation)
                if i == 0:
                    time_diff = jnp.array(-1.0)  # Sentinel for first step
                else:
                    time_diff = t_next - t_current

                # Get prior from flow
                if i == 0:
                    # First step: prior is centered at initial state
                    prior = MultivariateNormalDiag(
                        loc=z_current,
                        scale_diag=jnp.ones_like(z_current) * 0.1,
                    )
                else:
                    prior = self.get_prior(z_current, t_current, t_next, condition_single)

                # Encode observation
                embedding = self.encode(obs, condition_single)

                # Get posterior using step() with time_diff
                posterior, posterior_carry = self.infer_posterior_step(
                    embedding=embedding,
                    prior=prior,
                    time=t_next,
                    prior_initial_state=z_current,
                    time_diff=time_diff,
                    condition=condition_single,
                    carry=posterior_carry,
                )

                # Sample from posterior
                z_sample = posterior.sample(sample_key)

                # Reconstruction loss: E_q[log p(obs|z)]
                obs_dist = self.decode(z_sample, condition_single)
                if isinstance(obs, ObservationBase):
                    obs_value = obs.value
                else:
                    obs_value = obs
                reconstruction = obs_dist.log_prob(obs_value)

                # KL divergence: D_KL(q || p)
                # Uses analytical KL for Gaussian-Gaussian, Monte Carlo otherwise
                # For Affine Coupling flows (ContinuousNormalizingFlow prior),
                # we use KL to base distribution as approximation
                kl = compute_kl_divergence(posterior, prior, z_sample)

                total_reconstruction = total_reconstruction + reconstruction
                total_kl = total_kl + kl

                # Flow consistency losses (for t > 0)
                if i > 0:
                    flow_1_to_2, flow_2_to_1 = self._flow_losses_single(
                        z_prev=z_current,
                        z_curr=z_sample,
                        t_prev=t_current,
                        t_curr=t_next,
                        condition=condition_single,
                        key=flow_key,
                    )
                    total_flow_1_to_2 = total_flow_1_to_2 + flow_1_to_2
                    total_flow_2_to_1 = total_flow_2_to_1 + flow_2_to_1

                z_current = z_sample
                t_current = t_next

            return (
                total_reconstruction,
                total_kl,
                total_flow_1_to_2,
                total_flow_2_to_1,
                jnp.array(T, dtype=jnp.float32),
            )

        # Batch processing
        keys = jax.random.split(key, batch_size)

        # Stack observations into arrays for vmap
        obs_stacked = jnp.stack(observations, axis=1)  # [batch, T, obs_dim]

        # vmap over batch dimension
        def process_batch_elem(i: int):
            obs_seq = [obs_stacked[i, t] for t in range(T)]
            cond = condition[i] if condition is not None else None
            return process_single_trajectory(
                obs_seq, times[i], z_init[i], cond, keys[i]
            )

        # Use lax.scan or simple loop for batching
        results = jax.vmap(
            lambda i: process_batch_elem(i)
        )(jnp.arange(batch_size))

        total_reconstruction, total_kl, total_flow_1_to_2, total_flow_2_to_1, _ = results

        # Average over batch and time
        avg_reconstruction = total_reconstruction.mean() / T
        avg_kl = total_kl.mean() / T
        avg_flow_1_to_2 = total_flow_1_to_2.mean() / max(T - 1, 1)
        avg_flow_2_to_1 = total_flow_2_to_1.mean() / max(T - 1, 1)

        elbo = avg_reconstruction - avg_kl

        return LossComponents(
            elbo=elbo,
            reconstruction_loss=-avg_reconstruction,
            kl_divergence=avg_kl,
            flow_1_to_2_loss=avg_flow_1_to_2,
            flow_2_to_1_loss=avg_flow_2_to_1,
        )

    def _flow_losses_single(
        self,
        z_prev: Float[Array, "latent_dim"],
        z_curr: Float[Array, "latent_dim"],
        t_prev: Float[Array, ""],
        t_curr: Float[Array, ""],
        condition: Optional[Float[Array, "condition_dim"]],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Compute bidirectional flow consistency losses for a single sample."""
        # Sample intermediate time
        key, t_key, sample_key = jax.random.split(key, 3)
        alpha = jax.random.uniform(t_key)
        t_mid = t_prev + alpha * (t_curr - t_prev)

        # Bridge distribution: q(z_mid | z_prev, z_curr)
        bridge_condition = condition if self.use_condition_in_flow else None
        bridge_dist = self.bridge(z_prev, t_prev, z_curr, t_curr, t_mid, bridge_condition)

        # Sample from bridge
        z_mid = bridge_dist.sample(sample_key)

        # Flow 1→2: p(z_curr | z_mid) should match forward dynamics
        flow_condition = condition if self.use_condition_in_flow else None
        forward_dist = self.flow(z_mid, t_mid, t_curr, flow_condition)
        log_p_forward = forward_dist.log_prob(z_curr)
        log_q_bridge = bridge_dist.log_prob(z_mid)
        flow_1_to_2 = -(log_p_forward - log_q_bridge)

        # Flow 2→1: p(z_prev | z_mid, backward) should match backward dynamics
        backward_dist = self.flow(z_mid, t_mid, t_prev, flow_condition)
        log_p_backward = backward_dist.log_prob(z_prev)
        flow_2_to_1 = -(log_p_backward - log_q_bridge)

        return flow_1_to_2, flow_2_to_1

    def predict(
        self,
        z_init: Float[Array, "*batch latent_dim"],
        t_init: Float[Array, "*batch"],
        t_query: Float[Array, "*batch T_query"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
        return_distribution: bool = False,
    ) -> Union[Float[Array, "*batch T_query latent_dim"], list[Distribution]]:
        """Predict latent states at query times.

        Uses the flow model to propagate the initial state forward in time.

        Args:
            z_init: Initial latent state.
            t_init: Initial time.
            t_query: Query times (must be sorted, >= t_init).
            condition: Optional conditioning vector.
            return_distribution: If True, return distributions instead of means.

        Returns:
            If return_distribution is False: predicted latent means.
            If return_distribution is True: list of distributions at each query time.
        """
        flow_condition = condition if self.use_condition_in_flow else None

        T_query = t_query.shape[-1]
        predictions = []
        distributions = []

        z_current = z_init
        t_current = t_init

        for i in range(T_query):
            t_next = t_query[..., i]

            # Get flow distribution
            dist = self.flow(z_current, t_current, t_next, flow_condition)

            if return_distribution:
                distributions.append(dist)

            # Use mean for prediction
            z_next = dist.loc
            predictions.append(z_next)

            z_current = z_next
            t_current = t_next

        if return_distribution:
            return distributions

        return jnp.stack(predictions, axis=-2)

    def predict_observations(
        self,
        z_init: Float[Array, "*batch latent_dim"],
        t_init: Float[Array, "*batch"],
        t_query: Float[Array, "*batch T_query"],
        condition: Optional[Float[Array, "*batch condition_dim"]] = None,
    ) -> Float[Array, "*batch T_query obs_dim"]:
        """Predict observations at query times.

        Args:
            z_init: Initial latent state.
            t_init: Initial time.
            t_query: Query times.
            condition: Optional conditioning vector.

        Returns:
            Predicted observation means at each query time.
        """
        # Predict latent states
        z_predictions = self.predict(z_init, t_init, t_query, condition)

        # Decode to observations
        T_query = t_query.shape[-1]
        obs_predictions = []

        for i in range(T_query):
            z_i = z_predictions[..., i, :]
            obs_dist = self.decode(z_i, condition)
            obs_predictions.append(obs_dist.loc)

        return jnp.stack(obs_predictions, axis=-2)


def build_latent_stochastic_flow(
    *,
    latent_dim: int,
    obs_dim: int,
    condition_dim: int = 0,
    encoder_embedding_dim: int = 64,
    encoder_hidden_size: int = 64,
    encoder_depth: int = 2,
    posterior_type: Literal["mlp", "mlp_residual", "gru"] = "gru",
    posterior_hidden_size: int = 64,
    posterior_depth: int = 2,
    gru_hidden_size: int = 64,
    flow_type: Literal["mvn", "affine_coupling"] = "affine_coupling",
    flow_hidden_size: int = 64,
    flow_depth: int = 2,
    flow_num_layers: int = 4,
    bridge_type: Literal["mvn", "affine_coupling"] = "mvn",
    bridge_hidden_size: int = 64,
    bridge_depth: int = 2,
    decoder_hidden_size: int = 64,
    decoder_depth: int = 2,
    use_condition_in_encoder: bool = True,
    use_condition_in_flow: bool = True,
    autonomous: bool = True,
    key: PRNGKeyArray,
) -> LatentStochasticFlow:
    """Build a Latent Stochastic Flow model with default architecture.

    Args:
        latent_dim: Dimension of latent states.
        obs_dim: Dimension of observations.
        condition_dim: Dimension of conditioning vector.
        encoder_embedding_dim: Output dimension of encoder.
        encoder_hidden_size: Hidden size for encoder MLP.
        encoder_depth: Depth of encoder MLP.
        posterior_type: Type of posterior ("mlp", "mlp_residual", "gru").
        posterior_hidden_size: Hidden size for posterior MLP.
        posterior_depth: Depth of posterior MLP.
        gru_hidden_size: Hidden size for GRU (if posterior_type="gru").
        flow_type: Type of flow ("mvn" or "affine_coupling").
        flow_hidden_size: Hidden size for flow networks.
        flow_depth: Depth of flow networks.
        flow_num_layers: Number of affine coupling layers (if applicable).
        bridge_type: Type of bridge model.
        bridge_hidden_size: Hidden size for bridge networks.
        bridge_depth: Depth of bridge networks.
        decoder_hidden_size: Hidden size for decoder MLP.
        decoder_depth: Depth of decoder MLP.
        use_condition_in_encoder: Whether to use condition in encoder.
        use_condition_in_flow: Whether to use condition in flow.
        autonomous: Whether flow is time-homogeneous.
        key: PRNG key.

    Returns:
        Configured LatentStochasticFlow model.
    """
    from nsf_rl.models.bridge_models import (
        AffineCouplingBridgeModel,
        MultivariateNormalDiagBridgeModel,
    )
    from nsf_rl.models.decoders import MLPMultivariateNormalDiagDecoder
    from nsf_rl.models.encoders import MLPEncoder
    from nsf_rl.models.posteriors import (
        GRUResidualPosterior,
        MLPPosterior,
        MLPResidualPosterior,
    )
    from nsf_rl.models.stochastic_flows import (
        AffineCouplingStochasticFlow,
        MultivariateNormalDiagStochasticFlow,
    )

    keys = jax.random.split(key, 5)

    # Encoder
    encoder_condition_dim = condition_dim if use_condition_in_encoder else 0
    encoder = MLPEncoder(
        obs_dim=obs_dim,
        embedding_dim=encoder_embedding_dim,
        condition_dim=encoder_condition_dim,
        hidden_size=encoder_hidden_size,
        depth=encoder_depth,
        key=keys[0],
    )

    # Posterior (always receives condition)
    if posterior_type == "mlp":
        posterior = MLPPosterior(
            embedding_dim=encoder_embedding_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            autonomous=autonomous,
            hidden_size=posterior_hidden_size,
            depth=posterior_depth,
            key=keys[1],
        )
    elif posterior_type == "mlp_residual":
        posterior = MLPResidualPosterior(
            embedding_dim=encoder_embedding_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            autonomous=autonomous,
            hidden_size=posterior_hidden_size,
            depth=posterior_depth,
            key=keys[1],
        )
    elif posterior_type == "gru":
        posterior = GRUResidualPosterior(
            embedding_dim=encoder_embedding_dim,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            autonomous=autonomous,
            gru_hidden_size=gru_hidden_size,
            mlp_hidden_size=posterior_hidden_size,
            mlp_depth=posterior_depth,
            key=keys[1],
        )
    else:
        raise ValueError(f"Unknown posterior type: {posterior_type}")

    # Flow
    flow_condition_dim = condition_dim if use_condition_in_flow else 0
    if flow_type == "mvn":
        flow = MultivariateNormalDiagStochasticFlow(
            state_dim=latent_dim,
            condition_dim=flow_condition_dim,
            autonomous=autonomous,
            hidden_size=flow_hidden_size,
            depth=flow_depth,
            key=keys[2],
        )
    else:
        flow = AffineCouplingStochasticFlow(
            state_dim=latent_dim,
            condition_dim=flow_condition_dim,
            autonomous=autonomous,
            mvn_hidden_size=flow_hidden_size,
            mvn_depth=flow_depth,
            conditioner_hidden_size=flow_hidden_size,
            conditioner_depth=flow_depth,
            num_flow_layers=flow_num_layers,
            key=keys[2],
        )

    # Bridge
    if bridge_type == "mvn":
        bridge = MultivariateNormalDiagBridgeModel(
            state_dim=latent_dim,
            condition_dim=flow_condition_dim,
            autonomous=autonomous,
            hidden_size=bridge_hidden_size,
            depth=bridge_depth,
            key=keys[3],
        )
    else:
        bridge = AffineCouplingBridgeModel(
            state_dim=latent_dim,
            condition_dim=flow_condition_dim,
            autonomous=autonomous,
            mvn_hidden_size=bridge_hidden_size,
            mvn_depth=bridge_depth,
            conditioner_hidden_size=bridge_hidden_size,
            conditioner_depth=bridge_depth,
            key=keys[3],
        )

    # Decoder
    decoder = MLPMultivariateNormalDiagDecoder(
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        condition_dim=condition_dim,
        hidden_size=decoder_hidden_size,
        depth=decoder_depth,
        key=keys[4],
    )

    return LatentStochasticFlow(
        encoder=encoder,
        posterior=posterior,
        flow=flow,
        decoder=decoder,
        bridge=bridge,
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        condition_dim=condition_dim,
        use_condition_in_encoder=use_condition_in_encoder,
        use_condition_in_flow=use_condition_in_flow,
    )


__all__ = [
    "LatentStochasticFlow",
    "LossComponents",
    "SampleResult",
    "build_latent_stochastic_flow",
]
