"""Losses for conditional neural stochastic flows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from nsf_rl.models.conditional_flow import ConditionalAuxiliaryFlow, ConditionalNeuralStochasticFlow


def flow_1_to_2_loss(
    *,
    stochastic_flow: ConditionalNeuralStochasticFlow,
    auxiliary_model: ConditionalAuxiliaryFlow,
    x_init: Float[Array, "..."],
    t_init: Float[Array, "..."],
    t_middle: Float[Array, "..."],
    t_final: Float[Array, "..."],
    condition: Float[Array, "..."],
    key: PRNGKeyArray,
) -> Float[Array, "..."]:
    key1, key2 = jax.random.split(key)
    p1 = stochastic_flow(x_init=x_init, t_init=t_init, t_final=t_final, condition=condition)
    x_final, logp1 = p1.sample_and_log_prob(key1)
    q = auxiliary_model(
        x_init=x_init,
        x_final=x_final,
        t_init=t_init,
        t_final=t_final,
        t_eval=t_middle,
        condition=condition,
    )
    x_middle, logq = q.sample_and_log_prob(key2)
    p2_first = stochastic_flow(x_init=x_init, t_init=t_init, t_final=t_middle, condition=condition)
    logp2_first = p2_first.log_prob(x_middle)
    p2_second = stochastic_flow(x_init=x_middle, t_init=t_middle, t_final=t_final, condition=condition)
    logp2_second = p2_second.log_prob(x_final)
    return logp1 + logq - logp2_first - logp2_second


def flow_2_to_1_loss(
    *,
    stochastic_flow: ConditionalNeuralStochasticFlow,
    auxiliary_model: ConditionalAuxiliaryFlow,
    x_init: Float[Array, "..."],
    t_init: Float[Array, "..."],
    t_middle: Float[Array, "..."],
    t_final: Float[Array, "..."],
    condition: Float[Array, "..."],
    key: PRNGKeyArray,
) -> Float[Array, "..."]:
    key1, key2 = jax.random.split(key)
    p2_first = stochastic_flow(x_init=x_init, t_init=t_init, t_final=t_middle, condition=condition)
    x_middle, logp2_first = p2_first.sample_and_log_prob(key1)
    p2_second = stochastic_flow(x_init=x_middle, t_init=t_middle, t_final=t_final, condition=condition)
    x_final, logp2_second = p2_second.sample_and_log_prob(key2)
    q = auxiliary_model(
        x_init=x_init,
        x_final=x_final,
        t_init=t_init,
        t_final=t_final,
        t_eval=t_middle,
        condition=condition,
    )
    logq = q.log_prob(x_middle)
    p1 = stochastic_flow(x_init=x_init, t_init=t_init, t_final=t_final, condition=condition)
    logp1 = p1.log_prob(x_final)
    return logp2_first + logp2_second - logq - logp1


def flow_loss(
    *,
    stochastic_flow: ConditionalNeuralStochasticFlow,
    auxiliary_model: ConditionalAuxiliaryFlow,
    x_init: Float[Array, "..."],
    t_init: Float[Array, "..."],
    t_middle: Float[Array, "..."],
    t_final: Float[Array, "..."],
    condition: Float[Array, "..."],
    key: PRNGKeyArray,
    weight_1_to_2: float = 1.0,
    weight_2_to_1: float = 1.0,
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    key1, key2 = jax.random.split(key)
    loss_12 = flow_1_to_2_loss(
        stochastic_flow=stochastic_flow,
        auxiliary_model=auxiliary_model,
        x_init=x_init,
        t_init=t_init,
        t_middle=t_middle,
        t_final=t_final,
        condition=condition,
        key=key1,
    )
    loss_21 = flow_2_to_1_loss(
        stochastic_flow=stochastic_flow,
        auxiliary_model=auxiliary_model,
        x_init=x_init,
        t_init=t_init,
        t_middle=t_middle,
        t_final=t_final,
        condition=condition,
        key=key2,
    )
    total = weight_1_to_2 * loss_12 + weight_2_to_1 * loss_21
    return total, loss_12, loss_21


def maximum_log_likelihood(
    *,
    stochastic_flow: ConditionalNeuralStochasticFlow,
    x_init: Float[Array, "..."],
    x_final: Float[Array, "..."],
    t_init: Float[Array, "..."],
    t_final: Float[Array, "..."],
    condition: Float[Array, "..."],
) -> Float[Array, "..."]:
    dist = stochastic_flow(x_init=x_init, t_init=t_init, t_final=t_final, condition=condition)
    return dist.log_prob(x_final)


def time_between(
    *,
    key: PRNGKeyArray,
    t_init: Float[Array, "..."],
    t_final: Float[Array, "..."],
) -> Float[Array, "..."]:
    u = jax.random.uniform(key, shape=t_init.shape, minval=0.0, maxval=1.0)
    return t_init + u * (t_final - t_init)


__all__ = [
    "flow_loss",
    "flow_1_to_2_loss",
    "flow_2_to_1_loss",
    "maximum_log_likelihood",
    "time_between",
]
