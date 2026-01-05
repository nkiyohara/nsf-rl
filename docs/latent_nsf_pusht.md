# Latent Neural Stochastic Flow for PushT

This document describes the Latent Neural Stochastic Flow (LNSF) implementation for the PushT task with partial observations.

## Overview

The Latent NSF extends the standard Neural Stochastic Flow to handle **partially observed systems**. In the PushT task:

- **Observable**: Agent position and velocity (4D)
- **Hidden**: Block position and orientation (4D)
- **Goal**: Learn to infer hidden state and predict future dynamics

The model learns a latent representation that captures both observable and hidden information, enabling accurate prediction even when direct observation of the full state is unavailable.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Latent Neural Stochastic Flow                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Observation (4D)     Condition (DMP params)    time_diff (Δt) │
│       │                      │                       │          │
│       ▼                      │ (optional)            │          │
│  ┌─────────┐                 │                       │          │
│  │ Encoder │◄────────────────┘                       │          │
│  └────┬────┘                                         │          │
│       │ embedding                                    │          │
│       ▼                                              ▼          │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ Posterior (GRU or MLP)                               │      │
│  │   - Input: embedding + prior + time_diff + condition │      │
│  │   - GRU: maintains temporal context across steps     │      │
│  └────────────────────┬─────────────────────────────────┘      │
│                       │ z_t (latent state)                      │
│                       │                                         │
│       ┌───────────────┴───────────────┐                        │
│       │                               │                         │
│       ▼                               ▼                         │
│  ┌─────────────────┐           ┌─────────┐                     │
│  │ Stochastic Flow │           │ Decoder │                     │
│  │ (condition=DMP) │           │         │                     │
│  └────────┬────────┘           └────┬────┘                     │
│           │                         │                           │
│           ▼                         ▼                           │
│    p(z_{t+1} | z_t)          p(obs | z_t)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Encoder (`src/nsf_rl/models/encoders.py`)

Maps observations to fixed-dimensional embeddings.

```python
class MLPEncoder(Encoder):
    """MLP-based encoder."""
    def __call__(
        self,
        obs: Array,  # [obs_dim]
        condition: Optional[Array] = None,  # [condition_dim]
    ) -> Array:  # [embedding_dim]
```

- **Input**: Observation (4D) + optional condition (DMP params)
- **Output**: Embedding vector
- **Condition routing**: Configurable via `use_condition_in_encoder`

### 2. Posterior (`src/nsf_rl/models/posteriors/`)

Infers latent state distribution given embedding and prior. **Critically includes `time_diff` (Δt) as input.**

#### GRU Posterior (Recommended)

```python
class GRUResidualPosterior(PriorConditionedPosterior):
    """GRU-based posterior with temporal context."""
    
    def step(
        self,
        prior_dist: Distribution,
        embedding: Array,        # [embedding_dim]
        time: Array,             # scalar - absolute time
        prior_initial_state: Array,  # [latent_dim] - z at t_prev
        time_diff: Array,        # scalar - Δt since previous observation
        condition: Optional[Array] = None,  # [condition_dim]
        carry: Optional[PosteriorCarry] = None,  # GRU hidden state
    ) -> tuple[Distribution, PosteriorCarry]:
```

**GRU Input**:
- Observation embedding
- `time_diff` (Δt) - elapsed time since previous observation
- Condition (DMP parameters)

**MLP Input** (after GRU):
- GRU hidden state
- Prior initial state (z at t_prev)
- `time_diff` (Δt)
- Condition

#### MLP Posterior (Alternative)

```python
class MLPResidualPosterior(PriorConditionedPosterior):
    """MLP-only posterior (no temporal memory)."""
    
    def step(
        self,
        prior_dist: Distribution,
        embedding: Array,
        time: Array,
        prior_initial_state: Array,
        time_diff: Array,        # ← Also receives time_diff
        condition: Optional[Array] = None,
        carry: Optional[PosteriorCarry] = None,
    ) -> tuple[Distribution, None]:  # No carry for MLP
```

### 3. Stochastic Flow (`src/nsf_rl/models/stochastic_flows.py`)

Models temporal evolution in latent space: p(z_{t+1} | z_t, Δt, condition)

```python
class AffineCouplingStochasticFlow(StochasticFlow):
    """Flow with affine coupling bijectors."""
    def __call__(
        self,
        x_init: Array,  # [latent_dim]
        t_init: Array,  # scalar
        t_final: Array,  # scalar
        condition: Optional[Array] = None,  # [condition_dim] - DMP params
    ) -> Distribution:
```

**Key features**:
- **Time axis**: Absolute time (seconds), not phase
- **Autonomous**: Depends only on Δt = t_final - t_init (time-homogeneous)
- **Condition**: DMP parameters are passed via `use_condition_in_flow=True`

### 4. Decoder (`src/nsf_rl/models/decoders.py`)

Decodes latent states to observation distributions.

```python
class MLPMultivariateNormalDiagDecoder(Decoder):
    """Decoder outputting diagonal Gaussian."""
    def __call__(
        self,
        z: Array,  # [latent_dim]
        condition: Optional[Array] = None,
    ) -> MultivariateNormalDiag:  # p(obs | z)
```

### 5. Bridge Model (`src/nsf_rl/models/bridge_models.py`)

Auxiliary model for flow consistency losses.

```python
class MultivariateNormalDiagBridgeModel(BridgeModel):
    """Bridge distribution q(z_mid | z_init, z_final, t_init, t_mid, t_final)."""
```

### 6. Main Model (`src/nsf_rl/models/latent_stochastic_flow.py`)

Orchestrates all components.

```python
class LatentStochasticFlow(eqx.Module):
    """Complete Latent NSF model."""
    
    def elbo(
        self,
        observations: list[Array],
        times: Array,          # [batch, T] - absolute times in seconds
        z_init: Array,
        condition: Optional[Array] = None,
        key: PRNGKey = None,
    ) -> LossComponents:
        """Compute ELBO with proper time_diff handling."""
    
    def predict_observations(
        self,
        z_init: Array,
        t_init: Array,
        t_query: Array,
        condition: Optional[Array] = None,
    ) -> Array:
        """Predict observations at future times."""
```

## Dataset and Irregular Sampling

### Why Irregular Sampling?

The raw PushT data is a **regular time series** with constant frame interval (dt). This is problematic for NSF training because:

1. **`time_diff` is always constant** → model doesn't learn to generalize to variable intervals
2. **GRU's `time_diff` input is meaningless** → always receives the same value
3. **Poor generalization at inference** → struggles with different sampling rates

### Solution: Random Subsampling

The dataset randomly selects `seq_len` frames from the full trajectory (without replacement), then sorts by time:

```
Original (regular):     [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ...]  (T_full frames)
                        |   dt   |   dt   |   dt   |   dt   |

Random selection:       Indices = sort(random.choice(T_full, size=seq_len, replace=False))
                        e.g., [0, 2, 5, 6, 9, ...]

After subsampling:      [t0, t2, t5, t6, t9, ...]
                        |  2dt  |  3dt  |  dt  |  3dt  |

time_diff varies:       [-, 2dt, 3dt, dt, 3dt, ...]
```

### Configuration

```python
dataset = PushTLatentDataset(
    root=data_path,
    rng=rng,
    seq_len=16,
    irregular_sampling=True,   # Enable random subsampling (default: True)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `irregular_sampling` | `True` | Enable random subsampling for variable time intervals |

**Recommendation**: Keep `irregular_sampling=True` for training. This ensures the model learns to handle variable `time_diff` values.

## Data Format

### Observation (4D)
```
[agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y]
```
- Normalized to [-1, 1] range

### Full State (8D) - for training supervision
```
[agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y,
 block_x, block_y, sin(block_theta), cos(block_theta)]
```

### Condition Vector (DMP parameters)
```
[stiffness, damping, dmp_dt, dmp_alpha_s, n_basis, scale_flag,
 start_x, start_y, goal_x, goal_y, weights...]
```

**Important**: 
- Duration (`tau`) is NOT included in the condition
- Phase is NOT used
- The model generalizes to different durations via absolute time input

### Time
- **Absolute time** in **seconds** (not normalized phase)
- Reason: PD control dynamics depend on real-world time
- `time_diff` (Δt) is computed as the elapsed time between consecutive observations
- **Irregular intervals** ensure the model learns to handle variable `time_diff`

## Training

### ELBO with time_diff

The `elbo()` method:
1. Iterates through the observation sequence
2. Computes `time_diff = t[i] - t[i-1]` for each step
3. Passes `time_diff` to the posterior's `step()` method
4. Maintains `PosteriorCarry` (GRU hidden state) across steps

```python
for i in range(T):
    time_diff = times[i] - times[i-1] if i > 0 else -1.0  # sentinel for first step
    
    prior = flow(z_prev, t_prev, t_curr, condition)
    posterior, carry = posterior.step(
        prior_dist=prior,
        embedding=encoder(obs[i]),
        time=times[i],
        prior_initial_state=z_prev,
        time_diff=time_diff,      # ← Critical for temporal reasoning
        condition=condition,
        carry=carry,              # ← GRU state propagation
    )
```

### Loss Function

The model is trained using the Evidence Lower Bound (ELBO):

```
ELBO = E_q[log p(obs|z)] - β·D_KL(q(z|obs) || p(z|z_prev))
     = Reconstruction - β·KL
```

Plus flow consistency losses for time-reversibility:
```
L_flow = L_{1→2} + L_{2→1}
```

Total loss:
```
L_total = -ELBO + λ_flow·L_flow
        = L_reconstruction + β·L_KL + λ_flow·(L_{1→2} + L_{2→1})
```

### KL Divergence Computation

Since the Stochastic Flow uses **Affine Coupling** layers, the prior p(z|z_prev) is a `ContinuousNormalizingFlow`, not a simple Gaussian. KL divergence is computed as follows:

1. **Posterior**: Always `MultivariateNormalDiag` (Gaussian)
2. **Prior**: `ContinuousNormalizingFlow` with Gaussian base

```python
def compute_kl_divergence(posterior, prior, sample):
    # Case 1: Both Gaussian → Analytical KL
    if isinstance(posterior, MVN) and isinstance(prior, MVN):
        return posterior.kl_divergence(prior)
    
    # Case 2: Prior is Flow with Gaussian base → KL to base (approximation)
    if isinstance(posterior, MVN) and isinstance(prior, ContinuousNormalizingFlow):
        if isinstance(prior.base_distribution, MVN):
            return posterior.kl_divergence(prior.base_distribution)
    
    # Case 3: General → Monte Carlo estimation
    # KL(q || p) ≈ log q(z) - log p(z)  where z ~ q
    return posterior.log_prob(sample) - prior.log_prob(sample)
```

**Note**: For `ContinuousNormalizingFlow`, `log_prob()` correctly computes the density using the change-of-variables formula with Jacobian log-determinant.

### Usage

```bash
python scripts/train_latent_nsf_pusht.py \
    --data-root data/random_dmp_npz \
    --latent-dim 8 \
    --seq-len 16 \
    --batch-size 64 \
    --epochs 10000 \
    --lr 1e-3 \
    --posterior-type gru \        # Use GRU for temporal context
    --gru-hidden-size 64 \
    --use-condition-in-encoder \
    --use-condition-in-flow \     # Pass DMP params to flow
    --wandb-project Latent-NSF-PushT
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 8 | Latent state dimension |
| `seq_len` | 16 | Training sequence length |
| `posterior_type` | **"gru"** | Posterior architecture ("mlp", "mlp_residual", "gru") |
| `gru_hidden_size` | 64 | GRU hidden state size |
| `flow_type` | "affine_coupling" | Flow architecture |
| `kl_weight` | 1.0 | Weight for KL divergence |
| `flow_loss_weight` | 0.1 | Weight for flow consistency |
| `use_condition_in_encoder` | True | Pass DMP params to encoder |
| `use_condition_in_flow` | True | Pass DMP params to flow |

## Default Network Architecture

PushT環境での典型的な設定（obs_dim=4, condition_dim≈30, latent_dim=8）を想定。

### Encoder (MLPEncoder)
```
Input:  obs (4D) + condition (30D) = 34D   [if use_condition_in_encoder=True]
        obs (4D) = 4D                       [if use_condition_in_encoder=False]
Hidden: 64 → tanh → 64 → tanh              (depth=2, hidden_size=64)
Output: embedding (64D)
```

### Posterior (GRUResidualPosterior) ← **Default**
```
┌─────────────────────────────────────────────────────────────┐
│ GRU Cell                                                     │
│   Input:  embedding (64D) + time_diff (1D) + condition (30D) │
│           = 95D   [if autonomous=True]                       │
│   Hidden: 64D                                                │
│   Output: hidden_state (64D)                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Output MLP                                                   │
│   Input:  hidden_state (64D) + z_prev (8D) + time_diff (1D) │
│           + condition (30D) = 103D                           │
│   Hidden: 64 → tanh → 64 → tanh  (depth=2)                  │
│   Output: mean_residual (8D) + log_scale (8D) = 16D         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
        Posterior: N(prior_mean + residual, prior_scale * softplus(log_scale))
```

### Stochastic Flow (AffineCouplingStochasticFlow)
```
┌─────────────────────────────────────────────────────────────┐
│ Base MVN Network                                             │
│   Input:  z (8D) + time_diff (1D) + condition (30D) = 39D   │
│   Hidden: 64 → tanh → 64 → tanh  (depth=2)                  │
│   Output: mean_shift (8D) + raw_scale (8D) = 16D            │
│                                                              │
│   mean = z + mean_shift * Δt                                │
│   scale = softplus(raw_scale) * sqrt(Δt)                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Affine Coupling Layers (×4)                                  │
│   Conditioner MLP:                                           │
│     Input:  z_masked (8D) + z_prev (8D) + time_diff (1D)    │
│             + condition (30D) = 47D                          │
│     Hidden: 64 → tanh → 64 → tanh  (depth=2)                │
│     Output: shift (8D) + log_scale (8D)                      │
│   Alternating mask: [T,F,T,F,T,F,T,F] ↔ [F,T,F,T,F,T,F,T]   │
└─────────────────────────────────────────────────────────────┘
```

### Bridge Model (MultivariateNormalDiagBridgeModel)
```
Input:  z_init (8D) + z_final (8D) + time_features (3D) + condition (30D)
        = 49D  [time_features = (t-t_init, t_final-t, t_final-t_init)]
Hidden: 64 → tanh → 64 → tanh  (depth=2)
Output: mean_shift (8D) + raw_scale (8D) = 16D

mean = linear_interpolation(z_init, z_final) + correction
scale = softplus(raw_scale) * sqrt(α(1-α))  where α = (t-t_init)/(t_final-t_init)
```

### Decoder (MLPMultivariateNormalDiagDecoder)
```
Input:  z (8D) + condition (30D) = 38D
Hidden: 64 → tanh → 64 → tanh  (depth=2)
Output: mean (4D) + raw_scale (4D) = 8D

Observation distribution: N(mean, softplus(raw_scale))
```

### Summary Table

| Component | Input Dim | Hidden | Output Dim | Params (approx) |
|-----------|-----------|--------|------------|-----------------|
| **Encoder** | 34 | 64×2 | 64 | ~6.5K |
| **GRU Cell** | 95 | 64 | 64 | ~30K |
| **Posterior MLP** | 103 | 64×2 | 16 | ~11K |
| **Flow Base MVN** | 39 | 64×2 | 16 | ~6.5K |
| **Flow Coupling ×4** | 47 | 64×2 | 16 | ~26K |
| **Bridge** | 49 | 64×2 | 16 | ~7K |
| **Decoder** | 38 | 64×2 | 8 | ~6K |
| **Total** | - | - | - | **~93K** |

## Design Decisions

### 1. Why Absolute Time (not Phase)?

For PushT with PD control:
- The robot's response depends on **real-world time** dynamics
- PD control has characteristic timescales (rise time, settling time)
- Phase normalization would conflate these physical dynamics

### 2. Why time_diff (Δt) in Posterior?

- The posterior must understand **temporal context** to infer hidden states
- Δt tells the model how much the system could have evolved since the last observation
- GRU accumulates information over variable time intervals using Δt

### 3. Why GRU Posterior?

- **Temporal memory**: GRU maintains context from previous observations
- **Variable timing**: Handles irregular observation intervals via time_diff
- **Sequential inference**: Natural fit for time-series data

### 4. Why DMP params to Flow (not Encoder)?

- **Flow** models dynamics: p(z_{t+1} | z_t, condition)
- DMP parameters **directly affect dynamics** (stiffness, damping, goal)
- Encoder can optionally use condition for contextualizing observations

### 5. Why Condition Always to Posterior?

The posterior needs all available information to infer the correct latent state:
- Observation embedding
- Prior distribution (from flow)
- DMP parameters (condition)
- Time information (time_diff)

This ensures accurate inference even when dynamics vary with DMP params.

## File Structure

```
src/nsf_rl/
├── data/
│   ├── observation.py           # Observation classes
│   └── pusht_latent_dataset.py  # PushT dataset for Latent NSF
├── models/
│   ├── stochastic_flows.py      # Stochastic flow models
│   ├── bridge_models.py         # Bridge models for flow loss
│   ├── encoders.py              # Encoder models
│   ├── decoders.py              # Decoder models
│   ├── posteriors/              # Posterior models
│   │   ├── __init__.py
│   │   ├── base.py              # PosteriorCarry, step() interface
│   │   ├── mlp.py               # MLP posteriors with time_diff
│   │   └── gru.py               # GRU posterior with temporal context
│   └── latent_stochastic_flow.py  # Main LNSF model

scripts/
└── train_latent_nsf_pusht.py    # Training script

docs/
└── latent_nsf_pusht.md          # This document
```

## References

- [Neural Stochastic Flows (Deng et al.)](https://arxiv.org/abs/xxx)
- [`jax-nsf` reference implementation](../ref/jax-nsf/)
- [PushT environment documentation](../ref/pusht.md)
