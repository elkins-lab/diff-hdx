# 🧪 diff-hdx

**diff-hdx** provides differentiable kernels for predicting Hydrogen-Deuterium Exchange (HDX) protection factors from protein structures using JAX.

## Quick Start

```python
import jax
import jax.numpy as jnp
from diff_hdx import intrinsic_rates, protection_factors, deuterium_uptake

# Compute intrinsic exchange rates for a sequence (Bai et al. 1993)
k_int = intrinsic_rates("ACDEFGH", ph=7.0, temperature=293.15)

# Compute protection factors from atomic coordinates
coords  = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
h_bonds = jnp.array([1.0, 0.5])
pf = protection_factors(coords, h_bonds)

# Predict deuterium uptake at t = 10 minutes
d_t = deuterium_uptake(pf, k_int[:2], time=10.0)
print(d_t)

# Gradient of total uptake w.r.t. coordinates (for structure refinement)
loss = lambda c: jnp.sum(deuterium_uptake(protection_factors(c, h_bonds), k_int[:2], time=10.0))
grads = jax.grad(loss)(coords)
print(grads)
```
