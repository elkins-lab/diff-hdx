# 🧪 diff-hdx

**diff-hdx** provides differentiable kernels for predicting Hydrogen-Deuterium Exchange (HDX) protection factors from protein structures using JAX.

## Quick Start

```python
import jax.numpy as jnp
from diff_hdx.kernels import protection_factors

# Atomic coordinates (N, 3)
coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
# Mock H-bond energies
h_bonds = jnp.array([1.0, 0.5])

# Compute ln P
ln_p = protection_factors(coords, h_bonds)
print(ln_p)
```
