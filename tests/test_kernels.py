import jax
import jax.numpy as jnp

from diff_hdx.kernels import h_bond_energy, protection_factors, sasa_approx


def test_hdx_basic():
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    sasa = sasa_approx(coords)
    assert sasa.shape == (3,)
    assert jnp.all(sasa > 0)

    h_bond_energies = jnp.array([1.0, 2.0, 1.0])
    ln_p = protection_factors(coords, h_bond_energies)
    assert ln_p.shape == (3,)


def test_h_bond_energy():
    donors = jnp.array([[0.0, 0.0, 0.0]])
    acceptors = jnp.array([[2.0, 0.0, 0.0]])  # Within 3.5 cutoff

    count = h_bond_energy(donors, acceptors)
    assert count[0] > 0.5


def test_hdx_differentiable():
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    h_bonds = jnp.array([1.0, 1.0])

    def loss(x):
        return jnp.sum(protection_factors(x, h_bonds))

    grads = jax.grad(loss)(coords)
    assert grads.shape == coords.shape
    assert not jnp.any(jnp.isnan(grads))
