import jax
import jax.numpy as jnp
from diff_hdx.kernels import sasa_approx, protection_factors, h_bond_energy, intrinsic_rates


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


def test_intrinsic_rates_bai_parity():
    """
    Verify k_int against Bai et al. (1993) Ala-Ala reference at pH 7, 20C.
    k_int = ka[H+] + kb[OH-] + kw
    """
    # At pH 7, 20C:
    # [H+] = 1e-7, [OH-] = 10^(7 - 14.17) = 6.76e-8
    # ka = 10^1.62, kb = 10^10.18, kw = 10^-1.5
    # k_int = (41.69 * 1e-7) + (1.51e10 * 6.76e-8) + 0.0316
    # k_int = 0.000004 + 1023.29 + 0.0316 = 1023.32 min^-1
    
    rates = intrinsic_rates("AAAAA", ph=7.0, temperature=293.15)
    assert jnp.allclose(rates[0], 1023.32, rtol=1e-3)
