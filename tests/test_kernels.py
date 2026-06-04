import jax
import jax.numpy as jnp
from diff_hdx.kernels import h_bond_energy, intrinsic_rates, protection_factors, sasa_approx, deuterium_uptake


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

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(protection_factors(x, h_bonds))

    grads = jax.grad(loss)(coords)
    assert grads.shape == coords.shape
    assert not jnp.any(jnp.isnan(grads))


def test_intrinsic_rates_bai_parity():
    """
    Verify k_int against Bai et al. (1993) Ala-Ala reference at pH 7, 20C.
    """
    # implementation uses ka=10^1.39, kb=10^10.08, kw=10^-1.5
    rates = intrinsic_rates("AAAAA", ph=7.0, temperature=293.15)
    assert jnp.allclose(rates[0], 812.8622, rtol=1e-3)


def test_deuterium_uptake_parity():
    """
    Verify deuterium uptake kinetics (D(t) = 1 - exp(-k_obs * t)).
    """
    k_int = jnp.array([10.0])
    pf = jnp.array([2.0])
    # k_obs = 10 / 2 = 5.0
    # D(0.1) = 1 - exp(-5.0 * 0.1) = 1 - exp(-0.5) = 1 - 0.6065 = 0.3935

    d_t = deuterium_uptake(pf, k_int, time=0.1)
    assert jnp.allclose(d_t[0], 0.3935, atol=1e-3)
