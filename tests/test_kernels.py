import jax
import jax.numpy as jnp

from diff_hdx.kernels import (
    deuterium_uptake,
    h_bond_energy,
    intrinsic_rates,
    protection_factors,
    sasa_approx,
)


def test_hdx_basic() -> None:
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    sasa = sasa_approx(coords)
    assert sasa.shape == (3,)
    assert jnp.all(sasa > 0)

    h_bond_energies = jnp.array([1.0, 2.0, 1.0])
    ln_p = protection_factors(coords, h_bond_energies)
    assert ln_p.shape == (3,)


def test_h_bond_energy() -> None:
    donors = jnp.array([[0.0, 0.0, 0.0]])
    acceptors = jnp.array([[2.0, 0.0, 0.0]])  # Within 3.5 cutoff

    count = h_bond_energy(donors, acceptors)
    assert count[0] > 0.5


def test_hdx_differentiable() -> None:
    coords = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    h_bonds = jnp.array([1.0, 1.0])

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(protection_factors(x, h_bonds))

    grads = jax.grad(loss)(coords)
    assert grads.shape == coords.shape
    assert not jnp.any(jnp.isnan(grads))


def test_intrinsic_rates_bai_parity() -> None:
    """
    Verify k_int against Bai et al. (1993) Ala-Ala reference at pH 7, 20C.
    """
    # implementation uses ka=10^1.39, kb=10^10.08, kw=10^-1.5
    rates = intrinsic_rates("AAAAA", ph=7.0, temperature=293.15)
    assert jnp.allclose(rates[0], 812.8622, rtol=1e-3)


def test_intrinsic_rates_neighbor_sensitivity() -> None:
    """
    Verify that right-neighbour corrections are applied to residue i+1,
    not to residue i itself (Bai et al. 1993, Table 2).

    An Ile (I) right-neighbour has ar=-0.59 (strong acid retardation) and
    br=-0.23, which must reduce k_int at pH 7 relative to poly-Ala.
    """
    rates_ala = intrinsic_rates("AAAA", ph=7.0, temperature=293.15)
    rates_ile = intrinsic_rates("AAIA", ph=7.0, temperature=293.15)

    # Residue 1 (A) has Ile as its right neighbour in "AAIA" → ar=-0.59 retards ka.
    # Residue 1 in "AAAA" has Ala as its right neighbour → ar=0.00.
    # So k_int[1] should be lower in the Ile sequence.
    assert rates_ile[1] < rates_ala[1], "Right-neighbour Ile should reduce k_int relative to Ala"


def test_deuterium_uptake_parity() -> None:
    """
    Verify deuterium uptake kinetics (D(t) = 1 - exp(-k_obs * t)).
    """
    k_int = jnp.array([10.0])
    pf = jnp.array([2.0])
    # k_obs = 10 / 2 = 5.0
    # D(0.1) = 1 - exp(-5.0 * 0.1) = 1 - exp(-0.5) = 1 - 0.6065 = 0.3935

    d_t = deuterium_uptake(pf, k_int, time=0.1)
    assert jnp.allclose(d_t[0], 0.3935, atol=1e-3)


def test_sasa_probe_radius_effect() -> None:
    """
    Verify that probe_radius actually affects sasa_approx output.

    The old bug: probe_radius was accepted as a parameter but silently
    ignored (sigma was used alone, not sigma + probe_radius).  So the
    SASA value was identical regardless of probe_radius.
    """
    coords = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

    sasa_small_probe = sasa_approx(coords, probe_radius=0.0)
    sasa_large_probe = sasa_approx(coords, probe_radius=3.0)

    # A larger probe radius widens the occlusion shell, so each atom
    # appears less accessible.
    assert jnp.all(sasa_large_probe <= sasa_small_probe), (
        "Larger probe_radius must reduce (or equal) SASA accessibility"
    )
    # They must not be identical (the bug manifested as exactly equal values)
    assert not jnp.allclose(sasa_small_probe, sasa_large_probe), (
        "probe_radius=0 and probe_radius=3.0 must produce different SASA values "
        "(old code silently ignored probe_radius)"
    )


def test_intrinsic_rates_left_neighbor_sensitivity() -> None:
    """
    Verify that left-neighbour corrections (al, bl) affect residue i via
    residue i-1, not via residue i itself.

    Val (V) has al=-0.74, which strongly retards acid catalysis when it is
    the LEFT neighbour of the target residue.
    """
    # In "AVAA", residue 2 (A at index 2) has Val as its LEFT neighbour (index 1)
    # In "AAAA", residue 2 (A at index 2) has Ala as its left neighbour
    rates_ala = intrinsic_rates("AAAA", ph=4.0, temperature=293.15)  # acid pH
    rates_val = intrinsic_rates("AVAA", ph=4.0, temperature=293.15)

    # At pH 4.0 acid catalysis dominates; al=-0.74 for Val retards ka of residue 2
    assert rates_val[2] < rates_ala[2], (
        "Val left-neighbour (al=-0.74) should reduce k_int of residue 2 at acid pH"
    )


def test_intrinsic_rates_c_terminal_boundary() -> None:
    """
    The C-terminal residue has no right neighbour; Ala placeholder must be used.
    Verify the last residue's rate equals that of an internal Ala-Ala-Ala context.
    """
    # Last residue of "AAA" has right boundary → Ala
    # Second residue of "AAAA" has right neighbour = Ala
    rates_3 = intrinsic_rates("AAA", ph=7.0, temperature=293.15)
    rates_4 = intrinsic_rates("AAAA", ph=7.0, temperature=293.15)

    # For poly-Ala all corrections are 0, so all rates must be equal
    assert jnp.allclose(rates_3[-1], rates_4[-2], rtol=1e-5), (
        "C-terminal boundary condition must give same rate as internal Ala-Ala-Ala"
    )


def test_intrinsic_rates_jit_compatible() -> None:
    """
    intrinsic_rates must be compilable via jax.jit.

    The old bug: the Python for-loop in intrinsic_rates caused JAX tracing
    to unroll the loop symbolically for each residue, which works for small
    sequences but is not JIT-compatible in general (and is very slow to compile
    for long sequences).  The vectorised rewrite is properly JIT-able.
    """
    import jax

    jit_rates = jax.jit(intrinsic_rates, static_argnums=(0,))
    # Should compile and run without error
    rates = jit_rates("ACDEFGHIKLMNPQRSTVWY", ph=7.0, temperature=293.15)
    assert rates.shape == (20,)
    assert jnp.all(rates > 0)


def test_intrinsic_rates_temperature_dependence() -> None:
    """
    Rates must increase with temperature (Arrhenius activation energies > 0).
    At pH 7 where base catalysis dominates (E_b = 17 kcal/mol), raising
    temperature from 293 K to 310 K must increase k_int.
    """
    rates_cold = intrinsic_rates("AAAA", ph=7.0, temperature=293.15)
    rates_warm = intrinsic_rates("AAAA", ph=7.0, temperature=310.0)
    assert jnp.all(rates_warm > rates_cold), (
        "k_int must increase with temperature (Arrhenius behaviour)"
    )


def test_h_bond_energy_far_acceptor() -> None:
    """
    An acceptor far outside the cutoff should contribute near-zero H-bond energy.
    An acceptor close to the donor should contribute near-unity count.
    Verify the sigmoid correctly attenuates at large distances.

    h_bond_energy uses a sigmoid with sigma=0.5 A and cutoff=3.5 A.
    At r=1.5 A: sigmoid((3.5-1.5)/0.5) = sigmoid(4) = 0.982 > 0.9
    At r=20 A:  sigmoid((3.5-20)/0.5) = sigmoid(-33) ≈ 0
    """
    donors = jnp.array([[0.0, 0.0, 0.0]])
    near = jnp.array([[1.5, 0.0, 0.0]])  # 1.5 A, well within 3.5 cutoff
    far = jnp.array([[20.0, 0.0, 0.0]])  # 20 A >> cutoff

    count_near = h_bond_energy(donors, near)
    count_far = h_bond_energy(donors, far)

    assert count_near[0] > 0.9, (
        f"H-bond count must be ~1 for acceptor well within cutoff, got {count_near[0]:.4f}"
    )
    assert count_far[0] < 0.01, (
        f"H-bond count must be ~0 for acceptor far outside cutoff, got {count_far[0]:.6f}"
    )
