import jax
import jax.numpy as jnp
from jax import Array

# ---------------------------------------------------------------------------
# Bai et al. (1993) intrinsic-rate correction table.
# Stored as a fixed-order amino-acid array for vectorised (JIT-compatible) lookup.
# Columns: [al, ar, bl, br]  (log10 corrections for acid-left, acid-right,
#                              base-left, base-right catalysis)
# Order matches _AA_ORDER below.
# ---------------------------------------------------------------------------
_AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_AA_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_ORDER)}
_ALA_IDX: int = _AA_IDX["A"]

# Shape (20, 4): rows are amino acids, columns are [al, ar, bl, br]
_CORRECTIONS = [
    #      al     ar     bl     br
    [0.00, 0.00, 0.00, 0.00],  # A
    [-0.59, -0.32, 0.08, 0.22],  # R
    [-0.58, -0.13, 0.49, 0.32],  # N
    [-0.90, -0.12, 0.69, 0.60],  # D (COOH state)
    [-0.54, -0.46, 0.62, 0.55],  # C
    [-0.47, -0.27, 0.06, 0.20],  # Q
    [-0.60, -0.27, 0.24, 0.39],  # E (COOH state)
    [-0.22, 0.22, -0.03, 0.17],  # G
    [-0.10, 0.14, 0.00, 0.00],  # H
    [-0.91, -0.59, -0.73, -0.23],  # I
    [-0.57, -0.13, -0.58, -0.21],  # L
    [-0.56, -0.29, -0.04, 0.12],  # K
    [-0.64, -0.28, -0.01, 0.11],  # M
    [-0.52, -0.43, -0.24, 0.06],  # F
    [-0.19, -0.24, 0.00, 0.00],  # P
    [-0.44, -0.39, 0.37, 0.30],  # S
    [-0.79, -0.47, -0.07, 0.20],  # T
    [-0.40, -0.44, -0.41, -0.11],  # W
    [-0.41, -0.37, -0.27, 0.05],  # Y
    [-0.74, -0.30, -0.70, -0.14],  # V
]


def intrinsic_rates(
    sequence: str,
    ph: float = 7.0,
    temperature: float = 293.15,
) -> Array:
    """
    Compute intrinsic exchange rates (k_int) using the Bai et al. (1993) model.
    Includes full side-chain correction factors for all 20 standard amino acids.

    Per Bai et al. (1993) the correction for residue *i* uses:
      - the **left** neighbour  (residue i-1) via the "al" / "bl" factors, and
      - the **right** neighbour (residue i+1) via the "ar" / "br" factors.
    Boundary residues (N-terminus, C-terminus) use Ala as a placeholder.

    This implementation is fully vectorised and compatible with JAX JIT.

    Args:
        sequence: Protein sequence string (one-letter amino-acid codes).
        ph: pH value.
        temperature: Temperature in Kelvin.

    Returns:
        k_int array of shape (N,), rates in min⁻¹.
    """
    n = len(sequence)  # noqa: F841 -- kept for readability; not used in vectorised ops

    # Encode sequence as integer indices (unknown residues → Ala)
    seq_idx = [_AA_IDX.get(aa, _ALA_IDX) for aa in sequence]

    # Left-neighbour indices: residue i-1; N-terminal boundary → Ala
    left_idx = [_ALA_IDX] + seq_idx[:-1]
    # Right-neighbour indices: residue i+1; C-terminal boundary → Ala
    right_idx = seq_idx[1:] + [_ALA_IDX]

    # Look up correction arrays — pure Python lists, converted to JAX once
    corr = jnp.array(_CORRECTIONS)  # (20, 4)
    left_corr = corr[jnp.array(left_idx)]  # (N, 4)
    right_corr = corr[jnp.array(right_idx)]  # (N, 4)

    # Reference rates for NH in H₂O at 20 °C (293.15 K)
    k_a_ref = 10.0**1.39
    k_b_ref = 10.0**10.08
    k_w_ref = 10.0**-1.50  # estimated

    # [H⁺] and [OH⁻]; pKw at 20 °C ≈ 14.17
    h_plus = 10.0 ** (-ph)
    oh_minus = 10.0 ** (ph - 14.17)

    # Arrhenius temperature corrections (activation energies in kcal/mol)
    e_a, e_b, e_w = 14.0, 17.0, 19.0
    r_gas = 1.987e-3  # kcal / (mol·K)

    def temp_corr(k_ref: float, e_act: float) -> jnp.ndarray:
        return k_ref * jnp.exp(-e_act / r_gas * (1.0 / temperature - 1.0 / 293.15))  # type: ignore[no-any-return]

    ka_ref_t = temp_corr(k_a_ref, e_a)
    kb_ref_t = temp_corr(k_b_ref, e_b)
    kw_ref_t = temp_corr(k_w_ref, e_w)

    # Log-additive corrections — vectorised over all residues simultaneously
    # Columns: [al=0, ar=1, bl=2, br=3]
    ka = ka_ref_t * 10.0 ** (left_corr[:, 0] + right_corr[:, 1])  # al + ar
    kb = kb_ref_t * 10.0 ** (left_corr[:, 2] + right_corr[:, 3])  # bl + br
    kw = kw_ref_t * 10.0 ** (left_corr[:, 2] + right_corr[:, 3])  # same as kb

    return jnp.asarray(ka * h_plus + kb * oh_minus + kw)  # explicit Array, satisfies mypy


def sasa_approx(
    coords: jnp.ndarray,
    probe_radius: float = 1.4,
    sigma: float = 2.0,
) -> jnp.ndarray:
    """
    Differentiable approximation of Solvent Accessible Surface Area (SASA).
    Uses a Gaussian occlusion model.

    The probe radius is incorporated as an additive contribution to the
    effective Gaussian width (effective_sigma = sigma + probe_radius), so
    a larger probe widens the occlusion shell around each atom, reducing the
    accessible surface — consistent with standard SASA intuition.

    Note: this is a differentiable *surrogate*, not a true Shrake–Rupley SASA.
    It lacks per-atom van-der-Waals radii and returns dimensionless values in
    (0, 1].  It is suitable as a smooth proxy for gradient-based refinement.

    Args:
        coords: (N, 3) atomic coordinates in Angstroms.
        probe_radius: Radius of the solvent probe in Angstroms (default 1.4 Å).
        sigma: Base Gaussian width for the occlusion kernel in Angstroms.

    Returns:
        Approximate accessibility values (N,) in (0, 1]; 1 = fully exposed.
    """
    # Effective width combines atom-atom smoothing and the probe size
    effective_sigma = sigma + probe_radius

    # Pairwise squared distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)

    # Occlusion kernel: nearby atoms reduce accessibility.
    # Subtract the self-contribution (exp(0) = 1) from each row.
    occlusion = jnp.sum(jnp.exp(-dist_sq / (2 * effective_sigma**2)), axis=-1) - 1.0
    accessibility = 1.0 / (1.0 + occlusion)

    return accessibility


def h_bond_energy(
    donor_coords: jnp.ndarray,
    acceptor_coords: jnp.ndarray,
    cutoff: float = 3.5,
    sigma: float = 0.5,
) -> jnp.ndarray:
    """
    Compute a differentiable approximation of H-bond energy/count.
    Uses a sigmoid-like distance cutoff.

    Args:
        donor_coords: (N, 3) coordinates of donors.
        acceptor_coords: (M, 3) coordinates of acceptors.
        cutoff: Distance cutoff in Angstroms.
        sigma: Smoothing parameter for the transition.

    Returns:
        Approximate H-bond energy/count for each donor (N,).
    """
    # Compute pairwise distances (N, M)
    diff = donor_coords[:, None, :] - acceptor_coords[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    # Safe distance for gradients
    dist = jnp.sqrt(jnp.where(dist_sq > 0, dist_sq, 1.0))
    dist = jnp.where(dist_sq > 0, dist, 0.0)

    # Soft-cutoff: 1 / (1 + exp((r - r_cutoff) / sigma))
    # Sum over all potential acceptors for each donor
    hb_counts = jnp.sum(jax.nn.sigmoid((cutoff - dist) / sigma), axis=-1)
    return hb_counts


def protection_factors(
    coords: jnp.ndarray,
    h_bond_energies: jnp.ndarray,
    beta_c: float = 1.0,
    beta_asa: float = 1.0,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """
    Compute HDX protection factors (PF).
    PF = k_int / k_obs

    Uses the Linderstrøm-Lang model with separate scaling coefficients for
    H-bond and burial contributions:

        ln(PF) = beta_c * N_HB + beta_asa * (1 − SASA)

    Both coefficients default to 1.0, matching the original single-beta
    formulation for backward compatibility.  When fitting against experimental
    protection factors, beta_c and beta_asa should be treated as independent
    free parameters.

    Args:
        coords: (N, 3) coordinates.
        h_bond_energies: (N,) hydrogen bond energies (or counts).
        beta_c: Scaling coefficient for the H-bond contribution.
        beta_asa: Scaling coefficient for the burial (1 − SASA) contribution.
        probe_radius: Solvent probe radius passed to sasa_approx (Å).

    Returns:
        PF (N,) protection factors.
    """
    sasa = sasa_approx(coords, probe_radius=probe_radius)
    # ln PF = beta_asa*(1 − SASA) + beta_c*N_HB
    ln_pf = beta_asa * (1.0 - sasa) + beta_c * h_bond_energies
    return jnp.exp(ln_pf)


def deuterium_uptake(
    pf: jnp.ndarray,
    k_int: jnp.ndarray,
    time: float,
) -> jnp.ndarray:
    """
    Compute time-dependent deuterium uptake using EX2 kinetics.
    D(t) = 1 - exp(-k_obs * t)
    where k_obs = k_int / PF (Hvidt & Nielsen, 1966).

    Args:
        pf: (N,) protection factors.
        k_int: (N,) intrinsic exchange rates.
        time: Exposure time in minutes.

    Returns:
        D(t) (N,) fractional deuterium uptake.
    """
    k_obs = k_int / pf
    return 1.0 - jnp.exp(-k_obs * time)
