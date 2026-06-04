import jax
import jax.numpy as jnp


def intrinsic_rates(
    sequence: str,
    ph: float = 7.0,
    temperature: float = 293.15,
) -> jnp.ndarray:
    """
    Compute intrinsic exchange rates (k_int) using the Bai et al. (1993) model.
    Includes full side-chain correction factors for all 20 standard amino acids.

    Args:
        sequence: Protein sequence string.
        ph: pH value.
        temperature: Temperature in Kelvin.

    Returns:
        k_int (N,) rates in min^-1.
    """
    # Side-chain correction factors (L and R) for Acid and Base catalysis
    # Data from Bai et al. (1993). Factors are log10 corrections.
    corrections = {
        "A": {"al": 0.00, "ar": 0.00, "bl": 0.00, "br": 0.00},
        "R": {"al": -0.59, "ar": -0.32, "bl": 0.08, "br": 0.22},
        "N": {"al": -0.58, "ar": -0.13, "bl": 0.49, "br": 0.32},
        "D": {"al": -0.90, "ar": -0.12, "bl": 0.69, "br": 0.60},  # COOH state
        "C": {"al": -0.54, "ar": -0.46, "bl": 0.62, "br": 0.55},
        "Q": {"al": -0.47, "ar": -0.27, "bl": 0.06, "br": 0.20},
        "E": {"al": -0.60, "ar": -0.27, "bl": 0.24, "br": 0.39},  # COOH state
        "G": {"al": -0.22, "ar": 0.22, "bl": -0.03, "br": 0.17},
        "H": {"al": -0.10, "ar": 0.14, "bl": 0.00, "br": 0.00},
        "I": {"al": -0.91, "ar": -0.59, "bl": -0.73, "br": -0.23},
        "L": {"al": -0.57, "ar": -0.13, "bl": -0.58, "br": -0.21},
        "K": {"al": -0.56, "ar": -0.29, "bl": -0.04, "br": 0.12},
        "M": {"al": -0.64, "ar": -0.28, "bl": -0.01, "br": 0.11},
        "F": {"al": -0.52, "ar": -0.43, "bl": -0.24, "br": 0.06},
        "P": {"al": -0.19, "ar": -0.24, "bl": 0.00, "br": 0.00},
        "S": {"al": -0.44, "ar": -0.39, "bl": 0.37, "br": 0.30},
        "T": {"al": -0.79, "ar": -0.47, "bl": -0.07, "br": 0.20},
        "W": {"al": -0.40, "ar": -0.44, "bl": -0.41, "br": -0.11},
        "Y": {"al": -0.41, "ar": -0.37, "bl": -0.27, "br": 0.05},
        "V": {"al": -0.74, "ar": -0.30, "bl": -0.70, "br": -0.14},
    }

    # Reference rates for NH in H2O at 20C (293.15K)
    k_a_ref = 10**1.39
    k_b_ref = 10**10.08
    k_w_ref = 10**-1.50  # estimated

    # [H+] and [OH-]
    h_plus = 10**-ph
    # pKw at 20C is ~14.17
    oh_minus = 10 ** (ph - 14.17)

    # Activation energies (kcal/mol)
    e_a, e_b, e_w = 14.0, 17.0, 19.0
    r_gas = 1.987e-3  # kcal/(mol*K)

    def temp_corr(k_ref: float, e_act: float) -> jnp.ndarray:
        return k_ref * jnp.exp(-e_act / r_gas * (1.0 / temperature - 1.0 / 293.15))

    ka_ref_t = temp_corr(k_a_ref, e_a)
    kb_ref_t = temp_corr(k_b_ref, e_b)
    kw_ref_t = temp_corr(k_w_ref, e_w)

    k_ints = []
    for i in range(len(sequence)):
        # Amino acid corrections for residue i
        # Left neighbor (i-1) and Right neighbor (i)
        aa_l = sequence[i - 1] if i > 0 else "A"
        aa_r = sequence[i]

        corr_l = corrections.get(aa_l, corrections["A"])
        corr_r = corrections.get(aa_r, corrections["A"])

        # Log-additive corrections
        ka = ka_ref_t * 10 ** (corr_l["al"] + corr_r["ar"])
        kb = kb_ref_t * 10 ** (corr_l["bl"] + corr_r["br"])
        # Water rate usually corrected similarly to base or acid depending on model,
        # here we use a simplified sequence-independent kw or similar to ka/kb.
        kw = kw_ref_t * 10 ** (corr_l["bl"] + corr_r["br"])

        k_ints.append(ka * h_plus + kb * oh_minus + kw)

    return jnp.array(k_ints)


def sasa_approx(
    coords: jnp.ndarray,
    probe_radius: float = 1.4,
    sigma: float = 2.0,
) -> jnp.ndarray:
    """
    Differentiable approximation of Solvent Accessible Surface Area (SASA).
    Uses a Gaussian occlusion model.

    Args:
        coords: (N, 3) atomic coordinates.
        probe_radius: Radius of the solvent probe.
        sigma: Smoothing parameter for the occlusion kernel.

    Returns:
        Approximate SASA values (N,).
    """
    # Compute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)

    # Occlusion kernel: nearby atoms reduce accessibility
    # This is a simplified model: accessibility ~ 1 / (1 + sum(K(r)))
    occlusion = jnp.sum(jnp.exp(-dist_sq / (2 * sigma**2)), axis=-1) - 1.0
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
    beta: float = 1.0,
) -> jnp.ndarray:
    """
    Compute HDX protection factors (ln P).

    Args:
        coords: (N, 3) coordinates.
        h_bond_energies: (N,) hydrogen bond energies (or counts).
        beta: Scaling factor.

    Returns:
        ln P (N,) protection factors.
    """
    sasa = sasa_approx(coords)
    # ln P ~ (1 - SASA) + H-bonds
    ln_p = beta * ((1.0 - sasa) + h_bond_energies)
    return ln_p
