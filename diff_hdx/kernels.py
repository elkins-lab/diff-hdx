import jax
import jax.numpy as jnp


def intrinsic_rates(
    sequence: str,
    ph: float = 7.0,
    temperature: float = 293.15,
) -> jnp.ndarray:
    """
    Compute intrinsic exchange rates (k_int) using the Bai et al. (1993) model.
    Note: Simplified implementation for core residues.

    Args:
        sequence: Protein sequence string.
        ph: pH value.
        temperature: Temperature in Kelvin.

    Returns:
        k_int (N,) rates in min^-1.
    """
    # Reference rates for Ala-Ala at 20C (293.15K)
    k_a_ref = 10**1.62
    k_b_ref = 10**10.18
    k_w_ref = 10**-1.50
    
    # [H+] and [OH-]
    h_plus = 10**-ph
    # pKw at 20C is ~14.17
    oh_minus = 10**(ph - 14.17)
    
    # Activation energies (kcal/mol)
    e_a = 14.0
    e_b = 17.0
    e_w = 19.0
    r_gas = 1.987e-3  # kcal/(mol*K)
    
    def temp_corr(k_ref, e_act):
        return k_ref * jnp.exp(-e_act / r_gas * (1.0 / temperature - 1.0 / 293.15))
    
    ka = temp_corr(k_a_ref, e_a)
    kb = temp_corr(k_b_ref, e_b)
    kw = temp_corr(k_w_ref, e_w)
    
    # Final rate for Ala (simplified as constant for now)
    k_int_ala = ka * h_plus + kb * oh_minus + kw
    
    # Return array of rates (simplified to constant for the whole sequence)
    # In a full implementation, we would apply side-chain corrections here.
    return jnp.full(len(sequence), k_int_ala)


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
