import jax
import jax.numpy as jnp


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
