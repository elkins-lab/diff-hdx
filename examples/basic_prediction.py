import jax
import jax.numpy as jnp

from diff_hdx import h_bond_energy, protection_factors, sasa_approx


def main() -> None:
    # 1. Create a mock structural segment (linear chain of 5 atoms)
    coords = jnp.zeros((5, 3))
    coords = coords.at[:, 0].set(jnp.linspace(0, 12, 5))

    # 2. Estimate SASA (Occlusion)
    sasa = sasa_approx(coords)

    # 3. Compute H-bond energies (mock acceptors nearby)
    acceptors = jnp.array([[1.5, 2.0, 0.0], [4.5, 2.0, 0.0]])
    hb_energies = h_bond_energy(coords, acceptors)

    # 4. Predict Protection Factors
    pf = protection_factors(coords, hb_energies)

    print(f"Coordinates:\n{coords}")
    print(f"Approximate SASA: {sasa}")
    print(f"H-bond Energies: {hb_energies}")
    print(f"Protection Factors: {pf}")

    # 5. Gradient Descent (Refinement)
    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum((protection_factors(x, hb_energies) - 5.0) ** 2)

    grads = jax.grad(loss)(coords)
    print(f"Gradients on coordinates:\n{grads}")


if __name__ == "__main__":
    main()
