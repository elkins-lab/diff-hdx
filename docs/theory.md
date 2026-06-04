# HDX Theory

## Protection Factors

Hydrogen-Deuterium Exchange (HDX) measures the rate at which amide hydrogens in a protein exchange with the solvent. The exchange rate is usually expressed via a protection factor $P$:

$$P = \frac{k_{int}}{k_{obs}}$$

where $k_{int}$ is the intrinsic exchange rate and $k_{obs}$ is the observed rate. In **diff-hdx**, we model the logarithm of the protection factor ($ln P$) as a function of solvent accessibility and hydrogen bonding:

$$ln P \approx \beta \cdot ((1 - SASA) + E_{hb})$$

## SASA Approximation

We use a differentiable Gaussian occlusion model to estimate the Solvent Accessible Surface Area (SASA). The accessibility of atom $i$ is reduced by the proximity of neighboring atoms:

$$Accessibility_i = \frac{1}{1 + \sum_{j \neq i} \exp(-r_{ij}^2 / 2\sigma^2)}$$
