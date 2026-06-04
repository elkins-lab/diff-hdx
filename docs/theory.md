# HDX Theory

## Linderstrøm-Lang Model

Hydrogen-Deuterium Exchange (HDX) measures the rate at which amide hydrogens in a protein exchange with the solvent. The process is modeled as a two-step kinetic scheme:

$$\text{Closed} \xrightleftharpoons[k_{cl}]{k_{op}} \text{Open} \xrightarrow{k_{int}} \text{Exchanged}$$

In the common **EX2 regime** ($k_{cl} \gg k_{int}$), the observed rate $k_{obs}$ is proportional to the equilibrium opening constant $K_{op} = k_{op}/k_{cl}$ (Hvidt & Nielsen, 1966):

$$k_{obs} = K_{op} \cdot k_{int}$$

The **Protection Factor (PF)** is defined as:

$$PF = \frac{1}{K_{op}} = \frac{k_{int}}{k_{obs}}$$

## Deuterium Uptake Kinetics

The time-dependent fractional deuterium uptake $D(t)$ for a single residue is given by:

$$D(t) = 1 - \exp(-k_{obs} \cdot t)$$

In **diff-hdx**, we model the Protection Factor as a function of local structural environment (SASA and H-bonding):

$$ln PF \approx \beta \cdot ((1 - SASA) + E_{hb})$$

## SASA Approximation

We use a differentiable Gaussian occlusion model to estimate the Solvent Accessible Surface Area (SASA). The accessibility of atom $i$ is reduced by the proximity of neighboring atoms:

$$Accessibility_i = \frac{1}{1 + \sum_{j \neq i} \exp(-r_{ij}^2 / 2\sigma^2)}$$
