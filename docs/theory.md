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

In **diff-hdx**, we model the Protection Factor as a function of local structural environment (SASA and H-bonding) using separate scaling coefficients:

$$\ln PF \approx \beta_{asa} \cdot (1 - SASA) + \beta_c \cdot E_{hb}$$

where $\beta_{asa}$ weights the burial contribution and $\beta_c$ weights the hydrogen-bond contribution independently.  Both default to 1.0, recovering the original single-$\beta$ form.  When fitting to experimental protection factors, $\beta_{asa}$ and $\beta_c$ should be treated as independent free parameters.

## SASA Approximation

We use a differentiable Gaussian occlusion model to estimate the Solvent Accessible Surface Area (SASA). The accessibility of atom $i$ is reduced by the proximity of neighboring atoms:

$$\text{Accessibility}_i = \frac{1}{1 + \sum_{j \neq i} \exp\!\left(-r_{ij}^2 \,/\, 2\sigma_{eff}^2\right)}$$

where $\sigma_{eff} = \sigma + r_{probe}$ combines the Gaussian smoothing width with the solvent probe radius $r_{probe}$ (default 1.4 Å, matching the water probe used in Shrake–Rupley SASA).

> **Note:** This is a differentiable *surrogate*, not a true Shrake–Rupley SASA.  It lacks per-atom van-der-Waals radii and returns dimensionless values in $(0, 1]$.  It is appropriate as a smooth proxy for gradient-based refinement but should not be used to report physical SASA values in Å².
