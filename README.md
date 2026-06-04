# 🧪 diff-hdx: Differentiable HDX-MS Prediction in JAX

[![Tests](https://github.com/elkins/diff-hdx/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-hdx/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)

**diff-hdx** is a high-performance Python library for differentiable Hydrogen-Deuterium Exchange (HDX-MS) prediction. Built on **JAX**, it provides auto-differentiable kernels to bridge structural ensembles and experimental protection factors.

---

## 🎯 Features

- **Differentiable SASA Kernels:** Hardware-accelerated approximations of Solvent Accessible Surface Area using Gaussian occlusion models.
- **Protection Factor Modeling:** Implementations of Linderstrøm-Lang models for H-exchange rates ($PF$).
- **Kinetic Simulation:** Model time-dependent mass shifts using **EX2 kinetics** (Hvidt & Nielsen, 1966).
- **Gradient-Based Refinement:** Optimize protein structures or ensembles directly against experimental HDX-MS time-curves.
- **Vectorized Execution:** Native support for `vmap` to handle large conformational ensembles.

---

## 🏗️ Technical Architecture

- **Backend:** JAX (XLA-compiled) — supports CPU, GPU, and TPU.
- **Differentiability:** Full support for forward and reverse-mode autodiff.
- **Integration:** Compatible with `biotite` for structural parsing and `diff-biophys` for ensemble averaging.

---

## 🚀 Roadmap

- [x] Initial differentiable SASA and $ln P$ kernels.
- [x] Integration with JAX `vmap` for ensemble averaging.
- [ ] Support for residue-specific intrinsic exchange rates (Bai et al. 1993).
- [ ] Integration with MD trajectory loaders.

---

## 🚀 Installation

```bash
pip install diff-hdx
```

## 🧪 Scientific Validation

- **Parity Checks:** Kernels are validated against standard non-differentiable implementations (e.g., `biotite` SASA) to ensure physical accuracy.
- **Gradient Tests:** All kernels are verified using JAX's `gradcheck` to ensure numerically stable derivatives across the full support.
- **Ensemble Consistency:** Verified against `diff-biophys` ensemble averaging for IDP conformational ensembles.

---

## 🔗 Related Projects

diff-hdx is part of the **differentiable biophysics** ecosystem:

- [diff-biophys](https://github.com/elkins/diff-biophys) — Core differentiable biophysics engine.
- [diff-fret](https://github.com/elkins/diff-fret) — Differentiable FRET modeling.
- [diff-epr](https://github.com/elkins/diff-epr) — Differentiable EPR/DEER simulation.
- [synth-pdb](https://github.com/elkins/synth-pdb) — Synthetic structure generation.

---

## 📖 Citation

```bibtex
@software{diff_hdx,
  author  = {Elkins, George},
  title   = {diff-hdx: Differentiable HDX-MS prediction in JAX},
  year    = {2026},
  url     = {https://github.com/elkins/diff-hdx},
  version = {0.1.0}
}
```

## ⚖️ License

MIT
