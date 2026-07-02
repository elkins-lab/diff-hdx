# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-02

### Added
- Comprehensive tests for logical cases: pH dependence, beta parameter scaling, and H-bond distance cutoffs in the Linderstrøm-Lang models.
- Google Colab interactive tutorials for HDX-MS prediction in `examples/interactive_tutorials/`.
- Links to interactive tutorials in `README.md`.

### Fixed
- The `intrinsic_rates` calculation no longer assigns a positive H-D exchange rate to Proline residues, explicitly defaulting to `0.0` as Proline lacks an exchangeable amide proton.
- The `intrinsic_rates` function correctly returns an empty array when given an empty sequence string instead of throwing index errors or returning shape `(1,)`.
- `README.md` installation instructions simplified by removing GitHub URLs.
- Fixed JSON syntax in documentation setup blocks.
- Fixed broken tutorial links due to directory restructuring.

### Changed
- Forced CI javascript actions to use Node.js 24 to fix deprecation warnings.

## [0.1.1] - 2026-06-07

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.
