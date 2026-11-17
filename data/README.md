# Sample & Benchmark Datasets

- `sample_copolymers.csv`: 30 conjugated copolymers in `wD-MPNN` extended polymer graph notation (`poly_chemprop_input, EA vs SHE (eV), IP vs SHE (eV)`) extracted from the 42,966-copolymer VIP-EA dataset (Aldeghi & Coley, *Chem. Sci.* 2022) for rapid unit testing and smoke verification.
- `sample_diblock.csv`: 30 experimental diblock copolymers (`poly_chemprop_input, lamellar, cylinder, sphere, gyroid, disordered`) with degree-of-polymerization scaling (`~Xn`) from the experimental Block Copolymer Phase Behavior Database.

## Full Datasets
Full cross-validation splits (`cv9-monomer`, `cv10-random`, `polymer-poly_type_x5`, `polymer-comp_x5`, and `diblock-phases/cv10-stratified`) are publicly hosted in the official companion data repository:
- https://github.com/coleygroup/polymer-chemprop-data
