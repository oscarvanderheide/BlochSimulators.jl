[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://oscarvanderheide.github.io/BlochSimulators.jl/dev)

# BlochSimulators

BlochSimulators is a Julia package for performing Bloch simulations within the context of Magnetic Resonance Imaging. It allows one to build custom sequence and trajectory structs to perform MR signal simulations on different computational resources, including CUDA compatible GPU cards. The development aim of package was to achieve the highest possible runtime performance. The package can be used to simulate dictionaries for [MR Fingerprinting](https://doi.org/10.1038/nature11971) or to perform forward model evaluations for [MR-STAT](https://doi.org/10.1016/j.mri.2017.10.015).

![BlochSimulators.jl graphical overview ](./docs/src/overview.pdf)
#### Installation

BlochSimulators is registered in the General Julia registry. To install the package, enter the Pkg REPL by pressing `]`, followed by either

`pkg> add BlochSimulators` (if you want to use the package as-is)

or 

`pkg> dev BlochSimulators` (if you want to make modificatios to the source code).
#### Citation

See `CITATION.bib` for details on how to cite this work.
