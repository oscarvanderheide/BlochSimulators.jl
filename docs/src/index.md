```@meta
CurrentModule = BlochSimulators
```
# BlochSimulators

BlochSimulators is a Julia package for performing Bloch simulations within the context of Magnetic Resonance Imaging. It provides functionality to perform MR signal simulations of custom sequences and k-space trajectories. Simulations can be deployed on different computational resources, including CUDA compatible GPU cards. The aim of package was to achieve the highest possible runtime performance. The package can be used to simulate dictionaries for [MR Fingerprinting](https://doi.org/10.1038/nature11971) or to perform forward model evaluations for [MR-STAT](https://doi.org/10.1016/j.mri.2017.10.015).

#### Installation

BlochSimulators.jl is registered in the General Julia registry. To install the package, press `]` in the Julia REPL to enter package mode, followed by either

`pkg> add BlochSimulators` (if you want to use the package as-is)

or 

`pkg> dev BlochSimulators` (if you want to make modificatios to the source code).

#### Citation

See `CITATION.bib` for details on how to cite this work.
