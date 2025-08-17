# CuBFF

This project provides a (optionally) CUDA-based implementation of a
self-modifying soup of programs which show emergence of self-replicators. Most
experiments in the "Computational Life: How Well-formed, Self-replicating Programs 
Emerge from Simple Interaction" paper (arxiv link (https://arxiv.org/abs/2406.19108) were done using this code.

## Dependencies
On debian-based systems, install `build-essential` and `libbrotli-dev` (and optionally CUDA):

  `sudo apt install build-essential libbrotli-dev`

On Arch Linux, install the `brotli` and `base-devel` packages.

The project also provides a `flake.nix` file, so you may also make the
dependencies available with Nix using `nix develop`.

## Run instructions

Compile the code by running `make` (for the CUDA-enabled version) or `make
CUDA=0`.

You can then run a simulation, for example with

  `bin/main --lang bff_noheads`

The file `cubff.py` provides an example of how to use the Python bindings.

