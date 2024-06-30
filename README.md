# CuBFF

This project provides a (optionally) CUDA-based implementation of a
self-modifying soup of programs which show emergence of self-replicators. Most
experiments in the "Life on Computational Substrates: How Self Replicators Arise
from Simple Interactions" paper (arxiv link (https://arxiv.org/abs/2406.19108) were done using this code.

## Run instructions

Compile the code by running `make` (for the CUDA-enabled version) or `make
CUDA=0`.

You can then run a simulation, for example with

  `bin/main --lang bff_noheads`

The file `cubff.py` provides an example of how to use the Python bindings.

