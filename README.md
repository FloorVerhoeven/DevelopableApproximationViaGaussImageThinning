# Developable Approximation via Gauss Image Thinning

This folder contains the code related to the paper "Developable Approximation via Gauss Image Thinning" by Alexandre Binninger, Floor Verhoeven, Philipp Herholz and Olga Sorkine-Hornung. 

## Compilation

The only dependencies are `libigl` and `Eigen`. A Makefile is provided, issue `make` to compile. Please make sure you are compiling with g++.

## Running

The executable is `GaussThinning` (or `GaussThinningParallel`). The main function can be launched without entering any parameter. In this case, the program runs on a bunch of default examples with optimal parameters.

To try the code on another mesh, the arguments are:

`./GaussThinning [input] [output] [directory] [NB_ITERATIONS] [MIN_ANGLE]`

For example:


`./GaussThinning input.off out.obj ./examples/bunny 500 2.5`
