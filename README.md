# GaussThinning

This folder contains the code related to the Paper "Developable Approximation via Gauss Image Thinning". 

## Compilation

The only dependencies are `libigl` and `Eigen`. A Makefile is provided. Type `make` to compile. Please make sure you are compiling with g++.

## Running

The executable is `GaussThinning`. The main function can be launched without entering any parameter. In this case, the program runs on a bunch of default examples with optimal parameters.

To try the code on another mesh, the arguments are:

`./GaussThinning [input] [output] [directory] [NB_ITERATIONS] [MIN_ANGLE]`

For example:


`./GaussThinning input.off out.obj ./examples/bunny 500 2.5`