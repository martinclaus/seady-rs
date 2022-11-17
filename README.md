# SeaDy: A framework to build dynamical models for the ocean or atmosphere

The purpose of SeaDy is to allow for quick development of high-performance numerical models.
In contrast to other CFD frameworks, it is designed with a focus on ocean and atmospheric modelling.
This allows to expose a clear and simple to use API to the user who is implementing a numerical solver for a particular model.

## List of desired features
- [ ] Core generic data structures such as Array, Grid, Variable, State
- [ ] Abstractions of fundamental controllers (time stepping schemes, computational kernels, etc.)
- [ ] distributed asynchronous computing

## Current state
This project is in pre-alpha phase.
Currently, existing prototypes such as [blur-rs](https://github.com/martinclaus/blur-rs) and [fluidyn](https://github.com/martinclaus/fluidyn) are being absorbed into this code base.