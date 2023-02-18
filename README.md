# GenericBayes.jl

This is a Julia package for Markov Chain Monte Carlo (MCMC) using ideas 
from Information Geometry, i.e. the geometry of statistical models. The
methods implemented here are described in my [PhD thesis](https://researchportal.bath.ac.uk/en/studentTheses/geometric-markov-chain-monte-carlo).

The main idea of these MCMC algorithms is to make use of the dual geometry
of Amari and Chentsov. For example, one method samples complementary primal
and dual variables, which results in orthogonal steps in parameter space - 
where orthogonality is with respect to the Riemannian metric (e.g. the Fisher
Information of a statistical model). These algorithms have been shown to
have low integrated autocorrelation times.

## Install as Development Package

Press `;` in Julia REPL to enter shell and `cd` into `GenericBayes.jl`.
Press `]` in Julia REPL to enter package mode, and do `develop .`.
You should now be able to do `using GenericBayes`.
Remember: do `using Revise` BEFORE `using GenericBayes` to make changes
to `GenericBayes` without having to reload it each time.
