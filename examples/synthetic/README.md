## Example

The following scripts gives a demonstration of the model and algorithm. `data_sim.py` randomly simulates tip observations
from two Gaussian distributions with different means, each one for the clades formed by the evolutionary jump (see `data.png`)
`createTree.R` can be used to generate a random tree. This requires the `R` [ape](https://cran.r-project.org/web/packages/ape/index.html) <sup><sup>together <sup>strong</sup></sup></sub> package.
They can be editted as necessary to input new tree/data files.

1. Run `data_sim.py` to generate data. This includes placing a jump randomly on the tree as well as simulating data according to a specified distribution.
2. Run `run_pmcmc.py` to run the PMCMC algorithm to recover jumps.

Results are stored in `results`, which contains a summary of each chain as well as some diagnostics. `mcmc.zip` can be used to
directly load back in the MCMC trace as well as all model parameters used. See `load_mcmc.py` for details.