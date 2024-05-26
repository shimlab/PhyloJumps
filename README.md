# Background

Historical biological data shows that the distribution of phenotypes (biological traits such as height, mass, etc.) along a phylogenetic tree evolves over time, with the rate of evolution not being constant, but rather also including periods of rapid evolutionary ’jumps’. Given observations at the the leaves of a phylogenetic tree, we wish to identify where these jumps have occured.

We follow previous work with a non-parametric Bayesian model based on the Pitman-Yor Process, modelling each node distribution as a Gaussian mixture model with a Pitman-Yor
prior. Distributions after a jump on the tree are centered on the previous node, with
this hierarchy reflecting the connection between species within the tree.

# Installation

## `pip`

Using `pip`, install PhyloJumps to your current environment by running

```sh
pip install git+https://github.com/shimlab/PhyloJumps.gitc
```

This is equivalent to cloning the environment and building the package.

```sh
git clone https://github.com/shimlab/PhyloJumps.git
cd PhyloJumps
python -m pip install .
```

## `conda` environment

Included in the current repository is also a conda `environment.yaml` file to replicate the author's Python 3.8 environment.

```sh
conda env create --file=environment.yaml
```

The default name of the conda environment is `phylojumps`. 

# Usage

## Examples

See `examples`.

## Documentation

TODO





