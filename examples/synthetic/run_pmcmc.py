"""
run PMCMC algorithm with generated data from data_sim.py
"""

from treeHPYPcts import newick_from_file, RestFranchise, TreeMCMC
from treeHPYPcts.jump_proposal import MultiProposal, RandomWalkProposal, SwapProposal
import pandas as pd
import numpy as np
import scipy as sp
import os



def main(
    tree_file,
    data_file,
    ground_truth_file,
    result_dir,
    num_samples,
    num_particles,
    num_chains,
    var_prior,
    var_prior_params,
    prior_mean_njumps,
    disc
    ):
    # load in data and produce a base distribution
    data = pd.read_csv(data_file)
    ground_truth = np.load(ground_truth_file)
    mean = data.obs.mean()
    sd = data.obs.std()
    # technically does not need to be specified
    base_dist = sp.stats.norm(mean, sd)
    proposal_method = MultiProposal(strategy='cycle')
    proposal_method.add_proposals(
        RandomWalkProposal(),
        SwapProposal(method='parent_child'),
    )



    # run algorithm
    tree = RestFranchise(newick=newick_from_file(tree_file), disc=disc)
    tree.rescale()
    mcmc = TreeMCMC(
        tree, data, num_samples, num_particles,
        var_prior = var_prior, var_prior_params = var_prior_params,
        proposal_method=proposal_method, prior_mean_njumps=prior_mean_njumps,
        base_dist=base_dist
    )

    mcmc.run(num_chains = num_chains,
             parallel = True
             )
    mcmc.save_mcmc(os.path.join(result_dir, 'mcmc.zip'))
    mcmc.summarise_results(os.path.join(result_dir, 'summary'),
                           ground_truth=ground_truth)
    mcmc.save_diagnostics(os.path.join(result_dir, 'diagnostics'))

if __name__ == '__main__':
    # replace as necessary
    tree_file = '100.newick'
    data_file = './data.csv'
    ground_truth_file = 'ground_truth.npy'
    # parameters for PMCMC algorithm
    disc = 0.5
    var_prior_params = [1.]
    num_samples = 2000
    num_particles = 5
    num_chains = 6
    var_prior = 'fixed'
    result_dir = 'results'
    prior_mean_njumps = 1.
    main(
        tree_file,
        data_file,
        ground_truth_file,
        result_dir,
        num_samples,
        num_particles,
        num_chains,
        var_prior,
        var_prior_params,
        prior_mean_njumps,
        disc
    )