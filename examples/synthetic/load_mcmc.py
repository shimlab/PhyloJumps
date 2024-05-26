from treeHPYPcts import TreeMCMC, RestFranchise, newick_from_file
import pandas as pd
import os

def main():

    tree = RestFranchise(newick=newick_from_file('100.newick'))
    data = pd.read_csv('data.csv')
    mcmc = TreeMCMC(tree, data)
    mcmc.load_mcmc('results/mcmc.zip', load_samples=True, load_logs=True)
    mcmc.summarise_results('results/summary', burnin=mcmc.num_samples // 2)
    mcmc.save_diagnostics('results/diagnostics')

if __name__ == '__main__':
    main()