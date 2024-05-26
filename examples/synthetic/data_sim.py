"""
Generates synthetic data given a tree by placing ONE jump and then placing a Gaussian over each partition
of the tree induced by the jump
"""

from treeHPYPcts import newick_from_file, RestFranchise, plots, Particles, evals
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import ete3
import subprocess

N_LEAVES = 100

# generate synthetic data, 1 jump
print("generating tree")
subprocess.call(['Rscript', 'createTree.R', '100'])
tree = RestFranchise(newick = newick_from_file(f"{N_LEAVES}.newick"))
tree.simulate_jps_1equally(0.1*tree.nleaf, 0.5*tree.nleaf)

plots.annotated_tree(tree=tree, circular=True,
                     show_jumps_background=True, show_node_name=False, show_leaf_name=True,
                     differentiate_jump_bg=True,
                     title="Simulated Jump",
                    file = "tree.png")

# save ground truth jumps
np.save('./ground_truth.npy', tree.jps)

# partition the tree and simulate data by placing a gaussian over each partition

partition = tree.partition_by_jps()[0]
df = pd.DataFrame(data = partition, index = list(tree.nodes.keys())).rename(columns = {0:'cluster'})
# remove non leaves
df = df.drop(df[df.index.str.contains('__')].index)
df = df.drop('')
df = df.reset_index().rename(columns = {'index':'node_name'})
# separation between the Gaussians
mean_distance = 5
obs = []
# simulate data
for cluster in df.cluster:
    if cluster == 0:
        obs.append(sp.stats.norm(mean_distance / 2, 1).rvs(1)[0])
    else:
        obs.append(sp.stats.norm(-mean_distance / 2, 1).rvs(1)[0])

# save data
df['obs'] = obs
df[['node_name', 'obs']].to_csv('data.csv', index = False)
plt.hist(df[df.cluster ==0].obs, label = 'root', alpha=0.5, density=True, color = 'steelblue')
sns.kdeplot(df[df.cluster ==0].obs, color='steelblue')
plt.hist(df[df.cluster ==1].obs, label = 'jp1', alpha = 0.5, density=True, color = 'orange')
sns.kdeplot(df[df.cluster ==1].obs, color = 'orange')
plt.hist(df[df.cluster ==2].obs, label = 'jp1', alpha = 0.5, density=True, color = 'green')
sns.kdeplot(df[df.cluster ==2].obs, color = 'green')
plt.legend()
plt.savefig('data.png', dpi = 300)



print("ksTestStatistic: ", sp.stats.kstest(df[df.cluster==0].obs, df[df.cluster==1].obs)[0])
print("tree and data saved.")
