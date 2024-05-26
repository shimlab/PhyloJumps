"""

treeHPYP: utils.py

Created on 2021-07-29 10:00

@author: Hanxi Sun

"""
import scipy as sp
import numpy as np
from ete3 import Tree

def conjNormalPost(prior, data, variance):
    """
    Returns a normal posterior given normal prior and likelihood is normal,
    with known variance
    """
    prior_mu = prior.args[0]
    prior_var = prior.args[1] ^ 2
    n = len(data)

    post_var = 1 / (1 / prior_var + n / variance)
    post_mu = (prior_mu / prior_var + sum(data) / variance) * post_var

    return sp.stats.norm(post_mu, np.sqrt(post_var))

def newick_rename_nodes(newick):
    tr = Tree(newick, format=1)
    idx = 0
    for n in tr.traverse("preorder"):
        if not n.is_leaf():
            n.name = "__node__{}".format(idx)
            idx += 1
    return tr.write(format=3)


def newick_from_file(filename, rename=True):
    with open(filename, "r") as f:
        newick = f.read().replace('\n', '')
    if rename:
        newick = newick_rename_nodes(newick)  # print(tree)
    return newick



