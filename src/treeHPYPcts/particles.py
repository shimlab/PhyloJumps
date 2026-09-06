"""

treeHPYP: particles.py

Created on 2021-07-29 10:00

@author: Hanxi Sun, Steven Nguyen

"""

import numpy as np
import scipy as sp
import scipy.stats as stats
import concurrent.futures
from scipy.special import gamma,logsumexp
from ete3 import Tree
from .rest import TblCx
from .utils import *


# rather than having N trees we have one tree with N restaurants in each node
def deep_copy_labelled_data(labelled_data):
    return {k: labelled_data[k].copy() for k in labelled_data.keys()}

class ParticleNode(Tree):

    # keep track of cluster labels assigned to data

    def __init__(self, newick=None, newick_format=1):
        super().__init__(newick=newick, format=newick_format)
        self.njump = 0
        self.rests = []  # list of Rest
        self.n_particles = 0
        self.particles = None
        self.observed = False  # True if self is a leave in the original tree.

    def update_child_base(self, idx):
        """
        Update base measures in offspring nodes according to seats of customers & the base measure
        Needs a valid base measure at the restaurant of current node (self.rest)
        Only active when depend = True
        :return: None
        """
        children_base = self.rests[idx].p_tbl()
        for child in self.children:
            if child.njump > 0:
                child.rests[idx].base.discrete = children_base
            child.update_child_base(idx)

    def find_table(self, table = None, depend = None, idx = None):
        """
        sample cluster parameter from CRF
        """
        if depend is None:
            raise ValueError("parameter depend not specified (should be True or False). ")

        if idx is not None:
            rest = self.rests[idx]
            new_table, k = rest.find_table(table)

            if depend:
                if new_table and (not self.is_root()):
                    parent = self.up
                    parent.find_table(table=k, depend=depend, idx=idx)
                else:
                    self.update_child_base(idx)

            return k
        else:
            ks = []
            for i in range(len(self.rests)):
                k = self.find_table(depend=depend, idx=i)
                ks += [k]
            return ks

    def seat_obs(self, i, ks,  obs, labelled_data, is_first_obs = False):
        self.rests[i].seat_customer(table=ks[i], obs=obs)
        weight = -1
        if ks[i] in labelled_data[i].keys():
            if not is_first_obs:
                weight = self.particles.predictive_dens(x=obs, data=labelled_data[i][ks[i]])
            labelled_data[i][ks[i]].append(obs)
        else:
            if not is_first_obs:
                weight = self.particles.predictive_dens(x=obs, data=[])
            labelled_data[i][ks[i]] = [obs]
        return weight,i

    def pf_seat_obs_parallel(self, obs, labelled_data, executor,  is_first_obs=False, depend=None):
        ks = self.find_table(depend=depend)
        weights = np.zeros(self.n_particles)
        futures = [executor.submit(self.seat_obs, i, ks, obs, labelled_data, is_first_obs) for i in range(self.n_particles)]
        for future in concurrent.futures.as_completed(futures):
            weight, i = future.result()
            weights[i] = weight

        if is_first_obs:
            return 0,0

        weights = np.array(weights)
        avg_weight = logsumexp(weights) - np.log(self.n_particles)

        weights = np.exp(weights - logsumexp(weights))
        indices = np.random.choice(self.n_particles, self.n_particles, replace=True, p=weights)
        labelled_data = [deep_copy_labelled_data(labelled_data[i]) for i in indices]

        return avg_weight, indices

    def pf_seat_obs(self, obs, labelled_data, is_first_obs=False, depend=None):
        ks = self.find_table(depend=depend)
        weights = []
        for i in range(len(self.rests)):
            self.rests[i].seat_customer(table=ks[i], obs=obs)
            if ks[i] in labelled_data[i].keys():
                if not is_first_obs:
                    weights.append(self.particles.predictive_dens(x=obs, data=labelled_data[i][ks[i]]))
                labelled_data[i][ks[i]].append(obs)
            else:
                if not is_first_obs:
                    weights.append(self.particles.predictive_dens(x=obs, data=[]))
                labelled_data[i][ks[i]] = [obs]

        if is_first_obs:
            return 0,0

        weights = np.array(weights)
        # print(weights)
        avg_weight = logsumexp(weights) - np.log(self.n_particles)

        weights = np.exp(weights - logsumexp(weights))
        indices = np.random.choice(self.n_particles, self.n_particles, replace=True, p=weights)
        labelled_data = [deep_copy_labelled_data(labelled_data[i]) for i in indices]
        return avg_weight, indices

    def traverse(self, strategy='preorder', is_leaf_fn=None):
        if strategy == "preorder":
            return self._iter_descendants_preorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "levelorder":
            return self._iter_descendants_levelorder(is_leaf_fn=is_leaf_fn)
        elif strategy == "postorder":
            return self._iter_descendants_postorder(is_leaf_fn=is_leaf_fn)

    # def seat_new_obs(self, obs=None, depend=None, idx=None):
    #     if depend is None:
    #         raise ValueError("parameter depend not specified (should be True or False). ")
    #
    #     # self = self
    #     # while (not self.is_root()) and self.njump == 0:
    #     #     self = self.up
    #
    #     if idx is not None:
    #         rest = self.rests[idx]
    #         new_table, k = rest.seat_new_customer(obs)  # if obs is None, then generates an obs from the prior
    #         if depend:
    #             if new_table and (not self.is_root()):
    #                 parent = self.up
    #                 parent.seat_new_obs(obs=k, depend=depend, idx=idx)
    #             else:
    #                 self.update_child_base(idx)
    #         return k
    #     else:
    #         ks = []
    #         for i in range(len(self.rests)):
    #             k = self.seat_new_obs(obs=obs, depend=depend, idx=i)
    #             ks += [k]
    #         return ks


class Particles:
    def __init__(self, tree, num_particles, base = None, var_prior = None, kernel_var = None, forward=True):
        """
        Calculate the estimated log-likelihood with particle filtering
        :param tree:
        :param num_particles: number of particles
        :param forward: boolean, if True, then will look forward one step to calculate weight
        :return: estimated log-likelihood p(data|jps)
        """
        self.root = ParticleNode(newick=tree.root.write(format=1))

        self.n = num_particles  # number of particles
        self.forward = forward
        self.depend = tree.depend
        self.base = base
        self.tree = tree
        # get this through tree later, for now just hard code to 1
        self.kernel_var = kernel_var
        self.var_prior = var_prior
        self.labelled_data = None


        # get nodes & update info according to tree (name, njump, rests)
        self.nodes = {}
        for n, n0 in zip(self.root.traverse(), tree.root.traverse()):
            n.name = n0.name
            n.njump = n0.njump
            n.rests = [n0.rest.deep_copy() for _ in range(num_particles)] if n.njump > 0 else n.up.rests
            n.n_particles = num_particles
            n.particles = self
            self.nodes[n.name] = n


    def predictive_dens(self, x, data = None):
        """
        density of x under observed data
        if data is [] then calculate the prior density f(x)
        """

        if self.var_prior == "wishart":

            if len(data) == 0:
                return sp.stats.t(df = 2 * self.base.alpha, loc = self.base.mu0, scale = self.base.beta / self.base.alpha).logpdf(x)

            n = len(data)
            xbar = np.mean(data)
            nu = self.base.nu + n
            mu = (self.base.nu * self.base.mu0 + np.sum(data)) / nu
            alpha = self.base.alpha + n / 2
            beta = self.base.beta + (1/2) * (np.var(data) * (n - 1) + (n * self.base.nu * (xbar - self.base.mu0)**2 / nu))
            scale = beta * (nu + 1) / (alpha * nu)

            return sp.stats.t(df = 2* alpha, loc = mu, scale = scale).logpdf(x)

        else:
            if len(data) == 0:
                return stats.norm(loc = self.base.mean(),
                                  scale = np.sqrt(self.base.var() + self.kernel_var)).logpdf(x)

            # assume gaussian base distribution
            prior_var = self.base.var()
            prior_mean = self.base.mean()
            known_var = self.kernel_var
            n = len(data)
            predictive_mean = (np.sum(data) * prior_var + prior_mean * known_var) / (n * prior_var + known_var)
            predictive_var = known_var * (1 + prior_var / (n * prior_var + known_var))
            return stats.norm(loc = predictive_mean,
                              scale = np.sqrt(predictive_var)).logpdf(x)

    def get_particle(self, i=0):
        i = i % self.n
        # for n0, n in zip(self.tree.root.traverse(), self.root.traverse()):
        #     n0.rest = n.rests[i]

        return self.labelled_data[i]



    def particle_filter_integrated(self, data, executor = None):  # out_log=False, log=None
        """
        Run whole particle filtering step with given data
        particle filter algorithm 2 - with phi integrated out
        :param data: pandas DataFrame of data
        # :param out_log: todo boolean, if True, will print rests and weights after each update
        # :param log_file: todo str of None, the output log file (default: out_log.txt)
        # :param executor: parallel worker
        :return: estimated log-likelihood
        """
        loglik = self.predictive_dens(data['obs'][0], data = [])
        ndt = data.shape[0]
        labelled_data = [{} for _ in range(self.n)]

        for idx in range(ndt):  # idx = 0
            node_name, obs = data.iloc[idx]
            node = self.nodes[node_name]

            #print(labelled_data)
            # can parallelise the seating of observations and computing of weights

            if executor is None:
                avg_weight, resample_indices = node.pf_seat_obs(obs, depend=self.depend, is_first_obs = (idx == 0),
                                                                labelled_data = labelled_data)
            else:
                avg_weight, resample_indices = node.pf_seat_obs_parallel(obs, depend=self.depend, executor=executor,
                                                                            labelled_data = labelled_data,
                                                                             is_first_obs = (idx == 0)
                                                                             )
            # if the first observation don't do any weighting steps

            if (idx == 0):
                continue

            # is very slow
            for n in self.root.traverse():
                if n.njump == 0:
                    n.rests = n.up.rests
                else:
                    rests = [n.rests[i].deep_copy() for i in resample_indices]
                    n.rests = rests

            loglik += avg_weight

        self.labelled_data = labelled_data

        return loglik

    # def particle_filter(self, data):  # out_log=False, log=None
    #     """
    #     Run whole particle filtering step with given data
    #     :param data: pandas DataFrame of data
    #     # :param out_log: todo boolean, if True, will print rests and weights after each update
    #     # :param log_file: todo str of None, the output log file (default: out_log.txt)
    #     :return: estimated log-likelihood
    #     """
    #     loglik = 0.
    #     ndt = data.shape[0]
    #     labelled_data = [{} for _ in range(self.n)]
    #
    #     for idx in range(ndt):  # idx = 0
    #         node_name, obs = data.iloc[idx]
    #         node = self.nodes[node_name]
    #
    #         ks = node.seat_obs(obs, depend=self.depend)
    #         # weights are f(x | theta)
    #         weights = [(stats.norm(loc = k, scale = np.sqrt(self.kernel_var)).pdf(obs)) for k in ks]
    #
    #         avg_weight = sum(weights) / self.n
    #         loglik += np.log(avg_weight)
    #
    #         # resample the particles
    #         weights = [w / sum(weights) for w in weights]
    #         indices = np.random.choice(self.n, self.n, replace=True, p=weights)
    #         for n in self.root.traverse():
    #             if n.njump == 0:
    #                 n.rests = n.up.rests
    #             else:
    #                 rests = [n.rests[i].deep_copy() for i in indices]
    #                 n.rests = rests
    #
    #     return loglik
