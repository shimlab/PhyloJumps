"""

treeHPYP: rest_franchise.py

Created on 2021-07-29 10:00

@author: Hanxi Sun
The Chinese Restaurant Franchise. Contains the tree (as a rest_node) and associated model parameters.
    Provides a method for the main inference algorithm (particleMCMC) and a variety of utilities for managing
    the tree's jumps.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations
import warnings
import os
import concurrent.futures
import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as sps
import logging
import random
import multiprocessing
from ete3 import TreeStyle, NodeStyle, TextFace
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.colors as Colors
from matplotlib import cm
from scipy.stats import poisson, chisquare, invgamma, norm, gamma
from tqdm import tqdm
from .norminvgamma import norminvgamma
from .rest import Rest
from .rest_node import RestNode
from .particles import Particles
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .jump_proposal import Proposal

# stuff for plotting
def signif_rgb(v, if_less, criteria, MAX=1., MIN=0.):
    cmap = cm.get_cmap("Reds")
    if criteria is None:
        signif = 1.
    elif if_less:
        signif = (v - MIN) / (criteria - MIN) if criteria > MIN else 1.
    else:
        signif = (MAX - v) / (MAX - criteria) if criteria < MAX else 1.
    return to_hex(cmap(signif * .5 + .5))


def rgb_scale(n, gray_scale=False, CMAP="Paired", light_bg=False):
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    # gray: 1. = black; 0. = white
    # light_bg (for reverse cmaps): 1. = lightest; 0.= darkest
    grid = np.linspace(.2 if gray_scale else (0. if not light_bg else .5), 1., n)
    if type(CMAP) == str:
        cmap = cm.get_cmap("Greys" if gray_scale else CMAP)
    else:
        cmap = CMAP
    return cmap(grid)


def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=200):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = Colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

class RestFranchise:
    """
    The Chinese Restaurant Franchise. Contains the tree (as a rest_node) and associated model parameters.
    Provides a method for the main inference algorithm (particleMCMC) and a variety of utilities for managing
    the tree's jumps.
    """

    def __init__(self, newick=None, newick_format=1, disc=0.5, conc=0,
                 depend=True, var_prior = None, prior_params = None, base = None):
        """
        Initialize the tree. For inference, model prior does not need to be directly set (can be input in particleMCMC)
        but do need to be set to generate data from the prior.
        :param newick: inherited from ete3.Tree. Either a Newick str or a path to a newick file.
        :param newick_format: inherited from ete3.Tree
        :param disc: the discount parameter
        :param conc: the concentration parameter
        :param depend: whether there is dependency between parent and child
        :param var_prior: model to use - known mean, known variance, etc.
        :param prior_params: parameters for model
        """
        # the tree (root node)
        self._root = RestNode(newick, newick_format)
        # initialise some data

        self._root.njump = self._root.ROOT_JUMP
        self._logger = logging.getLogger(__name__)
        # parameters
        self._disc, self._conc, self._depend = self.check_parameters(disc, conc, depend)
        self._base = None
        self._var_prior = var_prior
        self._prior_params = prior_params
        self._rest_var = None

        if base is not None:
            self._base = base

        if var_prior == "fixed":
            # if the prior is normal prior_params should be (mean, std, kernel_std)
            self._base = norm(prior_params[0], prior_params[1])
            self._rest_var = prior_params[2] ** 2
        elif var_prior == "gamma":
            # params should be (mean, std, alpha, beta)
            self._base = norm(prior_params[0], prior_params[1])
            self._rest_var = invgamma.rvs(prior_params[3], prior_params[4], 1)
        elif var_prior == "wishart":
            self._base = norminvgamma(prior_params[0], prior_params[1], prior_params[2], prior_params[3])
        elif var_prior is None:
            # set an arbitrary base distribution and kernel variance just to establish the chinese restaurant franchise
            # cluster indices, that way we don't have to specify a base distribution when constructing the object
            # to do particle mcmc
            self._base = norm(0,1)
            self._rest_var = 1.

        if self._base is None:
            raise Warning("base distribution is currently not set")

        # total branch length, total number of branches, each branch length, #jumps on each branch
        self._tl, self._nb, self._bls, self._jps = self._root.all_saved_subtree_properties()  # recorded in "preorder"

        # observed nodes
        self._root.init_observed_nodes()
        self._observed = self._root.get_observed()

        # restaurants
        self.init_rests()

        # initialise some more useful data
        self.leaf_by_nodes = {}
        self.depth_by_nodes = {}

        for n in self._root.traverse(strategy='preorder'):
            if n.nleaf in self.leaf_by_nodes:
                self.leaf_by_nodes[n.nleaf].append(n)
            else:
                self.leaf_by_nodes[n.nleaf] = [n]

            if n.depth in self.depth_by_nodes:
                self.depth_by_nodes[n.depth].append(n)
            else:
                self.depth_by_nodes[n.depth] = [n]



        # name internal nodes
        if newick_format != 1:
            idx = 0
            for n in self.root.traverse("preorder"):
                if n.name is None:
                    n.name = "_node_{}".format(idx)
                    idx += 1



    #####################
    #    properties     #
    #####################

    def _set_var_prior(self, prior):
        if prior not in ["wishart", "fixed", "gamma", None]:
            raise ValueError("Prior should be one of wishart, fixed, gamma or None")
        self._var_prior = prior

    def _get_var_prior(self):
        return self._var_prior

    def _get_rest_var(self):
        return self._rest_var

    def _set_rest_var(self, value):
        if value < 0:
            raise ValueError("variance must be greater than 0")
        self._rest_var = value
        self.init_rests()

    def _get_root(self):
        return self._root

    def _get_disc(self):
        return self._disc

    def _get_node_ids(self):
        return dict(zip(self.nodes.keys(), range(self.nb + 1)))
    def _set_disc(self, value):
        d, _ = Rest.check_parameters(value, self.conc)
        self._disc = d
        self.init_rests()

    def _get_conc(self):
        return self._conc

    def _set_conc(self, value):
        _, c = Rest.check_parameters(self.disc, value)
        self._conc = c
        self.init_rests()

    def _get_depend(self):
        return self._depend

    def _set_depend(self, value):
        self._depend = self.check_depend(depend=value, conc=self.conc)

    def _get_base(self):
        return self._base

    def _set_base(self, value):
        if value is None:
            self._base = None
        #elif type(value) != Categorical:
        #    raise TypeError("Base measure should be of type Categorical. ")
        else:
            #value.normalize()
            self._base = value
            self.init_rests()

    def _get_total_length(self):
        return self._tl

    def _get_num_branches(self):
        return self._nb

    def _get_branch_lengths(self):
        return self._bls

    def _get_jps(self):
        return self._jps

    def _set_jps(self, value):
        self.root.jps = value
        self._jps = self.root.jps
        self.init_rests()

    def _get_observed(self):
        return self._observed

    def _get_leaves(self):
        return self.root.get_leaves()

    def _get_leaf_names(self):
        return self.root.get_leaf_names()

    def _get_nodes(self):  # todo
        return self.root.nodes

    def _get_nodes_by_id(self):
        return self.root.id_nodes

    def __getitem__(self, item):
        if type(item) == str:
            return self.nodes[item]
        else:
            return self.id_nodes[item]

    @property
    def nleaf(self):
        return self.root.nleaf

    root = property(fget=_get_root)  # the tree (root)
    disc = property(fget=_get_disc, fset=_set_disc)  # the discount parameter
    conc = property(fget=_get_conc, fset=_set_conc)  # the concentration parameter
    rest_var = property(fget = _get_rest_var)
    depend = property(fget=_get_depend, fset=_set_depend)  # whether there is dependency btwn parent & child node
    base = property(fget=_get_base, fset=_set_base)  # the universal base measure
    tl = property(fget=_get_total_length)  # total branch length
    nb = property(fget=_get_num_branches)  # total number of branches
    bls = property(fget=_get_branch_lengths)  # [list] each branch length
    jps = property(fget=_get_jps, fset=_set_jps)  # [list] #jumps on each branch
    observed = property(fget=_get_observed)  # [dict] all observed nodes (name:node)
    leaves = property(fget=_get_leaves)  # all leaves in the tree
    leaf_names = property(fget=_get_leaf_names)  # all leaves in the tree
    nodes = property(fget=_get_nodes)  # [dict] all nodes in the tree
    id_nodes = property(fget=_get_nodes_by_id)  # [dict] all nodes in the tree arranged by its id

    def divs(self, div_name):
        return self.root.divs(div_name)

    #####################
    #     utilities     #
    #####################

    def rescale(self):
        depths = pd.DataFrame(pd.Series(self.get_depth()), columns=['depth'])
        depths['count'] = depths['depth'].map(depths.depth.value_counts())
        depths = depths.drop('')
        for i in range(depths.shape[0]):
            node_name, scale_factor = depths.iloc[i].name, depths.iloc[i]['count']
            node = self.nodes[node_name]
            node.dist = node.dist / scale_factor
            # self._tl, self._nb, self._bls, self._jps = self._root.all_saved_subtree_properties()

    def traverse(self, strategy="preorder"):
        return self.root.traverse(strategy=strategy)

    def write(self, *args, format=1, **kwargs):
        newick = self.root.write(*args, format=format, **kwargs)
        return newick.replace(")1:", "):")

    def get_depth(self, dist=False):
        depth = {}
        for n in self.traverse():
            if n.is_root():
                depth[n.name] = 0
            else:
                depth[n.name] = depth[n.up.name] + (1 if not dist else n.dist)
        return depth

    def get_height(self, dist=False):
        height = {}
        for n in self.traverse(strategy="postorder"):
            # print(n.name)
            if n.is_root():
                break
            if n.is_leaf():
                height[n.name] = (0 if not dist else 0.)

            up_height = height[n.name] + (1 if not dist else n.dist)
            height[n.up.name] = min(up_height, height[n.up.name]) if n.up.name in height else up_height
        return height

    def get_nleaves(self):
        return {n.name: n.nleaf for n in self.traverse()}

    @staticmethod
    def check_depend(depend, conc):
        if type(depend) != bool:
            warnings.warn("depend ({}) is forced to be of type bool. ".format(depend))
            depend = bool(depend)
        if conc > 0 and depend:
            raise ValueError("When conc (={})> 0, depend should be False. ".format(conc))
        return depend

    def check_parameters(self, disc, conc, depend=None):
        disc, conc = Rest.check_parameters(disc, conc)
        depend = self.check_depend(depend, conc)
        return disc, conc, depend

    def set_parameters(self, disc, conc, depend=None):
        if depend is None:
            depend = self.depend
        self._disc, self._conc, self._depend = self.check_parameters(disc, conc, depend)
        self.init_rests()


    def is_dirichlet(self):
        return (not self.depend) and self.disc == 0

    def update_observed(self):
        self._observed = self.root.get_observed()

    def copy_properties(self, other):
        """
        reset properties with another RestFranchise
        """
        self._disc, self._conc = other.disc, other.conc
        self._tl = other.tl
        self._nb = other.nb
        self.leaf_by_nodes = other.leaf_by_nodes
        self.depth_by_nodes = other.depth_by_nodes
        self._bls = list(other.bls)
        self._jps = list(other.jps)
        self._observed = self.root.get_observed()


    def deep_copy(self):
        """
        perform a deep copy of current restaurant franchise
        """
        new = self.__class__(disc=self.disc, conc=self.conc, depend=self.depend, base=self.base)
        new._root = self.root.deep_copy()
        new.copy_properties(self)
        return new


    def get_node_by_name(self, node_name):
        node = None
        for n in self.root.traverse():
            if n.name == node_name:
                node = n
                break
        if node is None:
            raise ValueError("Unable to find a node with node_name {}".format(node_name))
        return node

    def is_offspring(self, parent_name, child_name):
        parent = self.get_node_by_name(parent_name)
        return parent.is_offspring(child_name)

    def __str__(self):
        return self.root.__str__()

    def str_print(self, tree=False, pruned=False, pruned_rests=False,
                  rests=False, bases=False, jps=False, observed=False):
        ret = [self.root.str_print(tree=True)] if tree else []
        if pruned:
            jns = [n.name for n in self.root.traverse() if n.njump > 0]
            jp_tree, ref_name = self.jps_prune()
            ref = ""
            for k in set(ref_name.values()):
                ref_k = [k1 for k1, v1 in ref_name.items() if v1 == k]
                ref += ("\n  {0}: ".format(k) + ", ".join(ref_k))
            ret.append("Jump nodes: " + ",".join(jns) + "\n" +
                       "Equivalent Jump Tree:" + jp_tree.__str__() + "\n" +
                       "Reference observation nodes: " + ref + '\n')
        if pruned_rests:
            jp_tree, _ = self.jps_prune()
            node_names = [n.name for n in jp_tree.root.traverse('preorder')]
            pruned_rests = ""
            for n in self.root.traverse('preorder'):
                if n.name in node_names:
                    pruned_rests += ('\n  {0} rest = '.format(n.name) + n.rest.__str__())
            ret.append("Restaurants (pruned): " + pruned_rests + '\n')
        if rests or bases or jps:
            ret.append(self.root.str_print(rests=rests, bases=bases, jps=jps))
        if observed:
            ret.append("Observed Nodes: " + '[' + ', '.join(self.observed.keys()) + ']')
        return '\n'.join(ret)

    ##################
    #   restaurants  #
    ##################

    def init_rests(self, node=None):
        """
        Initiate restaurants for all nodes
        """
        if node is None:
            node = self.root

        node.init_rests(self.disc, self.conc, self.base, self.rest_var)

    def find_table(self, node_name, obs=None):
        """
        Update all restaurants with the table configuration introduced by an observation
        The table configuration is sampled according to (6.2) in progress report of PhyloTreePY @ 03/28/18
        :param node_name: the name of species that the new observation comes from
        :param obs: the new observation (observed category)
        :return: None
        """
        node = self.observed[node_name]
        return node.find_table(obs, self.depend)

    def rests(self):
        """
        Returns restaurants in the franchise with their associated data
        """
        rests = {}
        for n in self.traverse(strategy='preorder'):
            if n.njump > 0:
                rests[n.name] = {}
                rests[n.name]['data'] = []
                rests[n.name]['clusters'] = {}
                for tbl in n.rest.keys():
                    rests[n.name]['data'] += n.rest[tbl].customers
                    rests[n.name]['clusters'][tbl] = len(n.rest[tbl].customers)

        return rests



    #####################
    #     jump_rate     #
    #####################

    def sample_prior_jump_rate(self, prior_mean_njumps=1.):
        """
        Sample jump_rate from the prior: exp(1/total_length)
        :return: the sample
        """
        return np.random.exponential(prior_mean_njumps / self.tl)

    def sample_post_jump_rate(self, prior_mean_njumps=1., mcmc_log = None):
        """
        Sample jump_rate from the posterior given jumps: Gamma(1 + total_jumps, 1/total_length + #branches)
        :return: the sample
        """
        post_alpha = sum(self.jps) + 1
        post_beta = 1 / 1 / (self.nb + (prior_mean_njumps / self.tl))
        return np.random.gamma(post_alpha, post_beta), post_alpha, post_beta

    def inhomogeneous(self, style: str = "delta time"):
        assert style in ["delta time", "number of leaves"]

        if style == "delta time":
            limits = {self.root.name: 0.}
            for n in self.traverse():
                if n.is_root():
                    continue
                limits[n.name] = limits[n.up.name] + n.dist
            nodes = [n for n in self.traverse()]
            nodes.sort(key=lambda node: limits[node.name])
            nodes = {n: i for i, n in enumerate(nodes)}
            points = np.sort(list(limits.values()))
            mid = ((points[1:] + points[:-1]) / 2).reshape(-1, 1)
            intervals = np.array([[limits[n.up.name], limits[n.name]] for n in self.traverse() if not n.is_root()])
            counts = np.sum((intervals[:, 0] < mid) & (intervals[:, 1] > mid), axis=1)
            lengths = points[1:] - points[:-1]
            for n, i in nodes.items():
                if n.is_root():
                    continue
                i0 = nodes[n.up]
                n.dist = np.sum(lengths[i0:i] / counts[i0:i])

        elif style == "number of leaves":
            for n in self.traverse():
                if n.is_root():
                    continue
                n.dist *= n.nleaf

        # update total branch length
        self._tl = self.root.tl

    #####################
    #  jps management   #
    #####################

    def subtree_has_jump(self, node_name):
        node = self.get_node_by_name(node_name)
        return node.subtree_has_jump()

    def update_jps(self, node=None):
        self._jps = self.root.jps
        self.init_rests(node=node)

    def jps_prune(self, init_rests=True):
        """
        Prune the tree according to its jumps (i.e. remove branches with 0 jump)
        and initialize all the restaurants (with discount (disc), concentration (conc) and base)
        Observed nodes will be preserved and the name_map will be returned
        :return: new: a copy of the original tree with branches pruned,
                 obs_name_ref: a dict that maps observed node names to names after merging {old_name:new_name}
        """
        new = self.deep_copy()

        ref_names = new.root.jps_prune(observed=new.observed)
        new.update_observed()
        if init_rests:
            new.init_rests()

        return new, ref_names

    def partition_by_jps(self):
        ref_names = {n.name: n.name for n in self.traverse()}
        new = self.deep_copy()

        for node in new.traverse():
            if node.njump == 0 and (not node.is_root()):
                parent = node.up
                ref_names[parent.name] = node.name
                parent.name = node.name
                parent.children += node.children
                for child in node.children:
                    child.up = parent
                parent.remove_child(node)
        for k, v, in ref_names.items():
            if k != v:
                idx = [k]
                while v != ref_names[v]:
                    idx.append(v)
                    v = ref_names[v]
                while ref_names[k] != v:
                    ref_names[k] = v

        idx_map = {n.name: i for i, n in enumerate(new.traverse())}
        Z = [idx_map[ref_names[n.name]] for n in self.traverse()]
        del new
        return Z, ref_names

    def jp_affected_leave(self):
        """
        Return the number of leaves affected by each node with a jump
        """
        result = []
        for i, (jps, (node_name, node)) in enumerate(zip(self.jps, self.nodes.items())):
            if jps >= 1 and node_name != '':
                result.append(
                    {
                        'node_idx' : i,
                        'node_name' : node_name,
                        'nleaf' : node.nleaf
                    }
                )

        return pd.DataFrame(result)

    def full_jps(self, njump):
        """
        Assign the same number of jumps to all branches in the tree, except for the root.
        :param njump: number of jumps on each branch
        :return: None
        """
        self.root.full_jps(njump)
        self.update_jps()

    def poisson_jps(self, jump_rate):
        """
        Sample jumps on each branch from the prior: Poisson(jump_rate * branch_length)
        :param jump_rate: the intensity of Poisson
        :return: None
        """
        self.root.poisson_jps(jump_rate)
        self.update_jps()

    def switch_two_jump_node(self, node1, node2, jr = None, return_result = False, debug=True):

        if node1.njump == node2.njump:
            if debug: self._logger.debug("Nodes %s and %s have the same number of jumps, skipping iteration", node1.name,
                               node2.name)
            ids = [node1.id, node2.id]
            log_ratio = 0.
            if_same = True
            if return_result:
                return ids, log_ratio, if_same

        ids = [node1.id, node2.id]
        tmp = node2.njump
        node2.njump = node1.njump
        node1.njump = tmp
        if return_result:
            try:
                log_ratio = (
                                poisson.logpmf(node1.njump, node1.dist * jr) +
                                poisson.logpmf(node2.njump, node2.dist * jr) -
                                poisson.logpmf(node2.njump, node1.dist * jr) -
                                poisson.logpmf(node1.njump, node2.dist * jr)
                        )

                return ids,log_ratio,False
            except ValueError:
                print("Jump rate must be specified")
                raise


        return

    def switch_jumps(self, how, jr = None, return_result = False, max_nleaf = np.inf, max_depth = np.inf,
                     min_depth = -np.inf, min_nleaf = -np.inf
                     ):
        methods = {'parent_child', 'sibling', 'same_nleaf', 'same_depth'}
        # default return
        ids, log_ratio, if_same = [0,0], 0., True
        if how not in methods:
            raise ValueError(f"Invalid method for swapping, available methods are {methods}")

        # validate filters for proposal types
        if (how == 'parent_child' or how == 'sibling') and (min_nleaf != -np.inf or max_nleaf != np.inf):
            raise ValueError(f"Leaf filters should not be set for {how} proposal to ensure reversibility.")



        if sum(self.jps) <= self.root.ROOT_JUMP:
            self._logger.debug(f"There are no jumps on the tree, skipping swap proposal with method {how}")
            if return_result: return ids, log_ratio, if_same


        jump_nodes = [n for n in self.nodes.values() if (n.njump > 0)
                        and (not n.is_root())
                        and (n.nleaf <= max_nleaf) and (n.nleaf >= min_nleaf)
                        and (n.depth <= max_depth) and (n.nleaf >= min_depth)
                      ]
        n_jump_nodes = len(jump_nodes)
        if n_jump_nodes == 0:
            self._logger.debug("No valid jump nodes, skipping")
            if return_result:
                return ids, log_ratio, if_same
            return
        self._logger.debug("Jump nodes being considered: %d", n_jump_nodes)
        jn = jump_nodes[np.random.randint(n_jump_nodes)]  # pick a jump node
        n = 0
        # now choose candidates
        if how == 'parent_child':
            pjn = jn.up
            candidates = [] if pjn.is_root() or pjn.depth < min_depth else [pjn]
            if jn.depth + 1 <= max_depth:
                candidates += jn.children
            n = len(candidates)

        elif how == 'sibling':
            pjn = jn.up
            candidates = [c for c in pjn.children if c.name != jn.name]
            n = len(candidates)
            self._logger.debug("Chose node %s. There are %d candidates nodes for swapping.",
                               jn.name, n
                               )
        elif how == 'same_nleaf':
            descendants = jn.nleaf
            candidates = [c for c in self.leaf_by_nodes[descendants] if c != jn
                          and (c.nleaf <= max_nleaf) and (c.nleaf >= min_nleaf)
                          and (c.depth <= max_depth) and (c.nleaf >= min_depth)
                          ]
            n = len(candidates)
            self._logger.debug("Chose node %s, which has %d descendants. There are %d candidates nodes for swapping.",
                               jn.name, descendants, n
                               )
        elif how == 'same_depth':
            depth = jn.depth
            candidates = [c for c in self.depth_by_nodes[depth] if c != jn
                          and (c.nleaf <= max_nleaf) and (c.nleaf >= min_nleaf)
                          and (c.depth <= max_depth) and (c.nleaf >= min_depth)
                          ]
            n = len(candidates)
            self._logger.debug("Chose node %s, which has depth %d. There are %d candidates nodes for swapping.",
                               jn.name, depth, n
                               )


        if n > 0:
            jn1 = candidates[np.random.randint(n)]
            self._logger.debug("Swapping with node: %s", jn1.name)
            result = self.switch_two_jump_node(jn,jn1,jr=jr, return_result=return_result)
            return result
        else:
            if return_result:
                return ids, log_ratio, if_same

    def new_jump_onOff(self, jr, uniform_branch = True, return_result = False):
        if uniform_branch:
            idx = np.random.randint(self.nb) + 1  # pick a branch to change (the root 0 will not be picked)
        else:
            # probabilities are softmax of branch lengths
            nleaves = list(self.get_nleaves().values())
            probs = sp.special.softmax(np.array(nleaves[1:]))
            idx = np.random.choice(self.nb, p=probs)
        jn = list(self.nodes.values())[idx]
        njump = jn.njump
        if njump == 0:
            jn.njump = 1
            log_ratio = np.log(jr * jn.dist)
        else:
            jn.njump = 0
            log_ratio = np.log(jr * jn.dist) * (-njump) - np.log(njump)
        if_same = False
        return [jn.id, 0], log_ratio, if_same

    def new_jump_birthDeath(self, jr, uniform_branch = True, return_result = False):
        if uniform_branch:
            idx = np.random.randint(self.nb) + 1  # pick a branch to change (the root 0 will not be picked)
        else:
            # probabilities are softmax of branch lengths
            nleaves = list(self.get_nleaves().values())
            probs = sp.special.softmax(np.array(nleaves[1:]))
            idx = np.random.choice(self.nb, p=probs)
        jn = list(self.nodes.values())[idx]
        self._logger.debug("Picked branch %s", jn.name)
        njump = jn.njump
        if njump == 0:
            self._logger.debug("Increasing number of jumps from 0 to 1")
            jn.njump = 1
            log_ratio = np.log(jr * jn.dist) - np.log(2.)
        else:
            jn.njump = njump + np.random.choice([-1, 1])
            self._logger.debug("Changed number of jumps from %d to %d", njump, jn.njump)
            log_ratio = np.log(jr * jn.dist) * (jn.njump - njump) - np.log(max(njump, jn.njump))
            if jn.njump == 0:
                log_ratio += np.log(2.)
        if_same = False
        if return_result: return [jn.id, 0], log_ratio, if_same

    def permute_all_jps(self, jr, max_nleaf = np.inf, max_depth = np.inf,
                     min_depth = -np.inf, min_nleaf = -np.inf, return_result = False):
        """
        permute all jumps, returning log ratio of result
        """

        # take all nodes except for the root
        nodes = np.array(list(self.nodes.values()))[1:]

        # these operations are slow, may be worth caching
        mask = [
            (n.nleaf <= max_nleaf)  and (n.nleaf >= min_nleaf)
            and (n.depth <= max_depth) and (n.depth >= min_depth) for n in nodes
        ]
        nodes_to_permute = nodes[mask]
        permutation = np.random.permutation(nodes)
        ids, log_ratio, if_same = [0,0], 0, True
        for n1, n2 in zip(nodes_to_permute, permutation):
            result = self.switch_two_jump_node(n1, n2, jr=jr, return_result=return_result, debug=False)
            if return_result:
                _, log_ratio_it, if_same_it = result
                log_ratio += log_ratio_it
                if_same *= if_same_it

        return ids, log_ratio, if_same





    ########################
    #  variance updating   #
    ########################

    def sample_post_var(self, data, alpha, beta):
        # get the observations belonging to each cluster of the tree formed by the jump configuration
        partitions = self.partition_by_jps()[0]
        partitions = pd.DataFrame(data=partitions, index=list(self.nodes.keys())).rename(columns={0: 'cluster'})
        partitions = data.join(partitions, on = 'node_name').groupby('cluster')
        variances = partitions['obs'].var().fillna(0)


        # calculate the sum of squared deviations from cluster means for each cluster
        # do this by calculating the variance for each cluster, then multiplying by n-1
        sse = np.dot(variances, partitions.size() - 1)
        n = data.shape[0]
        # return a sample
        post_alpha = alpha + n / 2.
        post_beta = beta + sse / 2.

        return invgamma.rvs(a = post_alpha, scale = post_beta, size = 1).item(),post_alpha, post_beta

    def particleMCMC (self, data: pd.DataFrame,
                       proposal: Proposal,
                       num_particles: int = 5,
                       seed: int = None, n_iter: int = 2000, return_particles: bool = False, return_rests: bool = False,
                       fix_jump_rate: bool = False, prior_mean_njumps: float = 1., init_jp: list [ int ] = None,
                       init_jr: float = None, init_log_lik: float = None,
                       conc: float = None, disc: float = None,
                       var_prior: str = None, var_prior_params: list [ float ] = None, plot_dir='',
                       progress_bar=True):
        """
        Generate posterior samples of (jump_rate, jps) with Particle MCMC algorithm,
        where jps denotes jumps on branches.
        :param data: the pandas.DataFrame data
        :param num_particles: number of particles for the particle filtering step
        :param n_iter: number of MCMC iterations
        :param return_particles: whether posterior samples of the particles will
                                 also be returned
        :param return_rests: whether rests of particles will be returned
        :param fix_jump_rate: whether the jump rate is fixed to "init_jr"
        :param init_jr: initial jump rate, will be a sample from the prior if None
        :param init_log_lik: initial log likelihood (in order to connect with previous
                             runs of samples.
        :param prior_mean_njumps: prior mean number of jumps
        :param progress_bar: whether a progress bar will be printed

        :return: info, posterior samples of jump_rate, jps, and particles (None
                 when not return_particles)
        """
        # log = get_log(log_file, out_log, "particleMCMC.txt")
        # if out_log:
        #     log.write("")

        # log of MCMC iterations
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        log = {"acc_rate": 0,
               'accepted': [False] * n_iter,
               'log_lik': [0.] * n_iter,
               'log_acc': [0.] * n_iter,
               'log_acc_prob': [0.] *n_iter,
               'log_lik_pre' : [0.] * n_iter,
               'jp_log_ratio' : [0.] * n_iter,
               'proposed': sps.lil_matrix((n_iter, self.nb + 1), dtype=int),
               "same_proposal_rate": 0.,
               'same_proposal': [False] * n_iter,
               }

        # check plotting
        make_plots = (plot_dir != '')
        if make_plots:
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

        # containers of posterior samples
        post_jrs = np.zeros(n_iter)  # jump rates
        post_jps = sps.lil_matrix((n_iter, self.nb + 1), dtype=int)
        post_Zs = sps.lil_matrix((n_iter, self.nb + 1), dtype=int)  # partition of the tree
        post_particles = [None] * n_iter if return_particles else None
        # check prior
        if var_prior not in ["wishart", "fixed", "gamma", None]:
            raise ValueError("a prior for kernel variance should be specified and one of wishart, fixed (constant) or gamma")
        if var_prior is None:
            var_prior = self._var_prior
        if var_prior_params is None:
            var_prior_params = self._prior_params
        params_to_set = {}
        if conc is None:
            if self.conc is None:
                raise ValueError("Concentration parameter must be specified")
            else:
                params_to_set['conc'] = self.conc
        else:
            params_to_set['conc'] = conc
        if disc is None:
            if self.disc is None:
                raise ValueError("Discount parameter must be specified")
            else:
                params_to_set['disc'] = self.disc
        else:
            params_to_set['disc'] = disc

        self.set_parameters(params_to_set['disc'], params_to_set['conc'], depend = True)

        if var_prior is None:
            raise ValueError("a prior for kernel variance should be specified and one of wishart, fixed (constant) or gamma")

        if var_prior == "gamma" or var_prior == "fixed":
            mean = var_prior_params[0]
            sd = var_prior_params[1]
            base = norm(mean, sd)

            if var_prior == "gamma":
                post_vars = np.zeros(n_iter)
                alpha = var_prior_params[2]
                beta = var_prior_params[3]
                kernel_var = invgamma.rvs(a = alpha, scale = 1 / beta, size = 1).item()
            else:
                kernel_var = var_prior_params[2]


        elif var_prior == "wishart":
            base = norminvgamma(var_prior_params[0], var_prior_params[1],
                                var_prior_params[2], var_prior_params[3])
            kernel_var = None

        # initialize jump rate and jps
        if init_jp is not None:
            self.jps = init_jp
            jr = np.sum(init_jp) / self.tl
        else:
            jr = prior_mean_njumps / self.tl if init_jr is None else init_jr
            self._logger.debug("Creating initial jump configuration.")
            self.poisson_jps(jr)
            self._logger.debug("%d initial jumps", sum(self.jps))
            if make_plots: self.plot_with_jp(file = os.path.join(plot_dir, "0.png"))

        # fake records for previous iteration (iteration -1)
        jps_pre, p_pre, Zs_pre = self.jps, None, self.partition_by_jps()[0]
        log_lik_pre = -np.Inf if init_log_lik is None else init_log_lik

        # Start the iteration
        self._logger.info("Generating posterior samples...")
        p = log_lik = log_acc = None

        for it in tqdm(range(n_iter), disable=(not progress_bar)):
            # new jump rate (if not fixed)
            self._logger.debug("Iteration %d ---------------------------------------------------------------------", it)
            self._logger.debug("Current number of jumps: %d", sum(self.jps))
            self._logger.debug("Performing Gibbs updates.")
            jr, jr_post_alpha, jr_post_beta = self.sample_post_jump_rate(prior_mean_njumps) if not fix_jump_rate else (jr,None,None)
            post_jrs[it] = jr
            self._logger.debug("New jump rate: %.5f", jr)
            if var_prior == "gamma":
                kernel_var, var_post_alpha, var_post_beta = self.sample_post_var(data, alpha, beta)
                post_vars[it] = kernel_var
                self._logger.debug("New kernel variance: %.5f", kernel_var)

            ids, log_ratio, if_same = proposal.propose_jps(self, jr)


            # Acceptance
            if if_same:  # if proposed jps is the same with the previous sample
                self._logger.debug("Proposed exact same jump configuration, skipping acceptance probability calculation.")
                log['same_proposal'][it] = True
                accepted = True
                log['accepted'][it] = accepted
                log_acc_prob = 0
                log_lik = log_lik_pre
                jp_log_ratio = 0
                log_acc = 0


            else:  # particle filter estimation of data log-likelihood
                ps = Particles(tree=self, num_particles=num_particles, forward=True,
                               var_prior = var_prior, kernel_var = kernel_var, base = base)
                log_lik = ps.particle_filter_integrated(data)  # out_log=out_log, log=log


                p = ps.get_particle() if (return_particles or return_rests) else None

                log_acc = log_lik - log_lik_pre + log_ratio
                log_acc_prob = np.log(np.random.rand(1))[0]
                accepted = (log_acc > log_acc_prob)
                self._logger.debug("Jump / proposal log ratio: %.5f", log_ratio)
                if it != 0: self._logger.debug("Previous loglik:  %.5f", log_lik_pre)
                self._logger.debug("New log_lik: %.5f", log_lik)
                if it != 0: self._logger.debug("Log acceptance:  %.5f", log_acc)
                self._logger.debug("Log acceptance probability: %.5f", log_acc_prob)
                self._logger.debug("Accepted: %s", accepted)

            log['log_acc_prob'][it] = log_acc_prob
            log['accepted'][it] = accepted
            log["acc_rate"] += accepted
            log['log_lik'][it] = log_lik
            log['log_lik_pre'][it] = log_lik_pre
            log['jp_log_ratio'][it] = log_ratio
            log['log_acc'][it] = log_acc
            log['proposed'][it, :] = self.jps

            # update and record samples
            if accepted:
                jps_pre = self.jps
                Zs_pre = self.partition_by_jps()[0]
                p_pre = p
                log_lik_pre = log_lik
            else:
                self.jps = jps_pre  # rewind jps
            post_jps[it, :] = jps_pre
            post_Zs[it, :] = Zs_pre
            if return_particles:
                post_particles[it] = p_pre

        if make_plots:
            for i, proposed in enumerate(log['proposed']):
                accepted = post_jps.getrow(i).toarray().flatten().tolist()
                self.plot_with_jp(jp=list(accepted), file=os.path.join(plot_dir, f"{i}_accepted.png"))
                self.plot_with_jp(jp=proposed, file=os.path.join(plot_dir, f"{i}_proposed.png"))
        # clean and process info
        log["acc_rate"] /= n_iter
        log['same_proposal_rate'] = np.mean(log['same_proposal'])
        # log["last_jr"] = jr  # last jump_rate
        # log["last_jps"] = self.jps  # last jps
        # log["last_log_lik"] = log_lik_pre  # last log_lik
        self.jps = {'Root': 1}

        if progress_bar:
            print("DONE!")

        if var_prior != "gamma":
            return {"log": log,
                    "jump_rate": post_jrs,
                    "jumps": post_jps,
                    "partitions": post_Zs,
                    "particles": post_particles,
                    }
        else:
            return {"log": log,
                    "jump_rate": post_jrs,
                    "jumps": post_jps,
                    "vars" : post_vars,
                    "partitions": post_Zs,
                    "particles": post_particles,
                    }

    ##################
    # synthetic data #
    ##################
    def simulate_jps_byPrior(self, jump_rate=None, njumps=None, min_affected_leaves=0, max_affected_leaves=None,
                             not_on_same_branch=True):
        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        # simulate jumps
        if jump_rate is not None:
            assert njumps is None, "Cannot set jump_rate and njumps simultaneously."
            self.jps = {}
            self.poisson_jps(jump_rate=jump_rate)
            for n in self.traverse():
                if n.nleaf <= min_affected_leaves or n.nleaf >= max_affected_leaves:
                    n.njump = 0
        elif njumps is not None:
            self.jps = {}
            if njumps > 0:
                probs, nodes = [], []
                for n in self.traverse():
                    if not n.is_root() and max_affected_leaves > n.nleaf > min_affected_leaves:
                        probs.append(n.dist)
                        nodes.append(n)
                jump_nodes = np.random.choice(nodes, size=njumps, p=np.array(probs)/sum(probs),
                                              replace=(not not_on_same_branch))
                for n in jump_nodes:
                    n.njump = 1
                self.update_jps()

    def _get_subtree_nodenames(self, node):
        if type(node) != RestNode:
            node = self[node]
        return list(node.nodes.keys())

    def simulate_jps_1equally(self, min_affected_leaves=0, max_affected_leaves=None,
                              not_in_subtree=None, in_subtree=None, CLEAR_OLD_JUMPS=True) -> RestNode:
        invalid = [] if not_in_subtree is None else self._get_subtree_nodenames(not_in_subtree)
        valid = self._get_subtree_nodenames(self.root if in_subtree is None else in_subtree)

        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf

        nodes = []
        for n in self.traverse():
            if not n.is_root() and CLEAR_OLD_JUMPS:
                n.njump = 0
            if (n.name not in invalid) and (n.name in valid) and max_affected_leaves > n.nleaf > min_affected_leaves:
                nodes.append(n)
        assert len(nodes) > 0, "Unable to find appropriate nodes."
        node = np.random.choice(nodes)
        node.njump = 1
        self.update_jps()
        return node

    def simulate_jps_1byNleaf(self, nleaf, tolerance=None, min_affected_leaves=0, max_affected_leaves=None,
                              not_in_subtree=None, in_subtree=None, CLEAR_OLD_JUMPS=True) -> RestNode:
        invalid = [] if not_in_subtree is None else self._get_subtree_nodenames(not_in_subtree)
        valid = self._get_subtree_nodenames(self.root if in_subtree is None else in_subtree)

        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        probs, nodes = [], []
        for n in self.traverse():
            if not n.is_root() and CLEAR_OLD_JUMPS:
                n.njump = 0
            if (n.name not in invalid) and (n.name in valid) and max_affected_leaves > n.nleaf > min_affected_leaves:
                if tolerance is None or abs(n.nleaf - nleaf) < tolerance:
                    nodes.append(n)
                    probs.append(1./((n.nleaf - nleaf + 1) ** 2))

        assert len(nodes) > 0, "Unable to find appropriate nodes."

        node = np.random.choice(nodes, p=np.array(probs)/sum(probs))
        node.njump = 1
        self.update_jps()
        return node

    # =================

    def simulate_data_byPrior(self, each_size: int = 1):
        assert each_size > 0
        new = self.deep_copy()
        assert new.base is not None, "Base measure is not provided."

        # simulate data
        data = []
        for n in new.observed.values():
            for _ in range(each_size):
                obs = n.find_table(depend=self.depend)
                data.append({"node_name": n.name, "obs": obs})
                #print(data)
        data = pd.DataFrame(data)
        del new
        return data

    # =================

    def simulate_prior(self, jump_rate=None, njumps=None, min_affected_leaves=0, max_affected_leaves=None,
                       not_on_same_branch=True, each_size: int = 1):
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf

        self.simulate_jps_byPrior(jump_rate, njumps, min_affected_leaves, max_affected_leaves, not_on_same_branch)
        return self.simulate_data_byPrior(each_size)

    def simulate_mixture(self, jump_rate=None, njumps=None, min_affected_leaves=0, max_affected_leaves=None,
                       not_on_same_branch=True, each_size: int = 1):
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf

        self.simulate_jps_byPrior(jump_rate, njumps, min_affected_leaves, max_affected_leaves, not_on_same_branch)
        return self.simulate_mixture_obs(each_size)

    def simulate_mixture_obs(self, each_size: int = 1):
        """
        Simulate mixture model data at the observed nodes
        """
        assert each_size > 0
        new = self.deep_copy()
        assert new.base is not None, "Base measure is not provided"

        # contains node, true cluster means and observations assigned to each cluster mean
        rest_data = []
        # contains just nodes and their observations
        data = []
        for n in new.observed.values():
            # store each row with node name, with other keys being the cluster means and values being arrays
            # of observations
            rest_data.append({"node_name":n.name})
            for _ in range(each_size):
                obs, k = n.seat_obs(depend=self.depend)
                if k in rest_data[-1].keys():
                    rest_data[-1][k].append(obs)
                else:
                    rest_data[-1][k] = [obs]

                data.append({"node_name":n.name, "obs": obs})


        return pd.DataFrame(data), rest_data


    def simulate_one_jump(self, min_affected_leaves=0, max_affected_leaves=None, affected_leaves=None, tolerance=None,
                          total_variation=0.,  each_size: int = 1):
        assert each_size > 0

        # locate jump
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        if max_affected_leaves is not None and 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if affected_leaves is not None:
            if 0 < affected_leaves < 1:
                affected_leaves *= self.nleaf
            if tolerance is not None and 0 < tolerance < 1:
                tolerance *= self.nleaf
            jn = self.simulate_jps_1byNleaf(
                nleaf=affected_leaves, tolerance=tolerance,
                min_affected_leaves=min_affected_leaves, max_affected_leaves=max_affected_leaves)
        else:
            jn = self.simulate_jps_1equally(
                min_affected_leaves=min_affected_leaves, max_affected_leaves=max_affected_leaves)

        rest_data = []
        data = []
        for n in self.observed.values():
            rest_data.append({"node_name": n.name})
            for _ in range(each_size):
                obs, k = n.seat_obs(depend=self.depend)
                if k in rest_data[-1].keys():
                    rest_data[-1][k].append(obs)
                else:
                    rest_data[-1][k] = [obs]

                data.append({"node_name": n.name, "obs": obs})

        return pd.DataFrame(data), rest_data

    def simulate_two_jumps(self, min_affected_leaves=None, mid_affected_leaves=None, max_affected_leaves=None,
                           single_jump_total_variation=0., each_size: int = 1):
        """
        simulate two jumps, one in the subtree of another, each creates
        single_jump_total_variation amount of difference in distributions before and after it
        The two jumps should be of the same direction
        """
        assert each_size > 0
        assert -0.5 <= single_jump_total_variation <= 0.5

        if min_affected_leaves is None:
            min_affected_leaves = 0
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        if mid_affected_leaves is None:
            mid_affected_leaves = 0.5
        if 0 < mid_affected_leaves < 1:
            mid_affected_leaves *= self.nleaf
        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf

        n1 = self.simulate_jps_1byNleaf(nleaf=(mid_affected_leaves + max_affected_leaves) / 2,
                                        min_affected_leaves=mid_affected_leaves,
                                        max_affected_leaves=max_affected_leaves)
        n2 = self.simulate_jps_1byNleaf(nleaf=(min_affected_leaves + mid_affected_leaves) / 2,
                                        min_affected_leaves=min_affected_leaves,
                                        in_subtree=n1, CLEAR_OLD_JUMPS=False)

        rest_data = []
        data = []
        for n in self.observed.values():
            rest_data.append({"node_name": n.name})
            for _ in range(each_size):
                obs, k = n.seat_obs(depend=self.depend)
                if k in rest_data[-1].keys():
                    rest_data[-1][k].append(obs)
                else:
                    rest_data[-1][k] = [obs]

                data.append({"node_name": n.name, "obs": obs})

        return pd.DataFrame(data), rest_data

    # figure out how to do total variation?
    def simulate_three_jumps(self, min_affected_leaves=None, max_affected_leaves=None, affected=None,
                             single_jump_total_variation=0., each_size: int = 1):
        """
        simulate two jumps, one in the subtree of another, each creates
        single_jump_total_variation amount of difference in distributions before and after it
        """
        assert each_size > 0
        assert -1/3 <= single_jump_total_variation <= 1/3
        assert affected is None or len(affected) == 3

        if min_affected_leaves is None:
            min_affected_leaves = 0
        if 0 < min_affected_leaves < 1:
            min_affected_leaves *= self.nleaf
        if max_affected_leaves is None:
            max_affected_leaves = self.nleaf
        if 0 < max_affected_leaves < 1:
            max_affected_leaves *= self.nleaf
        if affected is None:
            affected = max_affected_leaves - (np.arange(3) + 1) * (max_affected_leaves - min_affected_leaves) / 4

        n1 = self.simulate_jps_1byNleaf(nleaf=affected[0],
                                        min_affected_leaves=min_affected_leaves,
                                        max_affected_leaves=max_affected_leaves)
        n2 = self.simulate_jps_1byNleaf(nleaf=affected[1],
                                        min_affected_leaves=min_affected_leaves,
                                        in_subtree=n1, CLEAR_OLD_JUMPS=False)
        n3 = self.simulate_jps_1byNleaf(nleaf=affected[2],
                                        min_affected_leaves=min_affected_leaves,
                                        in_subtree=n2, CLEAR_OLD_JUMPS=False)

        rest_data = []
        data = []
        for n in self.observed.values():
            rest_data.append({"node_name": n.name})
            for _ in range(each_size):
                obs, k = n.seat_obs(depend=self.depend)
                if k in rest_data[-1].keys():
                    rest_data[-1][k].append(obs)
                else:
                    rest_data[-1][k] = [obs]

                data.append({"node_name": n.name, "obs": obs})

        return pd.DataFrame(data), rest_data

    def plot_annotated_tree(self, file=None, title=None, title_fsize=20,
                       val=None, std=None, L1=None, pvalue=None, value_name=None, int_value=False,
                       data=None, K=None, colors=None, data_color_leaf=False,
                       criteria=None, vs_node=None, direction='less', metric='p-value',
                       show_branch_length=False, show_jumps=True, show_jumps_background=False,
                       show_node_name=True, show_leaf_name=True,
                       mark_branch=None, mark_branches=None, post_probs=None, post_prob_cmap='autumn',
                       special_node_size=5, special_hzline_width=3, line_width=1,
                       jump_bg_color='lightyellow', jump_bg_cmap='autumn', differentiate_jump_bg=False,
                       scale_length=1, circular=False, dpi=200, height=500, width=500, units='px'):



        tr = self.deep_copy()
        ts = TreeStyle()
        ts.show_leaf_name = show_leaf_name
        if scale_length is not None:
            ts.scale = 50 / scale_length
        if circular:
            ts.mode = 'c'

        data_column = (not data_color_leaf) and (show_leaf_name or show_node_name or data is None or
                                                 data.shape[0] > len(self.leaves))
        if data is not None:
            if colors is None:
                K = len(np.unique(data.obs)) if K is None else K
                colors = rgb_scale(K, gray_scale=(K <= 2))

            if data_column:
                for leaf in tr.root.iter_leaves():
                    obs = data.obs[data.node_name == leaf.name].sort_values()
                    d = TextFace(" " * 1)
                    d.background.color = to_hex("white")
                    leaf.add_face(d, column=0, position="aligned")
                    for i, label in enumerate(obs):
                        d = TextFace(str(label))
                        d.background.color = to_hex(colors[i])
                        leaf.add_face(d, column=i + 1, position="aligned")

        if val is None and pvalue is None:
            metric = None

        if vs_node is not None:
            if type(vs_node) == str:
                i = tr[vs_node].id
            elif type(vs_node) == int:
                i = vs_node
            else:
                i = vs_node.id
            criteria = pvalue[i - 1] if metric == "p-value" else val[i - 1]

        if direction == "less" or direction == "l":
            if_less = True
        elif direction == "greater" or direction == "g":
            if_less = False
        else:
            raise ValueError("direction not recognized. Should be \"less\" or \"greater\". ")

        jump_counter = 0
        if differentiate_jump_bg:
            show_jumps_background = True
            jump_bg_color = sum(int(njump > 0) for njump in self.jps) - 1
        show_jumps = show_jumps if not show_jumps_background else True
        if show_jumps_background and type(jump_bg_color) == int and jump_bg_color > 0:
            jump_bg_color = [to_hex(c) for c in rgb_scale(jump_bg_color, CMAP=jump_bg_cmap, light_bg=True)]

        if post_probs is not None:
            branch_cmap = plt.get_cmap(post_prob_cmap)

        for i, node in enumerate(tr.traverse()):

            nstyle = NodeStyle()
            nstyle['size'] = 1
            nstyle["vt_line_width"] = line_width
            nstyle["hz_line_width"] = line_width

            if node.is_root():
                node.set_style(nstyle)
                continue

            if mark_branch is not None and (node.id == mark_branch or node.name == mark_branch):
                nstyle['size'] = 5
                nstyle['fgcolor'] = "red"
            if mark_branches is not None and ((node.id in mark_branches) or (node.name in mark_branches)):
                nstyle['size'] = 5
                nstyle['fgcolor'] = "red"

            if show_branch_length:
                node.add_face(TextFace(" length = {:.03f}".format(node.dist)),
                              column=0, position="branch-bottom")

            if post_probs is not None:
                b_col = to_hex(branch_cmap(post_probs[i]))
                nstyle["vt_line_color"] = b_col
                nstyle["hz_line_color"] = b_col

            if L1 is not None:
                node.add_face(TextFace(" L1(data) = {:.03f}".format(L1[i - 1])),
                              column=0, position="branch-bottom")

            if show_node_name or val is not None:
                msg = node.name if not node.is_leaf() else ""
                if val is not None:
                    if not node.is_leaf():
                        msg += ": "
                    msg += (((value_name + " = ") if value_name is not None else "") +
                            ("{:d}".format(val[i - 1]) if int_value else "{:.3f}".format(val[i - 1])) +
                            (" (±{:.3f})".format(std[i - 1]) if std is not None else ""))
                node.add_face(TextFace(msg), column=0, position="branch-top")

            if pvalue is not None:
                P = TextFace(" p-value = {:.04f}".format(pvalue[i - 1]))
                node.add_face(P, column=0, position="branch-bottom")

            if metric is not None:
                v = pvalue[i - 1] if metric == "p-value" else val[i - 1]
                signif = True if criteria is None else ((v <= criteria) if if_less else (v >= criteria))
                nstyle['hz_line_color'] = signif_rgb(v, if_less=if_less, criteria=criteria) if signif else to_hex(
                    "black")
                nstyle['hz_line_width'] = special_hzline_width if signif else 1

            if node.njump > 0 and show_jumps:
                if show_jumps_background:
                    if type(jump_bg_color) == str:
                        col = jump_bg_color
                    else:
                        if differentiate_jump_bg:
                            col = jump_bg_color[jump_counter]
                            jump_counter += 1
                        else:
                            col = jump_bg_color[node.njump - 1]
                    nstyle['bgcolor'] = col
                else:
                    nstyle['fgcolor'] = "goldenrod"
                    nstyle['size'] = special_node_size

            if data is not None and data_color_leaf and node.is_leaf():
                obs = data.obs[data.node_name == node.name]
                if len(obs) == 1:
                    c = colors[int(obs)]
                    nstyle['fgcolor'] = to_hex(c)
                    nstyle["hz_line_color"] = to_hex(c)
                    nstyle['size'] = special_node_size
                    nstyle['hz_line_width'] = special_hzline_width
                    # print(node.name, c)

            node.set_style(nstyle)

        if title is not None:
            ts.title.add_face(TextFace(title, fsize=title_fsize), column=0)

        if file is None:
            tr.root.show(tree_style=ts)
        else:
            tr.root.render(file, tree_style=ts, dpi=dpi, h=height, w=width, units=units)

    def plot_with_jp(self, jp = None, file = None, title = None, dpi = 500, height=400, width=400, units='px'):
        if jp is not None:
            tree = self.deep_copy()
            tree.jps = jp
        else:
            tree = self
        tree.plot_annotated_tree(circular=True, show_jumps_background=True, show_leaf_name=True, height=height,
                                 width = width,
                                 differentiate_jump_bg=True,
                                 title=title,
                                 file=file,
                                 dpi=dpi)








