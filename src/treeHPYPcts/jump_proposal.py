"""
Implements a variety of methods for proposing jumps for use in particle MCMC.

@author: Steven Nguyen
"""

import logging
import itertools
import numpy as np
from typing import List
from .rest_franchise import RestFranchise
from .rest_node import RestNode
from abc import ABC, abstractmethod
from scipy.stats import poisson

class Proposal(ABC):

    @abstractmethod
    def propose_jps(self, tree: RestFranchise, jr : float):
        pass


class SwapProposal(Proposal):

    _SWAP_METHODS = {'parent_child', 'sibling', 'same_nleaf', 'same_depth'}

    def __init__(self, method, name = None, max_depth=np.inf, max_nleaf=np.inf, min_depth=-np.inf, min_nleaf=-np.inf):
        if method not in self._SWAP_METHODS:
            raise ValueError(f"Invalid proposal type. Possible swaps are {self._SWAP_METHODS}")
        self.method = method
        self.max_depth = max_depth
        self.max_nleaf = max_nleaf
        self.min_nleaf = min_nleaf
        self.min_depth = min_depth
        self._logger = logging.getLogger(__name__)
        if name is not None:
            self.name = name
        else:
            self.name = f"{method}_swap"


    def propose_jps(self, tree, jr):
        self._logger.debug("Proposing new jumps using proposal %s", self.name)
        ids, log_ratio, if_same = tree.switch_jumps(how=self.method, jr=jr, return_result=True,
                                                    max_depth=self.max_depth,
                                                    max_nleaf=self.max_nleaf,
                                                    min_nleaf=self.min_nleaf,
                                                    min_depth=self.min_depth,
                                                    )
        if not if_same:
            tree.update_jps()



        return ids, log_ratio, if_same

class PermutationProposal(Proposal):

    def __init__(self, name='permutation', max_depth=np.inf, max_nleaf=np.inf, min_depth=-np.inf, min_nleaf=-np.inf):
        self.max_depth = max_depth
        self.max_nleaf = max_nleaf
        self.min_nleaf = min_nleaf
        self.min_depth = min_depth
        self._logger = logging.getLogger(__name__)
        self.name = name


    def propose_jps(self, tree, jr):
        self._logger.debug("Proposing new jumps using proposal %s", self.name)
        ids, log_ratio, if_same = tree.permute_all_jps(jr, max_depth=self.max_depth,
                                                       max_nleaf = self.max_nleaf,
                                                       min_nleaf = self.min_nleaf,
                                                       min_depth = self.min_depth,
                                                       return_result= True
                                                       )
        if not if_same:
            tree.update_jps()



        return ids, log_ratio, if_same


class RandomWalkProposal(Proposal):

    """
    to do - specifying probability distribution branch. for now just let the method in RestFranchise implement it
    """

    def __init__(self, walk = 'bdp', name = None, uniform_branch = True):
        self.walk = walk
        self.uniform_branch = uniform_branch
        if name is not None:
            self.name = name
        else:
            self.name = walk
        self._logger = logging.getLogger(__name__)

    def propose_jps(self, tree, jr):
        self._logger.debug("Proposing new jumps using proposal %s", self.name)
        if self.walk == 'bdp':
            ids, log_ratio, if_same = tree.new_jump_birthDeath(jr, self.uniform_branch, return_result=True)
        elif self.walk == 'on_off':
            ids, log_ratio, if_same = tree.new_jump_onOff(jr, self.uniform_branch, return_result = True)
        if not if_same:
            tree.update_jps()



        return ids, log_ratio, if_same

class MultiProposal(Proposal):

    _STRATEGIES = {'cycle', 'random'}

    def __init__(self, proposals : List[Proposal] = None, strategy:str ='cycle'):

        self.proposals = proposals
        if self.proposals is None:
            self.proposals = []

        self.strategy = strategy
        if self.strategy == 'cycle':
            self.cycle_it = 0

        self.n_proposals = len(self.proposals)

    def propose_jps(self, tree, jr):
        if self.strategy == 'cycle':
            proposal = self.proposals[self.cycle_it]
            self.cycle_it = (self.cycle_it + 1) % self.n_proposals
        elif self.strategy == 'random':
            proposal = self.proposals[np.random.rand(self.n_proposals)]

        return proposal.propose_jps(tree, jr)

    def add_proposals(self, *args):
        for arg in args:
            if isinstance(arg, Proposal):
                self.proposals.append(arg)
                self.n_proposals += 1
            else:
                raise ValueError("Expected proposal object")






