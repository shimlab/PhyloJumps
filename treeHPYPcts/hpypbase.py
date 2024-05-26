"""
HPYP base distribution class. Intended for use only in rest.py

@author: Steven Nguyen
"""
from .categorical import Categorical
from scipy.stats import rv_continuous
from .norminvgamma import norminvgamma
from typing import Union
from scipy.stats._distn_infrastructure import rv_continuous_frozen

class HpypBase:
    pass

class HpypBase:

    NEW_TABLE = '__NEW_TABLE__'

    # mixture model consisting of a discrete part and a base distribution
    def __init__(self,
                 base: Union[norminvgamma, rv_continuous, HpypBase] = None,
                 discrete: Categorical = None,
                 is_root: bool = False
                 ):

        self.base = base
        self.is_root = is_root
        # if is_root:
        #     if not isinstance(base, norminvgamma) or not isinstance(base, rv_continuous_frozen):
        #         raise ValueError("If root base, must be either a norminvgamma distribution or some other frozen continuous distribution object.")
        if discrete is None:
            self.discrete = Categorical({self.NEW_TABLE:1.})
        else:
            self.discrete = discrete

    def sample(self):
        if self.is_root:
            return self.base.rvs(1)[0]
        else:
            k = self.discrete.sample()
            if k == self.NEW_TABLE:
                return self.base.sample()
            else:
                return k

    # update with new discrete part
    def update(self, new_categorical):
        self.discrete = new_categorical

    # compute the probability the next observation is k
    def p_key(self, k):
        if self.is_root:
            return 0
        return self.discrete[k] + self.discrete[self.NEW_TABLE] * self.base.p_key(k)
