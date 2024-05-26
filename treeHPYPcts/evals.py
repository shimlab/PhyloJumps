"""

treeHPYP: evals.py

Created on 2021-07-29 10:00

@author: Hanxi Sun

"""
import numpy as np
import pandas as pd
from scipy.stats import poisson, chisquare
from scipy.sparse import spmatrix
from .rest_franchise import RestFranchise


def ess(x):
    """ Compute the effective sample size of estimand of interest. Vectorised implementation. """
    m_chains, n_iters = x.shape

    variogram = lambda t: ((x[:, t:] - x[:, :(n_iters - t)])**2).sum() / (m_chains * (n_iters - t))

    post_var = gelman_rubin(x)

    t = 1
    rho = np.ones(n_iters)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iters):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t-1:t+1]) < 0

        t += 1

    return int(m_chains*n_iters / (1 + 2*rho[1:t].sum()))

def gelman_rubin(x):
    """ Estimate the marginal posterior variance. Vectorised implementation. """
    m_chains, n_iters = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True))**2).sum() / (m_chains*(n_iters - 1))

    # (over) estimate of variance
    s2 = W * (n_iters - 1) / n_iters + B_over_n

    return s2

def post_prob_jumps(samples, min_jump = 1):
    """
    calculate posterior probability of at least jump on branch
    """
    return (samples >= min_jump).sum(axis = 0) / samples.shape[0]

def threshold_jumps(samples, min_jump = 1, threshold = 0.5):
    """
    estimate the jump location based on samples
    """
    return (post_prob_jumps(samples, min_jump) > threshold).astype(int)

def summarise_jump_trace(samples, post_Zs, nodes, min_jump = 1, threshold = 0.5, median: bool = True, ground_truth = None):
    n_samples = samples.shape[0]
    data_dic = {'node_name':nodes}
    data_dic['post_prob'] = post_prob_jumps(samples, min_jump).A1
    data_dic['post_mean_jps'] = samples.mean(axis=0).A1
    data_dic[f"post_prob_greater{int(threshold*100)}"] = threshold_jumps(samples, min_jump, threshold).A1
    if median: data_dic['predicted_jp'] = estimate_jps(samples, post_Zs, at_least_one_jump=True)
    if ground_truth is not None:
        data_dic['ground_trth'] = ground_truth
        data_dic['correct_hits'] = (samples == ground_truth).sum(axis=0)
        data_dic['incorrect_hits'] = n_samples - data_dic['correct_hits']

    return pd.DataFrame(data_dic)

def estimate_jps(post_jps, post_Zs, at_least_one_jump: bool = False,  iter2estC: int = None):
    """
    Cluster assignment is acquired by minimizing the Binder loss
    see
    - Lau, John W., and Peter J. Green.     print(data_dic)"Bayesian model-based clustering procedures." Journal of Computational
      and Graphical Statistics 16.3 (2007): 526-558.
    - BINDER, D. A. (1978). Bayesian cluster analysis. Biometrika, 65(1), 31–38. doi:10.1093/biomet/65.1.31
    :param post_jps:
    :param post_Zs:
    :param at_least_one_jump:
    :param iter2estC:   number of iterations used to estimate coincidence matrix C, default to be the first half.
    :return: Zh
    """
    K = 0.5  # K = a/(a+b). K=0.5 <=> posterior median estimation of Z. See (4.1) in Lau and Green, 2007.
    if at_least_one_jump:
        indices = [i for i in range(post_Zs.shape[0]) if np.sum(post_jps[i]) > 1]
        post_jps, post_Zs = post_jps[indices], post_Zs[indices]
    N = post_jps.shape[0]
    if N == 0:  # no post_jps available
        empty_jp  = np.zeros(post_jps.shape[1])
        empty_jp[0] = 1
        return empty_jp
        # empty jump (except 1 at the root)
    if iter2estC is None:
        iter2estC = N // 2
    # coincidence matrix
    if isinstance(post_Zs, spmatrix):
        post_Zs = post_Zs.toarray()

    Cs = np.stack([Z.reshape(-1, 1) == Z for Z in post_Zs])
    rho = np.mean(Cs[:iter2estC], axis=0)
    loss = np.zeros(N - iter2estC)

    for i in range(iter2estC, N):  # i, C = 0, Cs[0]
        loss[i - iter2estC] = np.sum(np.triu(Cs[i] * (rho - K), k=1))
    argmax = np.argmax(loss) + iter2estC
    if isinstance(post_jps, spmatrix):  
        return post_jps[argmax].toarray().flatten().tolist()
    else:
        return post_jps[argmax].tolist()


def prior_expected(NJ: int, fix_jump_rate: bool, jump_rate=None, njumps=None, tree: RestFranchise = None):
    if njumps is None:
        assert jump_rate is not None and tree is not None
        njumps = jump_rate * tree.tl

    expected = np.zeros(NJ + 1)

    if fix_jump_rate:  # prior is poisson
        expected[:-1] = poisson.pmf(np.arange(NJ), njumps)
    else:  # prior is geometric (poisson | exponential)
        p0 = 1. / (njumps + 1)
        expected[:-1] = (1 - p0) * (p0 ** np.arange(NJ))
    expected[-1] = 1 - np.sum(expected)
    return expected

def bayes_factor(jps, fix_jump_rate: bool, jump_rate=None, njumps=None, tree: RestFranchise = None,
                 burn_in: int = None):
    if burn_in is None:
        burn_in = jps.shape[0] // 2
    njps = np.sum(jps[burn_in:], axis=1) - 1
    post0 = np.mean(njps == 0)
    prior0 = prior_expected(1, fix_jump_rate, jump_rate, njumps, tree)[0]
    return (1. - post0) / post0 / ((1. - prior0) / prior0) if post0 > 0 else np.Inf




