"""
Wrapper for handling particle MCMC. Provides diagnostics, model comparison and
serialisation of MCMC logs.

@author: Steven Nguyen
"""

from . import RestFranchise, Particles, evals
from .jump_proposal import *
from .norminvgamma import norminvgamma
from scipy.stats import invgamma, poisson, norm
import scipy as sp
from typing import Union
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import zipfile
import tempfile
from typing import List

def zip_directory(directory_path, zip_file_path):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Traverse the directory tree
        for root, dirs, files in os.walk(directory_path):
            # For each file, create a path relative to the directory being zipped
            for file in files:
                # Create a full path for the file
                file_path = os.path.join(root, file)
                # Create a path relative to the directory being zipped
                relative_path = os.path.relpath(file_path, directory_path)
                # Add the file to the zip file
                zipf.write(file_path, relative_path)

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            # Convert NumPy integer to native Python int
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert NumPy float to native Python float
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # Convert NumPy array to list
            return obj.tolist()
        # Default serialization for other data types
        return super().default(obj)

def check_data(func):
    def wrapper(self, *args, **kwargs):
        if getattr(self, 'tree') is None:
            raise ValueError(f"Method requires mcmc object to have a RestFranchise")
        if getattr(self, 'data') is None:
            raise ValueError(f"Method requires mcmc object to have data")

    return wrapper



class TreeMCMC:

    def __init__(
        self,
        tree : RestFranchise = None,
        data : pd.DataFrame = None,
        num_samples : int = 5000,
        burnin : int = None,
        base_dist = None,
        num_particles : int = 5,
        prior_mean_njumps: float = 1.,
        fix_jump_rate : bool = False,
        disc: float = 0.5,
        conc: float = 0,
        var_prior: str = 'gamma',
        var_prior_params: List[float] = None,
        proposal_method: Proposal = None,
        progress_bar: bool = False,
    ):
        self.tree = tree
        self.data = data
        self.num_samples = num_samples
        self.burnin = burnin if burnin is not None else num_samples // 2
        self.progress_bar = progress_bar
        if proposal_method is None:
            proposal_method = MultiProposal(strategy='cycle')
            proposal_method.add_proposals(
                RandomWalkProposal(),
                SwapProposal(method='parent_child'),
            )
        self.proposal_method = proposal_method
        self.num_particles = num_particles
        self.fix_jump_rate = fix_jump_rate
        self.var_prior = var_prior
        self.prior_mean_njumps = prior_mean_njumps
        self.conc = conc
        self.disc = disc
        if var_prior_params is None:
            self.var_prior_params = self._default_var_prior_params()
        else:
            assert self._validate_prior(var_prior, var_prior_params, prior_mean_njumps)
            self.var_prior_params  =var_prior_params
        # check base distribution
        if base_dist is None:
            if self.var_prior == 'wishart':
                var_prior_params = self.var_prior_params
                self.base_dist = norminvgamma(var_prior_params[0], var_prior_params[1],
                                var_prior_params[2], var_prior_params[3])
            else:
                # run checks on data
                mean = data['obs'].mean()
                sd = data['obs'].std()
                self.base_dist = norm(mean, sd)

        else:
            # check if valid base distribution
            assert self._validate_base_dist(base_dist)
            self.base_dist = base_dist

        self._samples = None
        self._logs = None
        self._seeds = None
        self.diagnostics = {}

        pass



    def _default_var_prior_params(self):
        if self.var_prior == 'gamma':
            return [1., 1.]
        if self.var_prior == 'fixed':
            return [1.]
        else:
            return [1.,1.,1.,1.]

    def _validate_base_dist(self, base_dist) -> bool:
        return True

    def _validate_prior(self, var_prior, var_prior_params, prior_mean_njumps) -> bool:

        # to do validation

        return True

    def _load_prior_params(self, prior_params):
        assert self._validate_prior(prior_params)
        self.prior_params = prior_params
        self.var_prior = prior_params['var_prior']
        # check base distribution
        if self.var_prior == 'wishart':
            var_prior_params = self.var_prior_params
            self.base_dist = norminvgamma(var_prior_params[0], var_prior_params[1],
                                          var_prior_params[2], var_prior_params[3])
        else:
            # run checks on data
            mean = self.data['obs'].mean()
            sd = self.data['obs'].std()
            self.base_dist = norm(mean, sd)


    def get_samples(self):
        return self._samples

    def get_logs(self):
        return self._logs

    def save_samples(self, path):
        if self._samples is None:
            print("No samples to save")
            return

        post_samples = []
        jump_dir = os.path.join(path, 'jumps')
        partitions_dir = os.path.join(path, 'partitions')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(jump_dir):
            os.makedirs(jump_dir)
        if not os.path.exists(partitions_dir):
            os.makedirs(partitions_dir)


        for i, chain in enumerate(self._samples):
            sp.sparse.save_npz(os.path.join(jump_dir, str(i)),chain['jumps'].tocoo())
            sp.sparse.save_npz(os.path.join(partitions_dir, str(i)),chain['partitions'].tocoo())
            post_samples.append({'jump_rate':chain['jump_rate']})
            if 'vars' in chain:
                post_samples[i]['vars'] = chain['vars']

        pd.DataFrame(post_samples).to_parquet(os.path.join(path, 'parameter_trace.parquet'))


    def load_samples(self, path):
        param_trace = pd.read_parquet(os.path.join(path, 'parameter_trace.parquet'))
        samples = param_trace.to_dict('records')

        for f in os.listdir(os.path.join(path, 'jumps')):
            idx = int(f.split('.')[0])
            jp = sp.sparse.load_npz(os.path.join(path, 'jumps', f))
            jp = jp.tolil()
            samples[idx]['jumps'] = jp

        for f in os.listdir(os.path.join(path, 'partitions')):
            idx = int(f.split('.')[0])
            partition = sp.sparse.load_npz(os.path.join(path, 'partitions', f))
            partition = partition.tolil()
            samples[idx]['partitions'] = partition

        return samples


    def save_log(self, path):
        if self._logs is None:
            print("No log to save")
            return

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, 'proposed')):
            os.makedirs(os.path.join(path, 'proposed'))

        log_values = []
        to_select = ['accepted', 'log_lik', 'log_acc', 'log_acc_prob', 'log_lik_pre', 'jp_log_ratio', 'same_proposal']
        for i, log in enumerate(self._logs):
            sp.sparse.save_npz(os.path.join(os.path.join(path, 'proposed', str(i))),log['proposed'].tocoo())
            log_values.append({
                v : log[v] for v in to_select
            })

        pd.DataFrame(log_values).to_parquet(os.path.join(path, 'log.parquet'))


    def load_log(self, path):
        logs = pd.read_parquet(os.path.join(path, 'log.parquet'))

        logs = logs.to_dict('records')

        for f in os.listdir(os.path.join(path, 'proposed')):
            idx = int(f.split('.')[0])
            proposed = sp.sparse.load_npz(os.path.join(path, 'proposed', f))
            logs[idx]['proposed'] = proposed.tolil()

        return logs


    def load_metadata(self, path):
        """
        load params as a dict from a .json file
        """
        with open(path, 'r') as f:
            metadata = json.load(f)

        prior = metadata['prior']
        try:
            self.num_samples = metadata['num_samples']
            self.var_prior_params = prior['var_prior_params']
            self.var_prior = prior['var_prior']
            self.prior_mean_njumps = prior['prior_mean_njumps']
            self.fix_jump_rate = bool(prior['fix_jump_rate'])
            self.num_particles = metadata['num_particles']
            self._seeds = metadata['seeds']
            self.conc = metadata['hpyp_params']['conc']
            self.disc = metadata['hpyp_params']['disc']
            # check base dist
            if metadata['base_dist']['family'] == 'normal':
                self.base_dist = norm(metadata['base_dist']['params']['mean'],
                                      metadata['base_dist']['params']['std']
                                      )
            else:
                self.base_dist = norminvgamma(**metadata['base_dist']['params'])
        except KeyError as e:
            print(f"Could not find parameter {e} in metadata")


    def load_proposal(self, proposal_pkl:str):
        with open(proposal_pkl, 'rb') as f:
            self.proposal_method = pkl.load(f)

    def remove_chain(self, chain:int):
        self._samples.remove(chain)
        self._logs.remove(chain)

    # def add_chains(self, chains: List = None, dir: str = None):
    #     if mcmc_dir is None:
    #         with open(os.path.join(mcmc_dir, 'samples.pkl'), 'rb') as f:
    #             self._samples.extend(pkl.load(f))

    def save_mcmc(self, file: str):
        """
        save run of MCMC to a directory
        """
        BASE_TRACE_DIR = 'traces'
        BASE_LOG_DIR = 'logs'
        BASE_PROPOSAL_FILE = 'proposal.pkl'
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

        
        with tempfile.TemporaryDirectory() as temp_dir:
        
            with open(os.path.join(temp_dir, BASE_PROPOSAL_FILE), 'wb') as f:
                pkl.dump(self.proposal_method, f)
    
            metadata = {
                'num_samples' : self.num_samples,
                'num_particles' : self.num_particles,
                'prior' :{
                    'prior_mean_njumps': self.prior_mean_njumps,
                    'fix_jump_rate' : self.fix_jump_rate,
                    'var_prior' : self.var_prior,
                    'var_prior_params': self.var_prior_params
                },
                'hpyp_params' : {
                    'conc' : self.conc,
                    'disc' : self.disc
                },
                'seeds' : self._seeds if self._seeds is not None else '',
                'log_dir' : BASE_LOG_DIR,
                'trace_dir' : BASE_TRACE_DIR,
                'proposal_file': BASE_PROPOSAL_FILE
            }
            if isinstance(self.base_dist, norminvgamma):
                metadata['base_dist'] = {
                    'family' : 'normalinvgamma',
                    'params' : {'mu0' : self.base_dist.mu0,
                                'nu' : self.base_dist.nu,
                                'alpha' : self.base_dist.alpha,
                                'beta' : self.base_dist.beta
                                }
                }
            else:
                metadata['base_dist'] = {
                    'family' : 'normal',
                    'params' : {'mean' : self.base_dist.mean(),
                                'std': self.base_dist.std()
                                }
                }
    
            with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent = 4,cls=NumpyJSONEncoder)
    
            self.save_log(os.path.join(temp_dir,BASE_LOG_DIR ))
            self.save_samples(os.path.join(temp_dir, BASE_TRACE_DIR))
            if not file.endswith('.zip'):
                file += '.zip'
            with zipfile.ZipFile(os.path.join(file), 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Traverse the directory tree
                for root, dirs, files in os.walk(temp_dir):
                    # For each file, create a path relative to the directory being zipped
                    for f in files:
                        # Create a full path for the file
                        file_path = os.path.join(root, f)
                        # Create a path relative to the directory being zipped
                        relative_path = os.path.relpath(file_path, temp_dir)
                        # Add the file to the zip file
                        zipf.write(file_path, relative_path)

    def load_mcmc(self, mcmc_zip : str, load_samples : bool = True, load_logs : bool = False):
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(mcmc_zip, 'r') as zip_bytes:
                zip_bytes.extractall(temp_dir)

            self.load_metadata(os.path.join(temp_dir, 'metadata.json'))
            self.load_proposal(os.path.join(temp_dir, 'proposal.pkl'))

            # load traces and logs
            if load_logs : self._logs = self.load_log(os.path.join(temp_dir, 'logs'))
            if load_samples : self._samples = self.load_samples(os.path.join(temp_dir, 'traces'))

    def run_with(self, tree: RestFranchise, data: pd.DataFrame, trait_column:str = 'obs',
                 parallel: bool = False, max_workers:int = os.cpu_count()
                 ):
        self.tree = tree
        df = data[['node_name', trait_column]].rename(columns = {trait_column : 'obs'})
        self.data = df
        self.run(parallel, max_workers)

    #@check_data
    def run(self, num_chains: int, parallel: bool = False, max_workers: int = os.cpu_count()):
        var_prior = self.var_prior
        self.num_chains = num_chains
        var_prior_params = self.var_prior_params
        if (var_prior == 'fixed') or (var_prior == 'gamma'):
            var_prior_params = [self.base_dist.mean(), self.base_dist.std()] + var_prior_params

        seed = np.random.get_state()[1][0]
        self._seeds = [seed + i for i in range(num_chains)]

        if not parallel:
            self._logs = [self.tree.particleMCMC(
                data = self.data, proposal = self.proposal_method, num_particles = self.num_particles,
                n_iter = self.num_samples, prior_mean_njumps=self.prior_mean_njumps, var_prior=var_prior,
                conc = self.conc, disc = self.disc,
                var_prior_params = var_prior_params, progress_bar = self.progress_bar, seed = i + seed
            ) for i in range(num_chains)]

        else:
            self._multi_chain_parallel(max_workers, data = self.data, proposal = self.proposal_method, num_particles = self.num_particles,
                n_iter = self.num_samples, prior_mean_njumps=self.prior_mean_njumps, var_prior=var_prior, conc = self.conc, disc = self.disc,
                var_prior_params = var_prior_params, progress_bar = False)

        # write samples
        if self.var_prior == 'gamma':
            self._samples = [
                {
                    'jumps': log['jumps'],
                    'jump_rate': log['jump_rate'],
                    'partitions': log['partitions'],
                    'vars': log['vars']
                }
                for log in self._logs
            ]
        else:
            self._samples = [
                {
                    'jumps': log['jumps'],
                    'jump_rate': log['jump_rate'],
                    'partitions': log['partitions'],
                }
                for log in self._logs
            ]

        self._logs = [log['log'] for log in self._logs]
        for i,s in enumerate(self._seeds):
            self.diagnostics[int(s)] = {
                'acc_rate' : self._logs[i].pop('acc_rate'),
                'same_proposal_rate' : self._logs[i].pop('same_proposal_rate')
            }

            self.compute_diagnostics(i)


    def compute_diagnostics(self, chain:int):
        # to do more diagnostics
        pass


    def _multi_chain_parallel(self, max_workers: int, **kwargs):
        trees = [self.tree.deep_copy() for _ in range(self.num_chains)]
        futures = []
        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            for i in range(self.num_chains):
                kwargs['seed'] = self._seeds[i]
                futures.append(executor.submit(trees[i].particleMCMC, **kwargs))

            self._logs = [f.result() for f in futures]

    #@check_data
    def sample_ll_jp(self, jp: List[int], it: int = 1000, num_particles: int = None):
        """
        sample data likelihoods using particle filtering for a particular jump configuration
        also return posterior variance under the case of a gamma prior
        """
        if num_particles is None: num_particles = self.num_particles
        gamma_prior = self.var_prior == 'gamma'
        if self.var_prior == 'gamma':
            alpha,beta = self.var_prior_params
        elif self.var_prior == 'fixed':
            post_var = self.var_prior_params[0]
        else:
            post_var = 1.
        tree = self.tree.deep_copy()
        tree.jps = jp
        datall = np.zeros(it)
        postvar = np.zeros(it)
        for i in range(it):
            if gamma_prior:
                post_var, post_alpha, post_beta = tree.sample_post_var(self.data, alpha, beta)
                postvar[i] = invgamma.logpdf(x=post_var, a=post_alpha, scale=post_beta)
            ps = Particles(tree=tree, num_particles=self.num_particles, forward=True,
                           var_prior=self.var_prior, kernel_var=post_var, base=self.base_dist)
            datall[i] = ps.particle_filter_integrated(self.data)

        return datall, postvar

    #@check_data
    def compare_jps_by_ll(self, *args, parallel:bool = True, max_workers:int = 6, **kwargs):
        """
        sample the data loglikelihoods for jumps specified in *args
        **kwargs - parameters for particle filtering
        """
        futures = []
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for jp in args:
                    futures.append(executor.submit(self.sample_ll_jp, jp, **kwargs))

            lls = [f.result() for f in futures]
        else:
            lls = [self.sample_ll_jp(jp, **kwargs) for jp in args]

        # for now just return the loglikelihoods
        return lls


    #@check_data
    def sample_bayes_factors(self, jumps: List[int], num_it:int, burnin:int, seed:int = None):
        if seed is not None:
            np.random.seed(seed)

        tree = self.tree.deep_copy()
        prev_jp = None
        prev_loglik = -np.inf
        n = len(jumps)
        counts = [0 for _ in range(n)]
        prior =  self.var_prior

        if prior == 'gamma':
            alpha, beta = self.var_prior_params
        elif prior == 'fixed':
            post_var =  self.var_prior_params[0]
        elif prior == 'wishart':
            post_var = 1. # doesn't matter

        for j in range(num_it):
            tree.jps = jumps[j % n]
            if prior == 'gamma':
                post_var, post_alpha, post_beta = self.tree.sample_post_var(self.data, alpha, beta)
            ps = Particles(tree=tree, num_particles=self.num_particles, forward=True,
                           var_prior=self.var_prior, kernel_var=post_var, base=self.base_dist)
            log_lik = ps.particle_filter_integrated(self.data)
            log_acc_prob = np.log(np.random.rand(1))[0]
            if log_lik - prev_loglik > log_acc_prob:
                prev_jp = j % n
                prev_loglik = log_lik
                if j >= burnin:
                    counts[j % n] += 1
            else:
                if j >= burnin:
                    counts[prev_jp] += 1

        return counts

    #@check_data
    def bayes_factors_multiple(self, jumps:List[int], num_chains:int = 1, num_it:int = 1000, burnin:int = None, parallel: bool = False, n_cores:int = None):
        if burnin is None:
            burnin = num_it // 2
        if n_cores is None:
            n_cores = os.cpu_count()
        if not parallel:
            return np.array([
                self.sample_bayes_factors(jumps, num_it, burnin) for _ in range(num_chains)
            ])
        else:
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                futures = []
                for i in range(num_chains):
                    futures.append(executor.submit(self.sample_bayes_factors,
                                                   jumps, num_it, burnin, seed = i))

                samples = [f.result() for f in futures]
                return samples

    def _grid_trace_plot(self, traces: np.ndarray, figsize: (int, int)):
        n_chains = traces.shape[0]
        if n_chains == 1:
            fig, axes = plt.plot(traces[0], figsize)
        else:
            cols = int(np.ceil(np.sqrt(n_chains)))
            rows = int(np.ceil(n_chains / cols))
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
            axes = axes.flatten()
            for i in range(n_chains):
                ax = axes[i]
                ax.plot(traces[i])
                ax.set_title(label = f'Chain {i+1}')

            # Hide any unused subplots
            for j in range(n_chains, rows * cols):
                fig.delaxes(axes[j])

            # Adjust the layout to prevent overlap


        return fig, axes

    def plot_acceptance_rates(self, figsize: (int, int) = (10,8)):
        acc_rates = np.array([
            np.cumsum(log['accepted']) / (1 + np.arange(len(log['accepted']))) for log in self._logs
        ])
        fig, axes = self._grid_trace_plot(acc_rates, figsize)
        fig.suptitle('Acceptance Rates')
        return fig, axes

    def plot_traces(self, parameter: str, figsize: (int,int) = (10,8)):
        valid_params = {'jump_rate'} if self.var_prior != 'gamma' else {'jump_rate', 'vars'}
        if not parameter in valid_params:
            print(f"{parameter} is not a valid parameter to plot the trace of")
            return
        traces = np.array([
            sample[parameter] for sample in self._samples
        ])
        fig, axes = self._grid_trace_plot(traces, figsize)
        fig.suptitle(parameter)
        return fig, axes

    def thin_and_combine(self, thinning: int = 1, burnin: int = None):
        jps = []
        partitions = []
        jump_rate = []
        if self.var_prior == 'gamma' : var = []
        for chain in self._samples:
            if burnin is None:
                burnin = len(chain['jumps']) // 2

            jps.append(chain['jumps'][burnin::thinning])
            partitions.append(chain['partitions'][burnin::thinning])
            jump_rate.append(chain['jump_rate'][burnin::thinning])
            if self.var_prior == 'gamma': var.append(chain['vars'][burnin::thinning])

        return {
            'jumps' : jps,
            'jump_rate' : jump_rate,
            'partitions' : partitions,
            'vars' : var
        }



    def bayes_factor_nojumps(self, chain: Union[int, dict], burn_in: int = None):
        """
        compute the bayes factor for the hypothesis 'no jump' against 'at least one jump'
        """
        if isinstance(chain, int):
            chain = self._samples[chain]
        jps = chain['jumps']
        if burn_in is None:
            burn_in = jps.shape[0] // 2
        njps = np.sum(jps[burn_in:], axis=1) - 1
        post0 = np.mean(njps == 0)

        prior_expected_njps = self.prior_params['jump_rate'] * self.tree.tl
        if self.prior_params['fix_jump_rate']: # prior is poisson
            prior0 = poisson.pmf(0, prior_expected_njps)
        else: # prior is poisson | exponential
            prior0 = 1. / (prior_expected_njps + 1)


        return (1. - post0) / post0 / ((1. - prior0) / prior0) if post0 > 0 else np.Inf


    def summarise_results(self, result_dir: str, burnin: int = None, ground_truth: List[int] = None, chain: Union[int, dict] = None):
        """
        summarise chains by computing a summary table and returning the predicted jump configuration for each chain
        """
        if burnin is None:
            burnin = self.num_samples // 2

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        predicted_jps = []
        if chain is None:
            for i, chain in enumerate(self._samples):
                jps = chain['jumps']
                partitions = chain['partitions']
                summary = evals.summarise_jump_trace(jps[burnin:], partitions[burnin:],
                                                     list(self.tree.nodes.keys()), ground_truth=ground_truth)
                summary.to_csv(os.path.join(result_dir, f"{self._seeds[i]}_summary.csv"), index = False)
                self.tree.plot_with_jp(jp=summary['predicted_jp'].tolist(),
                                  file=os.path.join(result_dir, f"{self._seeds[i]}_tree.png"),
                                  dpi=200
                                  )
                predicted_jps.append(summary['predicted_jp'])
        elif isinstance(chain, int):
                i = chain
                chain = self._samples[i]
                jps = chain['jumps']
                partitions = chain['partitions']
                summary = evals.summarise_jump_trace(jps[burnin:], partitions[burnin:],
                                                     list(self.tree.nodes.keys()), ground_truth=ground_truth)
                summary.to_csv(os.path.join(result_dir, f"{self._seeds[i]}_summary.csv"), index=False)
                self.tree.plot_with_jp(jp=summary['predicted_jp'].tolist(),
                                       file=os.path.join(result_dir, f"{self._seeds[i]}_tree.png"),
                                       dpi=200
                                       )
                predicted_jps.append(summary['predicted_jp'])
        else:
            jps = chain['jumps']
            partitions = chain['partitions']
            summary = evals.summarise_jump_trace(jps[burnin:], partitions[burnin:],
                                                 list(self.tree.nodes.keys()))
            summary.to_csv(os.path.join(result_dir, f"_summary.csv"), index=False)
            self.tree.plot_with_jp(jp=summary['predicted_jp'].tolist(),
                                   file=os.path.join(result_dir, f"tree.png"),
                                   dpi=200
                                   )
            predicted_jps.append(summary['predicted_jp'])

        return predicted_jps

    def save_diagnostics(self, save_to):
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        with open(os.path.join(save_to, 'diagnostics.json'), 'w') as f:
            json.dump(self.diagnostics, f, indent=4, cls=NumpyJSONEncoder)

        fig, axes = self.plot_acceptance_rates(figsize = (20,12))
        plt.savefig(os.path.join(save_to, 'acceptance_rates.png'))
        plt.clf()

        fig, axes = self.plot_traces('jump_rate', figsize = (20,12))
        plt.savefig(os.path.join(save_to, 'jump_rates.png'))
        plt.clf()
        if self.var_prior == 'gamma':
            fig,axes = self.plot_traces('vars', figsize = (20,12))
            plt.savefig(os.path.join(save_to, 'vars.png'))
            plt.clf()

