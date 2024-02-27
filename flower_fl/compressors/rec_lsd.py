from flower_fl.compressors.compressor import Compressor

from typing import Dict
import torch
import math
import numpy as np
from scipy.stats import norm, multivariate_normal


class LSDRECCompressor(Compressor):
    def __init__(
            self,
            params: Dict,
            device: torch.device
    ) -> None:
        super().__init__(
            params=params,
            device=device
        )

    def compress(
            self,
            posterior_update: torch.Tensor,
            prior=None,
            compress_config: Dict = None,
            old_ids=None
    ):

        sampled_clients = (
            int(self.params.get('simulation').get('fraction_fit') * self.params.get('simulation').get('n_clients')))
        # sigma = 2 / (self.params.get('lsd').get('server_lr') * sampled_clients**2)
        sigma = 1
        prior = np.zeros_like(posterior_update)

        ids, kls = self.compute_indices(
            prior=prior,
            posterior=posterior_update,
            compress_config=compress_config,
            sigma=sigma,
            old_ids=old_ids
        )
        nd_updates = posterior_update * self.params.get('data').get('minibatch_size') / sampled_clients
        start_i = 0
        block_sizes = []
        nums_bits = []
        # print(f'--- Layer Size {len(posterior_update)} ---')
        rhos = []
        sampled_values = np.zeros_like(posterior_update)
        for i in ids:
            block_size = i - start_i + 1
            if compress_config.get('compressor').get('rec').get('adaptive'):
                num_bits = int(compress_config.get('compressor').get('rec').get('kl_rate'))
            else:
                num_bits = int(np.log2(compress_config.get('compressor').get('rec').get('num_samples')))
            num_samples = np.power(2, num_bits)
            local_prm = nd_updates[start_i: i + 1]
            server_prm = prior[start_i: i + 1]
            sampled_values[start_i: i + 1] = self.sample_index(
                prior_mean=server_prm,
                posterior_mean=local_prm,
                sampled_clients=sampled_clients,
                num_samples=num_samples
            )
            block_sizes.append(block_size)
            nums_bits.append(num_bits)
            rhos.append(num_bits / np.sum(kls[start_i: i + 1]))
            start_i = i + 1

        return sampled_values, block_sizes, nums_bits, rhos, ids

    def sample_index(self, prior_mean, posterior_mean, sampled_clients, num_samples):
        if len(prior_mean) > 1:
            prior_rv = multivariate_normal(mean=prior_mean,
                                           cov=2 / (self.params.get('lsd').get('server_lr') * sampled_clients ** 2) * np.eye(
                                               len(prior_mean)))
            posterior_rv = multivariate_normal(mean=posterior_mean,
                                               cov=2 / (self.params.get('lsd').get(
                                                     'server_lr') * sampled_clients ** 2) * np.eye(
                                                     len(posterior_mean)))
        else:
            prior_rv = norm(loc=prior_mean,
                            scale=np.sqrt(2 / (self.params.get('lsd').get('server_lr') * sampled_clients ** 2))
                            )
            posterior_rv = norm(loc=prior_mean,
                            scale=np.sqrt(2 / (self.params.get('lsd').get('server_lr') * sampled_clients ** 2))
                                )
        samples = prior_rv.rvs(num_samples)
        prior_probs = prior_rv.pdf(samples)
        posterior_probs = posterior_rv.pdf(samples)
        posterior_probs[posterior_probs == 0] = np.min(prior_probs)
        prob_weights = posterior_probs / prior_probs
        prob_weights /= np.sum(prob_weights)
        # print(samples.shape)
        if len(prior_mean) > 1:
            return samples[np.random.choice(np.arange(samples.shape[0]), p=prob_weights), :]
        else:
            return samples[np.random.choice(np.arange(samples.shape[0]), p=prob_weights)]

    def compute_indices(self, prior, posterior, compress_config, sigma, old_ids):
        kls = (sigma**2 + posterior**2) / (2*sigma**2) - 0.5
        if old_ids is not None:
            return old_ids, kls
        if not compress_config.get('compressor').get('rec').get('adaptive'):
            ids = list(np.arange(0, len(prior) + 2,
                                 compress_config.get('compressor').get('rec').get('block_size'))[1:])

            return ids, kls
        if np.sum(kls) < int(compress_config.get('compressor').get('rec').get('kl_rate')):
            return [len(kls)-1], kls
        ids = []
        s = 0
        run_kl = 0
        for i, k in enumerate(kls):
            s += 1
            run_kl += k
            if run_kl > int(compress_config.get('compressor').get('rec').get('kl_rate')) or s >= 16:
                ids.append(i)
                s = 0
                run_kl = 0
        if ids[-1] < (len(kls)-1):
            ids.append(len(kls)-1)
        return ids, kls
