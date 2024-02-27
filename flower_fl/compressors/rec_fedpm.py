from flower_fl.compressors.compressor import Compressor

from typing import Dict
import torch
import math
import numpy as np
import random
from scipy.stats import entropy


class FedPMRECCompressor(Compressor):
    def __init__(
            self,
            params: Dict,
            device: torch.device,
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
        orig_shape = posterior_update.shape
        posterior_update = np.reshape(posterior_update, [-1, 1])
        prior = np.reshape(prior, [-1, 1])

        assert prior is not None
        sampled_params = np.zeros_like(posterior_update)

        ids, kls = self.compute_indices(
            prior=prior.flatten(),
            posterior=posterior_update.flatten(),
            compress_config=compress_config,
            old_ids=old_ids
            )
        start_i = 0
        block_sizes = []
        nums_bits = []
        # print(f'--- Layer Size {len(posterior_update)} ---')
        rhos = []
        for i in ids:
            block_size = i - start_i + 1
            if compress_config.get('adaptive'):
                num_bits = int(compress_config.get('kl_rate'))
            else:
                num_bits = int(np.log2(compress_config.get('num_samples')))
            num_samples = np.power(2, num_bits)
            local_prm = posterior_update[start_i: i + 1]
            server_prm = prior[start_i: i + 1]

            samples = (np.random.uniform(
                size=(len(local_prm), num_samples)) < server_prm).astype(int)

            posterior_prob = local_prm * samples + (1 - local_prm) * (1 - samples)
            prior_prob = server_prm * samples + (1 - server_prm) * (1 - samples)
            posterior_prob = posterior_prob / prior_prob
            joint_posterior = np.exp(np.sum(np.log(posterior_prob), axis=0))
            joint_posterior = joint_posterior / np.sum(joint_posterior)
            index = random.choices(
                list(range(num_samples)), weights=list(joint_posterior), k=1)

            sampled_params[start_i: i + 1] = samples[:, index]
            block_sizes.append(block_size)
            nums_bits.append(num_bits)
            rhos.append(num_bits / np.sum(kls[start_i: i + 1]))
            start_i = i + 1

        # print(f'N° Params: {np.mean(block_sizes)} +- {np.std(block_sizes)} - Bits: {np.mean(nums_bits)}')
        sampled_params = np.where(sampled_params == 0, 0.01, sampled_params)
        sampled_params = np.where(sampled_params == 1, 0.99, sampled_params)

        return sampled_params, block_sizes, nums_bits, rhos, ids

    def compute_indices(self, prior, posterior, compress_config, old_ids):
        kls = posterior * np.log2(posterior/prior) + (1-posterior) * np.log2((1-posterior)/(1-prior))
        if old_ids is not None:
            return old_ids, kls
        if not compress_config.get('adaptive'):
            ids = list(np.arange(0, len(prior), compress_config.get('block_size'))[1:])
            ids.append(len(prior))
            return ids, kls
        if np.sum(kls) < int(compress_config.get('kl_rate')):
            return [len(kls)-1], kls
        ids = []
        s = 0
        run_kl = 0
        for i, k in enumerate(kls):
            s += 1
            run_kl += k
            if run_kl > int(compress_config.get('kl_rate')) or s >= 256:
                ids.append(i-1)
                s = 0
                run_kl = 0
        ids.append(len(kls))
        return ids, kls

