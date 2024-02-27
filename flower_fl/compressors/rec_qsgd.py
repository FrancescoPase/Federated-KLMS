from flower_fl.compressors.compressor import Compressor

from typing import Dict
import copy
import torch
import math
import numpy as np
import random
from scipy.stats import entropy


class QSGDRECCompressor(Compressor):
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
            updates: torch.Tensor,
            compress_config: Dict = None,
            iter_num = None,
    ) -> torch.Tensor:
        warm_up_period = len(self.params.get('compressor').get('rec').get('warm_up_num_samples'))
        if iter_num >= warm_up_period:
            num_samples = self.params.get("compressor").get("rec").get("num_samples")
        else:
            num_samples = self.params.get("compressor").get("rec").get("warm_up_num_samples")[iter_num]

        block_length = self.params.get("compressor").get("rec").get("block_size")
        compressed_delta = []
        num_params = 0
        for name, param in updates.items():
            num_params += torch.numel(param)
        num_blocks = math.ceil(num_params / block_length)

        for i, (name, param) in enumerate(updates.items()):
            param_numpy = copy.deepcopy(param.detach().view(-1)).cpu().numpy()
            norm = np.sqrt(np.sum(np.square(param_numpy)))

            # Layer_req calculation
            param_reshaped = np.reshape(param_numpy, [-1, 1])
            layer_level = param_reshaped / norm

            pos_layer_level = np.where(layer_level > 0, layer_level, 0)
            pos_freq = np.mean(pos_layer_level)
            neg_layer_level = np.where(layer_level < 0, - layer_level, 0)
            neg_freq = np.mean(neg_layer_level)
            layer_freq = np.asarray([neg_freq, 1.0 - neg_freq - pos_freq, pos_freq])
            # print('layer frequencies: {}'.format(layer_freq))

            sampled_params = np.reshape(np.zeros_like(param_numpy), [-1, 1])
            num_params = torch.numel(param)
            num_blocks = math.ceil(num_params / block_length)
            for i in range(num_blocks):
                if (i + 1) * block_length <= num_params:
                    local_prm = param_numpy[i * block_length: (i + 1) * block_length]
                else:
                    local_prm = param_numpy[i * block_length:]

                level_float = np.abs(local_prm) / norm

                # Construct the posterior.
                posterior_prob = np.zeros((len(level_float), 3))
                posterior_prob[:, 1] = 1 - level_float
                posterior_prob[local_prm < 0, 0] = level_float[local_prm < 0]
                posterior_prob[local_prm > 0, 2] = level_float[local_prm > 0]

                # Construct the prior.
                prior_prob = np.zeros((len(level_float), 3))
                prior_prob[:, 1] = layer_freq[1]
                prior_prob[:, 0] = layer_freq[0]
                prior_prob[:, 2] = layer_freq[2]

                # Generate samples.
                samples = np.zeros((num_samples, (len(local_prm))))
                for sample_coord in range(len(level_float)):
                    samples[:, sample_coord] = random.choices([-1, 0, 1], weights=layer_freq, k=num_samples)

                # Compute the new prob. distribution.
                posterior = np.where(samples > 0, posterior_prob[:, 0], samples)
                posterior = np.where(samples == 0, posterior_prob[:, 1], posterior)
                posterior = np.where(samples > 0, posterior_prob[:, 2], posterior)

                prior = np.where(samples > 0, prior_prob[:, 0], samples)
                prior = np.where(samples == 0, prior_prob[:, 1], prior)
                prior = np.where(samples > 0, prior_prob[:, 2], prior)

                posterior = posterior / prior
                joint_posterior = np.exp(np.sum(np.log(posterior), axis=1))
                joint_posterior = joint_posterior / np.sum(joint_posterior)

                # Take a sample from the new distribution.
                index = random.choices(list(range(num_samples)), weights=list(joint_posterior), k=1)
                sampled_params[i * block_length: i * block_length + len(local_prm)] = np.reshape(samples[index, :],
                                                                                                 [-1, 1])
            param_qsgd = torch.from_numpy(norm * sampled_params).view(param.size()).numpy()
            compressed_delta.append(param_qsgd)

        return compressed_delta, num_blocks * np.log2(num_samples) / num_params

