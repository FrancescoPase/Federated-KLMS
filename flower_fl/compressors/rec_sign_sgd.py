from flower_fl.compressors.compressor import Compressor

from typing import Dict
import copy
import torch
import math
import numpy as np
import random
from scipy.stats import entropy


class SignSGDRECCompressor(Compressor):
    def __init__(
            self,
            params: Dict,
            device: torch.device
    ) -> None:
        super().__init__(
            params=params,
            device=device
        )

    def sigmoid(self, xs):
        return 1 / (1 + np.exp(-xs))
    def get_dense_deltas(self, params):
        concat_params = []
        params_d = {}
        for k, v in params.items():
            v = copy.deepcopy(v.detach())
            concat_params.append(v.view(-1))
            params_d[k] = v.size()
        flattened = torch.cat(concat_params)
        return flattened, params_d

    def load_deltas(self, params, params_d, deltas):
        i = 0
        for k, k_shape in params_d.items():
            # TODO: do we need to change it to k.numel()?
            # TODO: Maybe sanity check without compression?
            #k_size = k_shape.numel()
            k_size = deltas[k].numel()
            deltas[k] = params[i : (i + k_size)].view(k_shape)
            i += k_size
        return deltas
    def compress(
            self,
            updates: torch.Tensor,
            compress_config: Dict = None,
            iter_num=None,
    ) -> torch.Tensor:
        warm_up_period = len(self.params.get('compressor').get('rec').get('warm_up_num_samples'))
        if iter_num >= warm_up_period:
            num_samples = self.params.get("compressor").get("rec").get("num_samples")
        else:
            num_samples = self.params.get("compressor").get("rec").get("warm_up_num_samples")[iter_num]

        block_length = self.params.get("compressor").get("rec").get("block_size")
        compressed_delta = []
        num_params = 0
        for k, v in updates.items():
            num_params += torch.numel(v)
        num_blocks = math.ceil(num_params / block_length)

        local_params, params_d = self.get_dense_deltas(updates)
        local_params = local_params.cpu().numpy()
        sampled_params = np.reshape(np.zeros_like(local_params), [-1, 1])

        for i in range(num_blocks):
            if (i + 1) * block_length <= num_params:
                local_prm = local_params[i * block_length: (i + 1) * block_length]
            else:
                local_prm = local_params[i * block_length:]
            local_prm = np.reshape(local_prm, [-1, 1])
            scale_factor = 20.0 / (np.max(np.abs(local_prm)) + 1e-15)
            local_prm = scale_factor * local_prm
            local_prm = self.sigmoid(local_prm)
            # TODO: For now, the prior is 0.5 for each update.
            samples = (np.random.uniform(size=(len(local_prm), num_samples)) < 0.5).astype(int)
            posterior = local_prm * samples + (1 - local_prm) * (1 - samples)
            prior = 0.5 * samples + (1 - 0.5) * (1 - samples)
            posterior = posterior / prior
            joint_posterior = np.exp(np.sum(np.log(posterior + 1e-15), axis=0))
            joint_posterior = joint_posterior / (np.sum(joint_posterior) + 1e-15)
            index = random.choices(list(range(num_samples)), weights=list(joint_posterior), k=1)

            sampled_params[i * block_length: i * block_length + len(local_prm)] = samples[:, index]
        sampled_params = np.where(sampled_params == 0, 0.01, sampled_params)
        sampled_params = np.where(sampled_params == 1, 0.99, sampled_params)
        model_update = self.load_deltas(torch.tensor(sampled_params, device=self.device).to(
            torch.float), params_d, updates)

        for i, (name, param) in enumerate(updates.items()):
            update = 2 * model_update[name] - 1
            compressed_delta.append(update.cpu().numpy())

        return compressed_delta, num_blocks * np.log2(num_samples) / num_params



