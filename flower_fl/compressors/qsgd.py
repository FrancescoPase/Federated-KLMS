from flower_fl.compressors.compressor import Compressor

from typing import Dict
import torch
import math
import numpy as np
import random


class QSGDCompressor(Compressor):
    def __init__(
            self,
            params: Dict,
            device: torch.device
    ) -> None:
        super().__init__(
            params=params,
            device=device
        )
        self.params = params

    def compress(
            self,
            updates: torch.Tensor,
            compress_config: Dict = None,
            iter_num = None,
    ) -> torch.Tensor:
        compressed_delta = []
        num_levels = self.params.get('qsgd').get('num_level')
        num_params = 0
        elias_bitrate = 0
        for i, (name, param) in enumerate(updates.items()):
            num_param = torch.numel(param)
            num_params += num_param
            elias_bitrate += (3 + 1.5 * math.log2(2 * (num_levels ** 2 + num_param) / num_levels * (
                    num_levels + np.sqrt(num_param)))) * num_levels * (num_levels + np.sqrt(num_param)) + 32

            param_numpy = param.cpu().numpy()
            norm = np.sqrt(np.sum(np.square(param_numpy)))
            level_float = num_levels * np.abs(param_numpy) / norm
            previous_level = np.floor(level_float)
            is_next_level = np.random.rand(*param_numpy.shape) < (level_float - previous_level)
            new_level = previous_level + is_next_level
            param_qsgd = np.sign(param_numpy) * norm * new_level / num_levels
            compressed_delta.append(param_qsgd)
        return compressed_delta, elias_bitrate / num_params
