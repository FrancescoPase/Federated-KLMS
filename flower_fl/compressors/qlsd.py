from typing import Dict

import torch

from .compressor import Compressor


class QLSDCompressor(Compressor):
    def __init__(
            self,
            params: Dict,
            device: torch.device
    ) -> None:
        super().__init__(
            params=params,
            device=device
        )

    def compress(self, posterior_update: torch.Tensor, compress_config: Dict = None) -> torch.Tensor:
        compression_parameter = compress_config.get('compression_parameter')
        return self.quantize(posterior_update, compression_parameter)

    def quantize(self, v: torch.Tensor, compression_parameter: int) -> torch.Tensor:
        if compression_parameter == 0:
            return v.to(self.device)
        v_norm = torch.norm(v, p=2)
        if v_norm == 0:
            return v.to(self.device)
        r = compression_parameter * torch.abs(v) / v_norm
        l = torch.floor(r)
        l += torch.ceil(r - l) - torch.ones_like(l)
        b = torch.bernoulli(r - l)
        xi = (l + b) / compression_parameter
        return (v_norm * torch.sign(v) * xi).to(self.device)
