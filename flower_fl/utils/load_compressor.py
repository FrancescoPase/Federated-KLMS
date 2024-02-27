from ..compressors.rec_lsd import LSDRECCompressor
from ..compressors.qlsd import QLSDCompressor
from ..compressors.sign_sgd import SignSGDCompressor
from ..compressors.rec_sign_sgd import SignSGDRECCompressor
from ..compressors.qsgd import QSGDCompressor
from ..compressors.rec_qsgd import QSGDRECCompressor
from ..compressors.rec_fedpm import FedPMRECCompressor
from ..compressors.compressor import Compressor


compressor_dict = {'lsd_rec': LSDRECCompressor,
                   'qlsd': QLSDCompressor,
                   'fedpm_rec': FedPMRECCompressor,
                   'sign_sgd': SignSGDCompressor,
                   'sign_sgd_rec': SignSGDRECCompressor,
                   'qsgd': QSGDCompressor,
                   'qsgd_rec': QSGDRECCompressor}


def get_compressor(
        compressor_type: str,
        **kwargs
) -> Compressor:
    return compressor_dict.get(compressor_type.lower())(**kwargs)




