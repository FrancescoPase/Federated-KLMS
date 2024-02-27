"""
    Utility script to prepare the simulation output folder.
"""

import shutil
import os


def create_output_folder(params):
    """
        Create the simulation output folder, initialize it by copying `params_{}.yaml`
        and return the folder name
    """
    sim_name = params.get('simulation').get('strategy') + '_' + \
               params.get('data').get('dataset') + '_' + \
               str(params.get('simulation').get('fraction_fit')) + '_' + \
               str(params.get('simulation').get('n_clients')) + '_' + \
               params.get('model').get('id') + '_' + \
               params.get('data').get('split')

    if params.get('data').get('split') == 'non-iid':
        sim_name += '_' + str(params.get('data').get('classes_pc'))

    if params.get('compressor').get('compress'):
        sim_name += '_' + params.get('compressor').get('type')
        if 'rec' in params.get('compressor').get('type'):
            sim_name += '_' + str(params.get('compressor').get('rec').get('num_samples'))
            sim_name += '_' + str(params.get('compressor').get('rec').get('block_size'))
        elif params.get('compressor').get('type') == 'qlsd':
            sim_name += '_' + str(params.get('compressor').get('qlsd').get('compression_parameter'))

    if params.get('simulation').get('strategy') == 'dense':
        if params.get('compressor').get('compress'):
            if params.get('compressor').get('type') == 'sign_sgd':
                sim_name += '_' + 'local_lr_' + str(params.get('sign_sgd').get('local_lr'))
                sim_name += '_' + 'server_lr_' + str(params.get('sign_sgd').get('server_lr'))
            elif params.get('compressor').get('type') == 'sign_sgd_rec':
                sim_name += '_' + 'local_lr_' + str(params.get('sign_sgd_rec').get('local_lr'))
                sim_name += '_' + 'server_lr_' + str(params.get('sign_sgd_rec').get('server_lr'))
            elif params.get('compressor').get('type') == 'qsgd':
                sim_name += '_' + 'local_lr_' + str(params.get('qsgd').get('local_lr'))
                sim_name += '_' + 'server_lr_' + str(params.get('qsgd').get('server_lr'))
            elif params.get('compressor').get('type') == 'qsgd_rec':
                sim_name += '_' + 'local_lr_' + str(params.get('qsgd_rec').get('local_lr'))
                sim_name += '_' + 'server_lr_' + str(params.get('qsgd_rec').get('server_lr'))
        else:
            sim_name += '_' + 'local_lr_' + str(params.get('fedavg').get('local_lr'))
            sim_name += '_' + 'server_lr_' + str(params.get('fedavg').get('server_lr'))

    elif params.get('simulation').get('strategy') == 'fedpm':
        sim_name += '_' + 'local_lr_' + str(params.get('fedavg').get('local_lr'))

    elif params.get('simulation').get('strategy') == 'lsd':
        sim_name += '_' + 'server_lr_' + str(params.get('fedavg').get('server_lr'))

    return sim_name