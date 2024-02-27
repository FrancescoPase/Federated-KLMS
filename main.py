import flwr.client

from flower_fl.strategies.dense_strategy import DenseStrategy
from flower_fl.strategies.lsd_strategy import QLSDStrategy
from flower_fl.strategies.fedpm_strategy import FedPMStrategy

import logging
import torch
from pathlib import Path

from flower_fl.clients.dense_client import DenseClient
from flower_fl.clients.lsd_client import LSDClient
from flower_fl.clients.fedpm_client import FedPMClient
from flower_fl.utils.read_data import read_params
from flower_fl.utils.load_data import get_data_loaders
from flower_fl.utils.store_output import create_output_folder
from flower_fl.utils.load_compressor import get_compressor

import os
import sys
sys.path.append(os.getcwd())

get_strategy = {
    'dense': DenseStrategy,
    'lsd': QLSDStrategy,
    'fedpm': FedPMStrategy
}

get_client = {
    'dense': DenseClient,
    'lsd': LSDClient,
    'fedpm': FedPMClient
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if "cuda" in DEVICE.type:
    client_resources = {"num_gpus": 1,
                        "num_cpus": 6}

current_folder = Path(__file__).parent.resolve()
params_path = current_folder.joinpath('flower_fl/configs/params.yaml')
params = read_params(params_path)

NUM_CLIENTS = params.get('simulation').get('n_clients')


def client_fn(cid) -> flwr.client.Client:
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    if params.get('compressor').get('compress'):
        compressor = get_compressor(
            compressor_type=params.get('compressor').get('type'),
            params=params,
            device=DEVICE
        )
    return get_client[params.get('simulation').get('strategy')](
        params=params,
        client_id=cid,
        train_data_loader=trainloader,
        test_data_loader=valloader,
        device=DEVICE,
        compressor=compressor
    )


if params.get('simulation').get('store_output'):
    sim_name = create_output_folder(params=params)
else:
    sim_name = None

if not params.get('simulation').get('verbose'):
    logging.disable()

for run in range(params.get('simulation').get('tot_sims')):

    trainloaders, valloaders, testloader = get_data_loaders(
        dataset=params.get('data').get('dataset'),
        nclients=NUM_CLIENTS,
        batch_size=params.get('data').get('minibatch_size'),
        classes_pc=params.get('data').get('classes_pc'),
        split=params.get('data').get('split')
    )

    strategy = get_strategy[params.get('simulation').get('strategy')](
        params=params,
        global_data_loader=testloader,
        fraction_fit=params.get('simulation').get('fraction_fit'),
        fraction_evaluate=params.get('simulation').get('fraction_evaluate'),
        min_fit_clients=0,
        min_evaluate_clients=0,
        min_available_clients=0,
        sim_folder=sim_name
    )

    flwr.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources=client_resources,
        config=flwr.server.ServerConfig(num_rounds=params.get('simulation').get('n_rounds')),
        strategy=strategy,
        ray_init_args={
            "ignore_reinit_error": params.get('ray_init_args').get('ignore_reinit_error'),
            "include_dashboard": params.get('ray_init_args').get('include_dashboard'),
            "num_cpus": params.get('ray_init_args').get('num_cpus'),
            "num_gpus": params.get('ray_init_args').get('num_gpus'),
            "local_mode": params.get('ray_init_args').get('local_mode')
        }
    )
