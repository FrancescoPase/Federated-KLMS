from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import math

import flwr
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

from ..utils.load_model import load_model
from ..utils.models import get_parameters, set_parameters

from torch.utils.data import DataLoader


class QLSDStrategy(flwr.server.strategy.Strategy):
    def __init__(
        self,
        params: Dict,
        global_data_loader: DataLoader,
        loss_fn=torch.nn.CrossEntropyLoss(),
        device='cpu',
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        sim_folder=None
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.params = params
        self.global_dataset = global_data_loader
        self.loss_fn = loss_fn
        self.device = device
        self.global_model = load_model(self.params).to(self.device)
        self.sim_folder = sim_folder

        if 'rec' in self.params.get('compressor').get('type'):
            self.rho_mean = None
            self.rho_std = None
            self.ids = None
            self.avg_block_size = None
            self.avg_kl = None

        if self.sim_folder is not None:
            try:
                import wandb
            except ImportError as error:
                print(error)
                print("Please install wandb via PIP \n\t $pip install wandb")
            self.wandb_runner = wandb.init(
                project='fed_rec',
                config=self.params,
                reinit=True
            )
            self.wandb_runner.name = self.sim_folder
            self.wandb_runner.watch(self.global_model)

    def __repr__(self) -> str:
        return "Federated QLSD"

    def initialize_parameters(self, client_manager: ClientManager):
        return flwr.common.ndarrays_to_parameters(get_parameters(self.global_model))

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        fit_configurations = []
        update_ids = None
        if 'rec' in self.params.get('compressor').get('type'):
            update_ids = self.update_ids()
        for idx, client in enumerate(clients):
            fit_configurations.append(
                (client, FitIns(parameters, {'iter_num': server_round,
                                             'old_ids': update_ids}))
            )
        return fit_configurations

    def update_ids(self):
        if self.rho_mean is None:
            return None
        else:
            idx_rate = np.log2(16) / self.avg_block_size
            # if idx_rate <= self.params.get('compressor').get('rec').get('kl_rate') - self.avg_kl * self.avg_block_size:
            if self.rho_mean > 3:
                print('Update Indices...')
                return None
            else:
                print('Keep Indices...')
                return self.ids

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Parameter updates - perform the QLSD step
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = aggregate(weights_results)
        delta_layers = []
        updated_params = []
        delta_grads = []
        for layer in parameters_aggregated:
            layer_fl = layer.flatten()
            noise = np.sqrt(2 / self.params.get('lsd').get('server_lr')) * np.random.normal(
                loc=np.zeros_like(layer_fl),
                scale=np.ones_like(layer_fl)
            )
            delta_layer = (self.params.get('data').get('minibatch_size') / len(weights_results)) * layer_fl + noise
            delta_layers.append(delta_layer.reshape(layer.shape))
            # TODO: is delta_grads used anywhere?
            delta_grads.extend(self.params.get('lsd').get('server_lr') * delta_layer.flatten())
        with torch.no_grad():
            for k, (name, param) in enumerate(self.global_model.named_parameters()):
                updated_params.append(param.cpu().numpy() - (self.params.get('lsd').get('server_lr') * delta_layers[k]))
        set_parameters(self.global_model, updated_params)

        log_metrics = {}
        for c in results:
            for k, v in c[1].metrics.items():
                if 'Ids' not in k:
                    if log_metrics.get(k) is None:
                        log_metrics[k] = v
                    else:
                        log_metrics[k] += v
        for k in log_metrics.keys():
            if 'Ids' not in k:
                log_metrics[k] /= len(results)

        if 'Rho Mean' in set(log_metrics.keys()):
            self.rho_mean = log_metrics.get('Rho Mean')
            self.rho_std = log_metrics.get('Rho Std')
            self.ids = self.aggregate_ids(results)
            self.avg_block_size = log_metrics.get('Block Size')
            self.avg_kl = log_metrics.get('KL Divergence')

        if self.sim_folder:
            self.wandb_runner.log(log_metrics)

        print(f'{len(weights_results)} aggregated clients')
        print()
        return ndarrays_to_parameters(updated_params), {}

    def aggregate_ids(self, results):
        ids_s = []
        sizes = []
        for r in results:
            ids_s.append(r[1].metrics.get('Ids'))
            sizes.append(len(r[1].metrics.get('Ids')))
        new_ids = []
        for i in range(max(sizes)):
            l = 0
            idx = 0
            for e in ids_s:
                if len(e) > i:
                    idx += e[i]
                    l += 1
            idx = math.ceil(idx/l)
            if len(new_ids) > 0:
                if idx > new_ids[-1]:
                    new_ids.append(idx)
            else:
                new_ids.append(idx)
        return new_ids

    def configure_evaluate(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(
            self,
            server_round: int,
            parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        correct = 0
        total = 0
        set_parameters(self.global_model, parameters_to_ndarrays(parameters))
        with torch.no_grad():
            inputs, labels = next(iter(self.global_dataset))
            self.global_model.zero_grad()
            outputs = self.global_model(inputs.to(self.device))
            loss = self.loss_fn(outputs, labels.to(self.device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f'Round {server_round} - Loss: {loss}  /  Accuracy: {accuracy}')

        if self.sim_folder is not None:
            self.wandb_runner.log({
                "Global Loss": loss,
                "Global Acc": accuracy
            })
        return loss, {'accuracy': accuracy}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
