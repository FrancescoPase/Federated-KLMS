from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import numpy as np
from collections import OrderedDict

import flwr
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
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


class DenseStrategy(flwr.server.strategy.Strategy):
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
        if params.get('compressor').get('compress'):
            if params.get('compressor').get('type') == 'sign_sgd':
                self.server_lr = params.get('sign_sgd').get('server_lr')
            if params.get('compressor').get('type') == 'sign_sgd_rec':
                self.server_lr = params.get('sign_sgd_rec').get('server_lr')
            if params.get('compressor').get('type') == 'qsgd':
                self.server_lr = params.get('qsgd').get('server_lr')
            if params.get('compressor').get('type') == 'qsgd_rec':
                self.server_lr = params.get('qsgd_rec').get('server_lr')
        else:
            self.server_lr = params.get('fedavg').get('server_lr')

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
        self.layer_id = []
        i = 0
        for name, layer in self.global_model.named_parameters():
            self.layer_id.append(i)
            i += 1

    def __repr__(self) -> str:
        return "Dense Training"

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
        for idx, client in enumerate(clients):
            fit_configurations.append(
                    (client, FitIns(parameters, {'iter_num': server_round}))
                )
        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        deltas_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        rate_results = [fit_res.metrics['Rate'] for _, fit_res in results]
        log_metrics = {
            "Rate": [np.mean(rate_results)]
        }
        for _, fit_res in results:
            for name, val in fit_res.metrics.items():
                if log_metrics.get(name) is None:
                    log_metrics[name] = [val]
                else:
                    log_metrics[name].append(val)
        for name, val in log_metrics.items():
            log_metrics[name] = np.mean(val)

        deltas_aggregated = aggregate(deltas_results)
        updated_params = []

        with torch.no_grad():
            print('len of deltas_aggregated:{}' .format(len(deltas_aggregated)))
            for k, (name, param) in enumerate(self.global_model.named_parameters()):
                print('layer name:{}' .format(name))
                print('idx of layer:{}' .format(k))
                updated_params.append(param.cpu().numpy() + (self.server_lr * deltas_aggregated[k]))

        if self.sim_folder:
            self.wandb_runner.log(log_metrics)
        return ndarrays_to_parameters(updated_params), {}

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
        # self.set_parameters(parameters)
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