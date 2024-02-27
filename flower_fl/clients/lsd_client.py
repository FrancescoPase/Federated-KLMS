from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import copy
from logging import DEBUG, INFO

from flwr.common.logger import log

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    GetParametersIns,
    GetParametersRes,
    EvaluateIns,
    EvaluateRes
)

from flower_fl.trainers.train_lsd import train_lsd
from flower_fl.clients.client import Client
from flower_fl.utils.load_compressor import get_compressor


class LSDClient(Client):
    def __init__(
            self,
            params: Dict,
            client_id: int,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            device='cpu',
            compressor=None
    ) -> None:

        super().__init__(
            params=params,
            client_id=client_id,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            device=device,
            compressor=compressor
        )
        self.compressor = compressor
        self.weight_decay = self.params.get('lsd').get('weight_decay')
        self.svrg_epoch = self.params.get('lsd').get('svrg_epoch')
        if self.svrg_epoch == 0:
            self.svrg_epoch = np.infty
        self.compression_parameter = self.params.get('compressor').get('compress')
        """if self.compression_parameter:
            self.compressor = get_compressor(
                compressor_type=self.params.get('compressor').get('type'),
                params=self.params,
                device=self.device
            )"""
        model_state = copy.deepcopy(self.local_model.state_dict())

        self.model_svrg = None
        if self.svrg_epoch != np.infty:
            # create the network model_svrg
            self.model_svrg = copy.deepcopy(self.local_model)
            self.model_svrg.to(self.device)
            # initialize the variance reduction parameter
            self.grad_svrg = dict()
            for name, param in model_state.items():
                self.grad_svrg[name] = torch.zeros_like(param)
        # Define the memory term
        self.memory_term = dict()
        # initialize the memory term
        self.use_memory_term = self.params.get('lsd').get('use_memory_term')
        for name, param in model_state.items():
            self.memory_term[name] = param if self.use_memory_term else torch.zeros_like(param)
        # Define the memory rates : if memory_rate = 0, there is no memory term
        self.memory_rate = dict()

        for name, param in model_state.items():
            if self.use_memory_term and self.compression_parameter > 0:
                alpha = np.sqrt(len(torch.flatten(param))) / self.compression_parameter
                self.memory_rate[name] = 1 / (min(alpha, alpha ** 2) + 1)
            else:
                self.memory_rate[name] = 0

    def fit(
            self,
            fitins: FitIns
    ) -> FitRes:
        self.set_parameters(fitins.parameters)
        full_grad = train_lsd(
            model=self.local_model,
            trainloader=self.train_data_loader,
            iter_num=fitins.config.get('iter_num'),
            device=self.device,
            params=self.params.get('lsd'),
            memory_term=self.memory_term,
            model_svrg=self.model_svrg
        )
        round_kl = []
        round_rate = []
        round_block_sizes = []
        rhos_round = []
        sampled_mask = []
        flatten_grad = []
        grad_sum = []

        for name, param in full_grad.items():
            if self.compression_parameter:
                flatten_grad.extend(param.cpu().numpy().flatten())
            else:
                grad_sum.append(full_grad[name].cpu().numpy())
        flatten_grad = np.asarray(flatten_grad)
        if self.compression:
            delta_grad, block_sizes, bits, rhos, ids = self.compressor.compress(
                flatten_grad,
                compress_config=self.params,
                old_ids=fitins.config.get('old_ids')
            )
            j_start = 0
            for i, (name, param) in enumerate(full_grad.items()):

                new_grad = np.reshape(delta_grad[j_start: j_start + torch.numel(param)], param.shape)
                grad_sum.append(new_grad + self.memory_term[name].cpu().numpy())
                j_start += torch.numel(param)
                # update the memory term
                self.memory_term[name] += self.memory_rate[name] * torch.Tensor(new_grad).to(self.device)
            sampled_clients = int(self.params.get('simulation').get('fraction_fit') * self.params.get('simulation').get('n_clients'))
            sigma = 2 / (self.params.get('lsd').get('server_lr') * sampled_clients**2)
            k = len(flatten_grad)
            round_kl.append(
                (0.5 * (-k + (1/sigma) * np.sum(flatten_grad**2) + k*sigma - k * np.log2(sigma))) / len(delta_grad)
                )
            round_block_sizes.append(np.mean(block_sizes))
            if fitins.config.get('old_ids') is None and \
                    self.params.get('compressor').get('rec').get('adaptive'):
                round_rate.append(np.mean((np.asarray(bits) + np.log2(256)) / np.asarray(block_sizes)))
            else:
                round_rate.append(np.mean(np.asarray(bits) / np.asarray(block_sizes)))
            rhos_round.extend(rhos)

        rhos_round = np.asarray(rhos_round)
        rhos_round[rhos_round > 100] = 100
        rhos_round = list(rhos_round)
        metrics = {
            "KL Divergence": np.mean(round_kl),
            "Block Size": np.mean(round_block_sizes),
            "Rate": np.mean(round_rate)
        }
        if 'rec' in self.params.get('compressor').get('type'):
            metrics.update({
                "Ids": ids,
                "Rho Mean": np.mean(rhos_round),
                "Rho Std": np.std(rhos_round)
            })

        if fitins.config.get('iter_num') % self.svrg_epoch == 0 and self.svrg_epoch != np.infty:
            self.model_svrg = copy.deepcopy(self.local_model)

        parameters = ndarrays_to_parameters(grad_sum)
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_data_loader),
            metrics=metrics,
        )

    def evaluate(
            self,
            ins: EvaluateIns
    ) -> EvaluateRes:

        loss_fn = torch.nn.CrossEntropyLoss()
        self.set_parameters(ins.parameters)
        loss, accuracy = 0, 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_id, (inputs, labels) in enumerate(self.test_data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss += loss_fn(outputs, labels)
                correct += (predicted == labels).sum().item()
                total += labels.shape[0]
            accuracy = 100 * correct / total
        status = Status(Code.OK, message="Success")
        log(
            INFO,
            "Client %s Accuracy: %f   Loss: %f",
            self.client_id,
            accuracy,
            loss
        )
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.test_data_loader),
            metrics={"accuracy": float(accuracy),
                     "loss": float(loss)}
        )
