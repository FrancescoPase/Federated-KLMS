import copy
from collections import OrderedDict

from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from scipy.stats import bernoulli
from logging import DEBUG, INFO
from scipy.stats import entropy

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

from flower_fl.trainers.train_fedpm import train_fedpm
from flower_fl.clients.client import Client


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class FedPMClient(Client):
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
        self.epsilon = 0.01

    def sample_mask(self, mask_probs: Dict) -> List[np.ndarray]:
        sampled_mask = []
        with torch.no_grad():
            for layer_name, layer in mask_probs.items():
                if 'mask' in layer_name:
                    theta = torch.sigmoid(layer).cpu().numpy()
                    updates_s = bernoulli.rvs(theta)
                    updates_s = np.where(updates_s == 0, self.epsilon, updates_s)
                    updates_s = np.where(updates_s == 1, 1 - self.epsilon, updates_s)
                    sampled_mask.append(updates_s)
                else:
                    sampled_mask.append(layer.cpu().numpy())

        return sampled_mask

    def fit(
            self,
            fitins: FitIns
    ) -> FitRes:
        prior = parameters_to_ndarrays(fitins.parameters)
        self.set_parameters(fitins.parameters)
        mask_probs = train_fedpm(
            model=self.local_model,
            trainloader=self.train_data_loader,
            iter_num=fitins.config.get('iter_num'),
            device=self.device,
            params=self.params,
        )
        sampled_mask = None
        round_kl = []
        round_rate = []
        round_block_sizes = []
        rhos_round = []
        sampled_mask = []
        posterior_params = []
        prior_params = []
        if self.compression:
            for i, (name, param) in enumerate(mask_probs.items()):
                if 'mask' in name:
                    posterior_params.extend(sigmoid(param.cpu().numpy().flatten()))
                    prior_params.extend(sigmoid(prior[i]).flatten())
        else:
            sampled_mask = self.sample_mask(mask_probs)

        prior_params = np.asarray(prior_params)
        posterior_params = np.asarray(posterior_params)
        if self.compression:
            sampled_params, block_sizes, bits, rhos, ids = self.compressor.compress(
                posterior_update=np.float64(posterior_params),
                prior=np.float64(prior_params),
                compress_config=self.params.get('compressor').get('rec'),
                old_ids=fitins.config.get('old_ids')
            )
            # sampled_mask.append(sampled_params)
            j_start = 0
            for i, (name, param) in enumerate(mask_probs.items()):
                if 'mask' in name:
                    sampled_mask.append(np.reshape(sampled_params[j_start: j_start + torch.numel(param)],
                                                   param.shape))
                    j_start += torch.numel(param)
                else:
                    sampled_mask.append(param.cpu().numpy())
            round_kl.append(
                np.mean(
                    posterior_params * np.log2(posterior_params / prior_params) +
                    (1 - posterior_params) * np.log2((1 - posterior_params) / (1 - prior_params))
                ))
            round_block_sizes.append(np.mean(block_sizes))
            if fitins.config.get('old_ids') is None and \
                    self.params.get('compressor').get('rec').get('adaptive'):
                round_rate.append(np.mean((np.asarray(bits) + np.log2(256)) / np.asarray(block_sizes)))
            else:
                round_rate.append(np.mean(np.asarray(bits) / np.asarray(block_sizes)))
            rhos_round.extend(rhos)

        parameters = ndarrays_to_parameters(sampled_mask)
        status = Status(code=Code.OK, message="Success")
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
        # metrics.update(entropy_dict)
        return FitRes(
            status=status,
            parameters=parameters,
            num_examples=len(self.train_data_loader),
            metrics=metrics
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



