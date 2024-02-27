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

from flower_fl.trainers.train_dense_classification import train_dense_classification
from flower_fl.clients.client import Client
from flower_fl.utils.load_compressor import get_compressor


class DenseClient(Client):
    def __init__(
            self,
            params: Dict,
            client_id: int,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            device='cpu',
    ) -> None:

        super().__init__(
            params=params,
            client_id=client_id,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            device=device
        )
        self.compressor = None
        self.compression = self.params.get('compressor').get('compress')
        if self.compression:
            self.compressor = get_compressor(
                compressor_type=self.params.get('compressor').get('type'),
                params=self.params,
                device=self.device
            )


    def fit(
            self,
            fitins: FitIns
    ) -> FitRes:
        self.set_parameters(fitins.parameters)
        deltas = train_dense_classification(
            model=self.local_model,
            trainloader=self.train_data_loader,
            iter_num=fitins.config.get('iter_num'),
            device=self.device,
            params=self.params,
        )

        if self.compression:
            compressed_delta, avg_bitrate = self.compressor.compress(
                updates=deltas,
                compress_config=self.params.get('compressor').get('rec'),
                iter_num=fitins.config.get('iter_num')
            )
            round_rate = avg_bitrate
        else:
            compressed_delta = []
            for i, (name, param) in enumerate(deltas.items()):
                compressed_delta.append(param.cpu().numpy())
            round_rate = 32
        parameters = ndarrays_to_parameters(compressed_delta)
        status = Status(code=Code.OK, message="Success")
        metrics = {
            "Rate": np.mean(round_rate),
        }
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



