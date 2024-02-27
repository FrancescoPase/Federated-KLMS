from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl

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

from ..utils.load_model import load_model
from ..utils.models import get_parameters, set_parameters


class Client(fl.client.Client):

    def __init__(
            self,
            params: Dict,
            client_id: int,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            device='cpu',
            compressor=None
    ) -> None:

        self.params = params
        self.client_id = client_id
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.device = device
        self.local_model = load_model(self.params).to(self.device)
        self.compressor = compressor
        self.compression = True if self.compressor is not None else False

    def get_parameters(
            self,
            ins: GetParametersIns = None
    ) -> GetParametersRes:
        ndarrays = get_parameters(self.local_model)
        parameters = ndarrays_to_parameters(ndarrays)
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )

    def set_parameters(self, parameters):
        set_parameters(
            self.local_model,
            parameters_to_ndarrays(parameters)
        )

    def fit(
            self,
            ins: FitIns
    ) -> FitRes:
        self.set_parameters(ins.parameters)
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            parameters=self.get_parameters().parameters,
            num_examples=len(self.train_data_loader),
            metrics={},
            status=status
        )

    def evaluate(
            self,
            ins: EvaluateIns
    ) -> EvaluateRes:
        self.set_parameters(ins.parameters)
        loss, accuracy = 0, 0
        status = Status(Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.test_data_loader),
            metrics={"accuracy": float(0)}
        )

