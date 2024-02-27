from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg, Strategy
import flwr as fl


class Server(fl.server.Server):

    def __init__(
            self,
            params: Dict,
            client_manager: ClientManager,
            strategy: Optional[Strategy] = None
    ) -> None:
        self.params = params
        self.client_manager = client_manager
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
