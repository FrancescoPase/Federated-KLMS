import copy

import numpy as np
import torch
from typing import Dict
import torch.nn as nn
from torch.utils.data import dataset

from flower_fl.compressors.compressor import Compressor


def train_lsd(
        model: nn.Module,
        trainloader: dataset,
        iter_num: int,
        device: torch.device,
        params: Dict,
        memory_term: Dict,
        loss_fn=nn.CrossEntropyLoss(reduction='mean'),
        model_svrg: nn.Module = None,
) -> Dict:

    weight_decay = params.get('weight_decay')
    svrg_epoch = params.get('svrg_epoch')
    local_dataset_size = len(trainloader.dataset)
    local_epochs = params.get('local_epochs')
    grad_svrg = None

    model.train()

    # Use Variance-Reduce variant
    if svrg_epoch == 0:
        svrg_epoch = np.infty
    else:
        assert model_svrg is not None
        grad_svrg = dict()
        for name, param in model.named_parameters():
            grad_svrg[name] = torch.zeros_like(param)

    model.zero_grad()

    for epoch in range(local_epochs):
        inputs, labels = next(iter(trainloader))
        inputs, labels = inputs.to(device), labels.to(device)

        if iter_num % svrg_epoch != 0 and svrg_epoch != np.infty:
            model_svrg.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        # Regularization term
        for param in model.parameters():
            loss += weight_decay / local_dataset_size * torch.norm(param) ** 2

        # compute the gradient of the local model
        loss.backward()
        # compute the loss associated with the net_svrg
        if iter_num % svrg_epoch != 0 and svrg_epoch != np.infty:
            loss_svrg = loss_fn(model_svrg(inputs), labels)
            for param in model_svrg.parameters():
                # TODO Why factor 2 ??
                loss_svrg += weight_decay / (2 * local_dataset_size) * torch.norm(param) ** 2
            loss_svrg.backward()

    full_grad = dict()
    with torch.no_grad():
        for k, (name, param) in enumerate(model.named_parameters()):
            if iter_num % svrg_epoch == 0 and svrg_epoch != np.infty:
                grad_svrg[name] = local_dataset_size * param.grad.data
                full_grad[name] = grad_svrg[name] - memory_term[name]
            elif svrg_epoch != np.infty:
                g_svrg = list(model_svrg.parameters())[k].grad.data
                full_grad[name] = local_dataset_size * (param.grad.data - g_svrg) + \
                                  grad_svrg[name] - memory_term[name]
            else:
                full_grad[name] = local_dataset_size * param.grad.data - memory_term[name]

    return full_grad

