import numpy as np
import torch
from typing import Dict
import torch.nn as nn
from torch.utils.data import dataset


def train_fedpm(
        model: nn.Module,
        trainloader: dataset,
        iter_num: int,
        device: torch.device,
        params: Dict,
        loss_fn=nn.CrossEntropyLoss(reduction='mean'),
        optimizer: torch.optim.Optimizer = None
) -> Dict:
    """
        Compute local epochs, the training strategies depends on the adopted model.
    """
    loss = None
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('fedpm').get('local_lr'))

    for epoch in range(params.get('fedpm').get('local_epochs')):
        running_loss = 0
        total = 0
        correct = 0
        for batch_idx, (train_x, train_y) in enumerate(trainloader):
            train_x, train_y = train_x.to(device), train_y.to(device)
            total += train_x.size(0)
            optimizer.zero_grad()
            y_pred = model(train_x)
            loss = loss_fn(y_pred, train_y)
            running_loss += loss.item()
            _, pred_y = torch.max(y_pred.data, 1)
            correct += (pred_y == train_y).sum().item()
            loss.backward()
            optimizer.step()
        # print("Epoch {}: train loss {}  -  Accuracy {}".format(epoch + 1, loss, correct/total))
    return model.state_dict()
