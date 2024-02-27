import copy
import torch
import torch.nn as nn
from typing import Dict
from torch.utils.data import dataset


def train_dense_classification(
        model: nn.Module,
        trainloader: dataset,
        iter_num: int,
        device: torch.device,
        params: Dict,
        loss_fn=nn.CrossEntropyLoss(reduction='mean'),
        optimizer: torch.optim.Optimizer = None
) -> None:
    """Train the network on the training set."""
    if params.get('compressor').get('compress'):
        if params.get('compressor').get('type') == 'sign_sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=params.get('sign_sgd').get('local_lr'))
        if params.get('compressor').get('type') == 'sign_sgd_rec':
            optimizer = torch.optim.SGD(model.parameters(), lr=params.get('sign_sgd').get('local_lr'))
        if params.get('compressor').get('type') == 'qsgd':
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('qsgd').get('local_lr'))
        if params.get('compressor').get('type') == 'qsgd_rec':
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('qsgd_rec').get('local_lr'))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('fedavg').get('local_lr'))

    global_model = copy.deepcopy(model.state_dict())
    model.train()
    for epoch in range(params.get('dense').get('local_epochs')):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    full_grad = dict()
    with torch.no_grad():
        for k, (name, param) in enumerate(model.state_dict().items()):
            full_grad[name] = param - global_model[name]
    return full_grad


def test_dense_classfication(
        model: nn.Module,
        testloader: dataset,
        device: torch.device
):

    """Evaluate the network on the entire test set."""

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
