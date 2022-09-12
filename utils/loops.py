import warnings

import torch
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, trainloader, loss_fn, optimizer, scaler, Ncrop=True):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast():
            if Ncrop:
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

            labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_tr += loss.item()

            _, predictions = torch.max(outputs.data, 1)
            correct_count += (predictions == labels).sum().item()
            n_samples += labels.size(0)

    accuracy = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return accuracy, loss


def evaluate(net, dataloader, loss_fn, Ncrop=True):
    net = net.eval()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        if Ncrop:
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            outputs = net(inputs)

            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops
        else:
            outputs = net(inputs)

        loss = loss_fn(outputs, labels)

        loss_tr += loss.item()

        _, predictions = torch.max(outputs.data, 1)
        correct_count += (predictions == labels).sum().item()
        n_samples += labels.size(0)

    accuracy = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return accuracy, loss
