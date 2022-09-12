import sys
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import get_dataloaders

from utils.checkpoint import save
from utils.loops import train, evaluate
from utils.setup_network import setup_network, setup_hparams

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(net, hps, logger):
    trainloader, valloader, testloader = get_dataloaders(bs=hps['bs'])

    net = net.to(device)

    learning_rate = float(hps['lr'])

    scaler = GradScaler()

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("Training", hps['vgg_type'], "on", device)

    f = open(hps['model_save_dir'] + '/log.txt', "a")
    f.write('VGG type: ' + hps['vgg_type'] + '\n')
    f.close()

    for epoch in range(hps['start_epoch'], hps['n_epochs']):

        acc_tr, loss_tr = train(net, trainloader, loss_fn, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, loss_fn)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate
        scheduler.step(acc_v)

        if acc_v > best_acc:
            best_acc = acc_v
            logger.save_plt(hps)

        if (epoch + 1) % hps['save_freq'] == 0:
            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        message = 'Epoch %2d' % (epoch + 1) + '\t' \
                  + 'Train Accuracy: %2.4f %%' % acc_tr + '\t' \
                  + 'Val Accuracy: %2.4f %%' % acc_v

        print(message)
        f = open(hps['model_save_dir'] + '/log.txt', "a")
        f.write(message + '\n')
        f.close()

    acc_test, loss_test = evaluate(net, testloader, loss_fn)

    result_message = 'Test Accuracy: %2.4f %%' % acc_test + '\n' \
                     + 'Test Loss: %2.6f' % loss_test

    print(result_message)

    f = open(hps['model_save_dir'] + '/log.txt', "a")
    f.write('\n' + result_message)
    f.close()


if __name__ == "__main__":
    hps = setup_hparams(sys.argv[1:])
    net, logger = setup_network(hps)

    main(net, hps, logger)
