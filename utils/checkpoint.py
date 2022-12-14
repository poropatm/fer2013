import os

import torch


def save(net, logger, hps, epoch):
    # Path gdje ce se spremati checkpoints
    path = os.path.join(hps['model_save_dir'], 'checkpoints', 'epoch_' + str(epoch))

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict()
    }

    torch.save(checkpoint, path)


def restore(net, logger, hps):
    path = os.path.join(hps['model_save_dir'], 'checkpoints', 'epoch_' + str(hps['restore_epoch']))

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            logger.restore_logs(checkpoint['logs'])
            net.load_state_dict(checkpoint['params'])
            print("Network Restored!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            hps['start_epoch'] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        hps['start_epoch'] = 0


def load_features(model, params):
    model_dict = model.state_dict()

    imp_params = {k: v for k, v in params.items() if k in model_dict}

    model_dict.update(imp_params)
    model.load_state_dict(imp_params)

    for name, param in model.named_parameters():
        param.requires_grad = False
