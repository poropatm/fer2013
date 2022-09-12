import os
from datetime import datetime

from models import vgg
from utils.checkpoint import restore
from utils.logger import Logger

# parametri koji se unose
hps = {
    'vgg_type': 'VGG11',
    'name': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    'n_epochs': 5,
    'model_save_dir': None,  # gdje će se spremati rezultati
    'restore_epoch': None,
    'start_epoch': 0,
    'save_freq': 20,
    'lr': 0.01,
    'drop': 0.1,
    'bs': 64,
}

vgg_types = {'VGG11', 'VGG13', 'VGG16', 'VGG19'}


# Provjere nepravilnih unosa
def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    if hps['vgg_type'] not in vgg_types:
        raise ValueError("Invalid vgg type.\nPossible ones include:\n - " + '\n - '.join(vgg_types))

    try:
        hps['n_epochs'] = int(hps['n_epochs'])
        hps['start_epoch'] = int(hps['start_epoch'])
        hps['save_freq'] = int(hps['save_freq'])
        hps['lr'] = float(hps['lr'])
        hps['drop'] = float(hps['drop'])
        hps['bs'] = int(hps['bs'])

        if hps['restore_epoch']:
            hps['restore_epoch'] = int(hps['restore_epoch'])
            hps['start_epoch'] = int(hps['restore_epoch'])

        if hps['n_epochs'] < 20:
            hps['save_freq'] = min(5, hps['n_epochs'])

    except Exception as e:
        raise ValueError("Invalid input parameters")

    hps['model_save_dir'] = os.path.join(os.getcwd(), 'results', hps['name'])

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])
        os.makedirs(os.path.join(hps['model_save_dir'], 'checkpoints'))

    return hps


def setup_network(hps):
    net = vgg.Vgg(hps['vgg_type'])
    logger = Logger()

    if hps['restore_epoch']:
        restore(net, logger, hps)

    return net, logger
