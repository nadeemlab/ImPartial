
import numpy as np
import torch
import json
from pathlib import Path

def to_np(t):
    return t.cpu().detach().numpy()

def random_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_params_save(filename,classifier_network, optimizer, save_optimizer = False):
    if save_optimizer:
        torch.save([classifier_network.state_dict(),optimizer.state_dict()], filename)
    else:
        torch.save([classifier_network.state_dict()], filename)

def model_params_load(filename,classifier_network, optimizer, DEVICE):
    saves_model = torch.load(filename, map_location=DEVICE)
    classifier_network.load_state_dict(saves_model[0])
    if (len(saves_model) > 1) & (optimizer is not None):
        optimizer.load_state_dict(saves_model[1])

def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)

def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

class early_stopping(object):
    def __init__(self, patience, counter, best_loss):
        self.patience = patience #max number of nonimprovements until stop
        self.counter = counter #number of consecutive nonimprovements
        self.best_loss = best_loss

    def evaluate(self, loss):
        save = False #save nw
        stop = False #stop training
        if loss < 0.999*self.best_loss:
            self.counter = 0
            self.best_loss = loss
            save = True
            stop = False
        else:
            self.counter += 1
            if self.counter > self.patience:
                stop = True

        return save, stop

class TravellingMean:
    def __init__(self):
        self.count = 0
        self._mean= 0

    @property
    def mean(self):
        return self._mean

    def update(self, val, mass=None):
        if mass is None:
            mass = val.shape[0]
        self.count+=mass
        self._mean += ((np.mean(val)-self._mean)*mass)/self.count

    def __str__(self):
        return '{:.3f}'.format(self._mean)

    def __repr__(self):
        return '{:.3f}'.format(self._mean)