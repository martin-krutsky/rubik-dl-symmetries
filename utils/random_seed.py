import random
import numpy as np
import torch


def seed_worker(worker_id: int):
    '''
    Set random seed for DataLoader initialization
    
    :param int worker_id: id given by DataLoader
    '''
    worker_seed = 123
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def seed_all(seed: int = 123):
    '''
    Set random seeds (all of them) for PyTorch training and eval loop
    
    :param int seed: random seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv3d):
        torch.nn.init.uniform_(m.weight)
        m.bias.data.fill_(0.01)    
    