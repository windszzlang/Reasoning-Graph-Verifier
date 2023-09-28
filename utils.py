import json
import os
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(data_path, load_part=None):
    data = []
    num = 0
    with open(data_path) as f:
        for line in f.readlines():
            num += 1
            if load_part != None and num > load_part:
                break
            data.append(json.loads(line))
    return data


def compute_accuracy_score(gold, pred):
    crt = 0.
    for g, p in zip(gold, pred):
        if g == p:
            crt += 1
    return crt / len(gold)


def compute_graph_judge_score(gold, pred):
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred)
    precision = precision_score(gold, pred)
    recall = recall_score(gold, pred)
    return acc, f1, precision, recall