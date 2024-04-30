import array
from datetime import datetime
import time
from torch.utils.data import DataLoader, Dataset
import numpy as np
from models import GAT_skip_forward as GAT_model
from sklearn.metrics import  confusion_matrix
from protein_dataset import ProteinDataset
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from protein_utils import ENCODEAA2NUM, PROTEINLETTER3TO1
import seaborn as sns
import matplotlib.pyplot as plt

config = yaml.load(open('test.yaml', 'r'), Loader=yaml.FullLoader)
#dataset = ProteinDataset(data_dir='dataset_test')
cuda_available = torch.cuda.is_available()

from utils import debug_log_test

def Debug_Log(text):
    debug_log_test(text, config['log_file_path'] + config['log_file_name'])

def init_model():
    global model, optimizer, dataset
    dataset = ProteinDataset(data_dir=config['test_jsonfile_path'])
    print(len(dataset))
    time.sleep(10)
    model = GAT_model(nfeat=dataset.n_feat,
            nhid=config['hidden'],
            nclass=36,
            dropout=config['dropout'],
            nheads=config['nb_heads'],
            alpha=config['alpha'],
            n_hidden_layers=config['n_hidden_layers'],
            use_bn=config['use_bn'],
            n_gatconvs=10
            )
    optimizer = optim.Adam(model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay'])
    if cuda_available:
        model = model.cuda()

def process_data(data):
    _labels = [d[0] for d in data]
    _features = np.array([d[1] for d in data])
    _name = np.array([d[2] for d in data])
    _neighbours = np.array([d[3] for d in data])
    _edges = np.empty((0, 2), dtype=int)
    
    for i in range(len(data)):
        _edges = np.append(_edges, [[i, i]], axis=0)
        for neighbour_name in _neighbours[i][:20]:
            neighbour_idx = np.where(_name == neighbour_name)
            if len(neighbour_idx) == 0:
                continue
            if neighbour_idx[0].shape[0] != 1:
                continue
            _edges = np.append(_edges, [[int(neighbour_idx[0]), i]], axis=0)

    _edges = torch.tensor(_edges, dtype=torch.long).t()
    _edges = _edges.cuda()
    _features = torch.FloatTensor(np.array(_features))
    # _labels = torch.LongTensor(np.array(_labels))
    if cuda_available:
        _features = _features.cuda()
        # _labels = _labels.cuda()
    return _features, _edges, _labels

def test():
    global confusion_matr
    global correct_all
    correct_all=0
    confusion_matr=np.zeros((20, 20), dtype=int)
    model.eval()
    #load parameters from .pkl file
    pkl_file_path = config['pkl_file_path']
    model.load_state_dict(torch.load(pkl_file_path))
    test_loss = 0
    correct = 0
    count=int(len(dataset)/3000)
    res=len(dataset)-3000*count
    print("count是"+str(count))
    for i in range(count+1):    
        torch.cuda.empty_cache()
        print("第"+str(i+1)+"批数据")
        if (i==count):
            test_data = [dataset[j] for j in range(i*3000,i*3000+res)]
        else:
            test_data = [dataset[j] for j in range(i*3000,i*3000+3000)]
        features, edges, labels = process_data(test_data)
        print(len(features))
        print(len(edges))
        output = model(features, edges)
        label_logits = [d['logits'] for d in labels]
        label_logits = torch.LongTensor(label_logits)
        label_logits = label_logits.cuda()
        output_logits = output[:, :21]
        pred = output_logits.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        for i in range(len(pred)):
            # reverse ENCODEAA2NUM with all keys += 1
            pred_AA_name = list(ENCODEAA2NUM.keys())[list(ENCODEAA2NUM.values()).index(pred[i].item() - 1)]
            # reverse PROTEINLETTER3TO1
            pred_AA_full_name = list(PROTEINLETTER3TO1.keys())[list(PROTEINLETTER3TO1.values()).index(pred_AA_name)]

            # same for label_logits
            label_AA_name = list(ENCODEAA2NUM.keys())[list(ENCODEAA2NUM.values()).index(label_logits[i].item() - 1)]
            label_AA_full_name = list(PROTEINLETTER3TO1.keys())[list(PROTEINLETTER3TO1.values()).index(label_AA_name)]

            # debug_log_test(str(i) + " " + str(pred[i].item()) + " " + str(label_logits[i].item()) + "\n", config['log_file_path'] + config['log_file_name'])
            Debug_Log(str(i) + " " + str(pred_AA_full_name) + " " + str(label_AA_full_name) + "\n")
        confusion = confusion_matrix(label_logits.cpu().numpy(), pred.cpu().numpy())
        confusion_matr+=confusion
        correct += pred.eq(label_logits.view_as(pred)).sum().item()
        #test_loss /= len(dataset)
    print('Accuracy: {}/{} ({:.0f}%)'.format(correct, len(dataset), 100. * correct/len(dataset)))
    


def heatmap(confusion_matr):

#     ENCODEAA2NUM1 = {
#     "A": 1,
#     "C": 2,
#     "D": 3,
#     "E": 4,
#     "F": 5,
#     "G": 6,
#     "H": 7,
#     "I": 8,
#     "K": 9,
#     "L": 10,
#     "M": 11,
#     "N": 12,
#     "P": 13,
#     "Q": 14,
#     "R": 15,
#     "S": 16,
#     "T": 17,
#     "V": 18,
#     "W": 19,
#     "Y": 20,
    
# }
    # ENCODEAA2NUM1 = {
    #     "H": 7,
    #     "K": 9,
    #     "R": 15,
    #     "D": 3,
    #     "E": 4,
    #     "S": 16,
    #     "T": 17,
    #     "N": 12,
    #     "Q": 14,
    #     "A": 1,
    #     "V": 18,
    #     "L": 10,
    #     "I": 8,
    #     "M": 11,
    #     "F": 5,
    #     "Y": 20,
    #     "W": 19,
    #     "P": 13,
    #     "G": 6,
    #     "C": 2
    # }


    # NUM2ENCODEAA = {value: key for key, value in ENCODEAA2NUM1.items()}
    PROTEINLETTER3TO1 = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }
# PROTEINLETTER1TO3 = {value: key for key, value in PROTEINLETTER3TO1.items()}
    # confusion_matr_swapped = confusion_matr[np.ix_([ENCODEAA2NUM1[col]-1 for col in NUM2ENCODEAA.values()], [ENCODEAA2NUM1[row]-1 for row in NUM2ENCODEAA.values()])]
    # print("confusion_matr_swapped:",confusion_matr_swapped)
# 设置横纵坐标标签
    labels = [value for value in PROTEINLETTER3TO1.values()]
   # print(labels)
# 创建热力图
    plt.figure(figsize=(10, 8))
    # confusion_matr_normalized = confusion_matr / confusion_matr.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(confusion_matr / confusion_matr.sum(axis=1)[:, np.newaxis], annot=True, fmt='.2f', cmap='viridis', linewidths=0,vmax=1)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    # for i in range(20):
    #     for j in range(20):
    #         ax.text(i+0.5, j+0.5, str(confusion_matr[i][j]), ha='center', va='center', fontsize=10)
# 设置标签
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix Heatmap')
    
# 显示热力图
    plt.show()
    plt.savefig("heatmap.png")
# create if log file path does not exist
# if not os.path.exists(config['log_file_path']):
#     os.makedirs(config['log_file_path'])
# # create if log file does not exist
# if not os.path.exists(config['log_file_path'] + config['log_file_name']):
#     with open(config['log_file_path'] + config['log_file_name'], 'w') as f:
#         f.write('')
        

init_model()
test()
print("Confusion Matrix:",confusion_matr)
heatmap(confusion_matr)