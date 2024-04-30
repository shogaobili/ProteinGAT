import numpy
import numpy as np
import scipy.sparse as sp
import torch
from Bio.PDB.Polypeptide import three_to_index
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

c_dict = []
predict_idx_dict = []

devicenumber = 0

def debug_log(text, logfile = "default log.txt"):
    with open(logfile,'a+') as logfile:
        logfile.writelines(text)

def debug_log_test(text, logfile = "default log test.txt"):
    with open(logfile,'a+') as logfile:
        logfile.writelines(text)

def encode_onehot(labels):
    global c_dict
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps sto index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[
        i, :] for i, c in enumerate(classes)}
    # print("classes_dict:")
    # print(classes_dict)
    c_dict = classes_dict
    print(c_dict)
    labels_onehot = np.array(
        list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def initialize_data(path="./data/dssp/", dataset="data"):
    print('[utils.py] Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt(
        "{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # labels = encode_onehot(idx_features_labels[:, -1])
    features_vector = idx_features_labels[:, 0:-1]
    labels_vector = idx_features_labels[:, -1]
    edges_unordered_vector = np.genfromtxt(
        "{}{}.cites".format(path, dataset), dtype=np.int32)
    result_edges = {}
    current_vec = []
    current_idx = -1
    for i in range(len(edges_unordered_vector)):
        if edges_unordered_vector[i][0] != current_idx:
            current_idx = edges_unordered_vector[i][0]
            result_edges[current_idx] = []
            current_vec = result_edges[current_idx]
        current_vec.append(edges_unordered_vector[i][1])
    return result_edges, features_vector, labels_vector


def load_data(edges_unordered_vector, features_vector, labels_vector, idx_train=range(5, 10), idx_val=range(10, 20),
              idx_test=range(20, 30)):
    print("idx_train:", idx_train)
    print("idx_val:", idx_val)
    print("idx_test:", idx_test)

    new_features_vector = np.append(np.append(features_vector[idx_train, 1:-1], features_vector[idx_val, 1:-1],
                                              axis=0), features_vector[idx_test, 1:-1], axis=0)
    new_labels_vector = np.append(np.append(labels_vector[idx_train], labels_vector[idx_val], axis=0),
                                  labels_vector[idx_test], axis=0)

    idx = np.array(np.append(np.append(features_vector[idx_train, 0], features_vector[idx_val, 0],
                                       axis=0), features_vector[idx_test, 0], axis=0), dtype=np.int32)
                                       
    dict_id_to_idx = {}
    dict_idx = []

    for i in range(idx.shape[0]):
        dict_id_to_idx[idx[i]] = i
        dict_idx.append(idx[i])

    new_edges_unordered_vector = np.ndarray(shape=(0, 2), dtype=np.int32)

    for i in range(idx.shape[0]):
        if idx[i] in edges_unordered_vector:
            for j in range(len(edges_unordered_vector[idx[i]])):
                if edges_unordered_vector[idx[i]][j] in dict_id_to_idx:
                    new_edges_unordered_vector = np.append(new_edges_unordered_vector,
                                                           np.array((idx[i],
                                                                     edges_unordered_vector[idx[i]][j]),
                                                                    dtype=np.int32).reshape(1,
                                                                                            2),
                                                           axis=0)

    features = sp.csr_matrix(new_features_vector, dtype=np.float32)
    labels = encode_onehot(new_labels_vector)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, new_edges_unordered_vector.flatten())), dtype=np.int32).reshape(
        new_edges_unordered_vector.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    # print("normalized:", features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    return adj, features, labels


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    print("[utils.py] predicts:")
    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    # print(preds)
    print(preds_cpu)
    print("[utils.py] labels:")
    # print(labels)
    print(labels_cpu)
    # debug_log("predicts:\n")
    # debug_log(str(preds) + "\n")
    # debug_log("labels:\n")
    # debug_log(str(labels) + "\n")
    # preds_cpu = preds.cpu().numpy()
    # labels_cpu = labels.cpu().numpy()

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_ss3_ss8_without_P(output, labels):
    preds = output.max(1)[1].type_as(labels)
    print("[utils.py] predicts:")
    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    
    print(preds_cpu)
    print("[utils.py] labels:")
    
    print(labels_cpu)
    # if labels_cpu[i] = 'P', ignore i
    correct = 0
    length = 0
    for i in range(len(labels_cpu)):
        if labels_cpu[i] != 8:
            length += 1
            if preds_cpu[i] == labels_cpu[i]:
                correct += 1
    if length == 0:
        return 0.0
    return float(correct / length)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    # return correct / len(labels)

def accuracy_without_VOXEL(output, labels):
    preds = output.max(1)[1].type_as(labels)
    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    print("[utils.py] predicts:")
    print(preds)
    print("[utils.py] labels:")
    print(labels)
    # calculate accuracy if label is not VOXEL
    correct = 0
    length = 0
    for i in range(len(labels_cpu)):
        if labels_cpu[i] != 20:
            length += 1
            if preds_cpu[i] == labels_cpu[i]:
                correct += 1
    if length == 0:
        return 0.0
    return float(correct / length)

def accuracy_predict(output, labels, pos):
    global c_dict
    print("[utils.py] accuracy_predicts:")


    preds = output.max(1)[1].type_as(labels)

    print('output:', preds)
    print('labels:', labels)

    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    xandylabels = []
    
    max_key = 0
    for key in c_dict:
        c_dict[key] = np.argmax(c_dict[key])
    c_dict = {value: key for key, value in c_dict.items()}
    
    for key in label_res_dict:
        xandylabels.append(label_res_dict[key])

    print(c_dict)
    for key in c_dict:
        max_key = max(max_key, key)

    # 将数字标签转换为对应的氨基酸名称
    preds_names = [c_dict[min(i, max_key)] for i in preds_cpu]
    labels_names = [c_dict[min(i, max_key)] for i in labels_cpu]

    print(preds_names)
    print(labels_names)

    print(pos)
    print(preds_names[pos])
    print(labels_names[pos])

    output_acc = []
    for i in output[pos]:
        output_acc.append(i.item())
    # TODO: return output[pos] (get the probability of the predict)
    return preds_names[pos], labels_names[pos], output_acc, c_dict 

def accuracy_test_draw(output, labels):
    global c_dict
    preds = output.max(1)[1].type_as(labels)
    preds_cpu = preds.cpu().numpy()
    labels_cpu = labels.cpu().numpy()

    xandylabels = []

    for key in c_dict:
        c_dict[key] = np.argmax(c_dict[key])
    c_dict = {value: key for key, value in c_dict.items()}
    for key in label_res_dict:
        xandylabels.append(label_res_dict[key])
    # 将数字标签转换为对应的氨基酸名称
    preds_names = [c_dict[min(i, 19)] for i in preds_cpu]
    labels_names = [c_dict[min(i, 19)] for i in labels_cpu]
    print(c_dict.values())
    print(preds_names)
    print(labels_names)
    # 生成混淆矩阵并可视化为热图
    cm = confusion_matrix(labels_names, preds_names,
                          labels=xandylabels)

    # 计算每个类别被正确预测的百分比
    cm_percentage = np.round(cm.astype('float') /
                             cm.sum(axis=1)[:, np.newaxis] * 100.0, decimals=2)
    
    # 计算总个数
    # cm_percentage = cm.astype('int')

    # 可视化热图
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(cm_percentage, annot=True, cmap='RdBu_r', fmt='g',
                ax=ax, xticklabels=xandylabels, yticklabels=xandylabels, annot_kws={"fontsize":10})
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig('heatmap.png')

    # 计算准确率
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# label_res_dict = {0: 'HIS', 1: 'LYS', 2: 'ARG', 3: 'ASP', 4: 'GLU', 5: 'SER', 6: 'THR', 7: 'ASN', 8: 'GLN', 9: 'ALA',
#                   10: 'VAL', 11: 'LEU', 12: 'ILE', 13: 'MET', 14: 'PHE', 15: 'TYR', 16: 'TRP', 17: 'PRO', 18: 'GLY',
#                   19: 'CYS', 20: 'VOXEL'}
# res_label_dict = {'HIS': 0, 'LYS': 1, 'ARG': 2, 'ASP': 3, 'GLU': 4, 'SER': 5, 'THR': 6, 'ASN': 7, 'GLN': 8, 'ALA': 9,
#                   'VAL': 10, 'LEU': 11, 'ILE': 12, 'MET': 13, 'PHE': 14, 'TYR': 15, 'TRP': 16, 'PRO': 17, 'GLY': 18,
#                   'CYS': 19, 'VOXEL': 20}
# res_group_dict = {
#     0: 'group1', 1: 'group1', 2: 'group1',
#     3: 'group2', 4: 'group2',
#     5: 'group3', 6: 'group3', 7: 'group3', 8: 'group3',
#     9: 'group4', 10: 'group4', 11: 'group4', 12: 'group4', 13: 'group4',
#     14: 'group5', 15: 'group5', 16: 'group5',
#     17: 'group6', 18: 'group6',
#     19: 'group7'
# }
#group1:碱性氨基酸
#group2:酸性氨基酸
#group3:极性中性氨基酸
#group4:非极性中性氨基酸
#group5:芳香环氨基酸
#group6:脯氨酸、甘氨酸
#group7:半胱氨酸

AA_NAMES = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
            "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]

aa_to_onehot = {aa: [int(i == three_to_index(aa))
                     for i in range(20)] for aa in AA_NAMES}


def accuracy_group(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = 0
    correct_heat = [[]]
    for i in range(9):
        for j in range(10):
            correct_heat[i].append(0)
        correct_heat.append([])
    correct_cnt = [0]
    for i in range(9):
        correct_cnt.append(0)
    print(correct_heat)
    print(correct_cnt)
    for i in range(len(preds)):
        if res_group_dict[int(preds[i])] == res_group_dict[int(labels[i])]:
            correct += 1
        correct_heat[int(labels[i])][int(preds[i])] += 1
        correct_cnt[int(labels[i])] += 1

    print(correct_heat)
    print(correct_cnt)
    correct = float(correct)
    correct = correct / len(preds)
    for i in range(10):
        for j in range(10):
            if (correct_cnt[i] == 0):
                continue
            print(float(correct_heat[i][j]) / float(correct_cnt[i]), end=' ')
        print()
    return correct
