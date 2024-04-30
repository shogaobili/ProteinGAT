import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 读取文件
data = np.genfromtxt('predictions/11-30/7400.txt', dtype=str)

# 获取预测结果和正确标签
predictions = data[:, 1]
labels = data[:, 2]

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
    "XXX": "X",
}
predictions = [PROTEINLETTER3TO1[prediction] for prediction in predictions]
labels = [PROTEINLETTER3TO1[label] for label in labels]

# 创建混淆矩阵
cm = confusion_matrix(labels, predictions)

# 计算每个类别的概率
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 获取类别列表
classes = np.unique(np.concatenate((labels, predictions)))

# 绘制混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap='viridis', xticklabels=classes, yticklabels=classes, fmt='.2f')

# 设置图像参数
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# 保存图像
plt.savefig('confusion_matrix.png')

# 显示图像
plt.show()