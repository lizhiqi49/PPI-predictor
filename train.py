#%%

# lzq
# 2021.04.24
"""
训练模型，并输出中间文件:
    prob_score_mat.csv: 蛋白质互作概率矩阵 shape:[num_protein, num_protein]
    prob_scores.csv: 每次训练的模型在测试集上的概率值 shape:[num_edges, TIMES]
    logs.csv: 训练模型在训练过程中loss和accuracy值的记录

python train.py -h 查看参数说明
"""
import argparse

parser = argparse.ArgumentParser(description="protein interaction predictor")
parser.add_argument("--DEVICE", "-d", default='cpu', help="Device, cpu or cuda")
parser.add_argument("--NUM_FEATURE", "-f", type=int, default=5,
                    help="Length of node eigenvector")
parser.add_argument("--LEARNING_RATE", "-lr", type=float,
                    default=0.001, help="Learning rate")
parser.add_argument("--TIMES", "-t", default=10, type=int,
                    help="The times of selecting different groups of unlabelled edges to train")
parser.add_argument("--EPOCH", "-e", type=int, default=200, help="Epoch number")
parser.add_argument("--THRESHOLD", "-s", type=float, default=0.5,
                    help="The probility threshold of the existence of edge")
parser.add_argument("--DATASET", "-data", help="Dataset path")
parser.add_argument("--OUT_DIR", "-o", help="The directory of output")
args = parser.parse_args()

NUM_FEATURE = args.NUM_FEATURE
LEARNING_RATE = args.LEARNING_RATE
TIMES = args.TIMES
EPOCH = args.EPOCH
THRESHOLD = args.THRESHOLD
DATA_PATH = args.DATASET
OUT_DIR = args.OUT_DIR
DEVICE = args.DEVICE

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


dataset_df = pd.read_csv(DATA_PATH,sep='\t')
#dataset_df = pd.read_csv('./dataset/Scere20170205.txt',sep='\t')
"""
    shape:(22977, 18)
    interactor A: 4351
    interactor B: 4235
"""
#dataset_df = pd.read_csv('./dataset/Scere20170205CR.txt', sep='\t')
"""
    shape:(5470, 18)
    interactor A: 1971
    interactor B: 1946
"""
all_prot = np.unique(dataset_df.values[:,:2].reshape((-1)))
num_prot = len(all_prot)
# %%
A_df = pd.DataFrame(np.zeros((num_prot,num_prot),dtype=int), index=all_prot, columns=all_prot)
for interactor in dataset_df.values[:,:2]:
    intact_a = interactor[0]
    intact_b = interactor[1]
    A_df.loc[intact_a, intact_b] = 1
    A_df.loc[intact_b, intact_a] = 1
A = A_df.values
# 邻接矩阵


def edgeIndex(A):
    r"""
        Func: 获取边的索引
        args: 
            A: 邻接矩阵
        return:
            pos_edge_index (torch.tensor): shape:(2, num_pos_edges).
            neg_edge_index (torch.tensor): shape:(num_group, 2, num_neg_edges), 将unlabelled edge
            分为num_group组, 每组边数num_neg_edges=num_pos_edges.
    """
    n = len(A)
    pos_edge_index = []
    neg_edge_index = []
    
    for i in list(range(n)):
        for j in list(range(i,n)):
            if i != j:
                if A[i,j] == 1:
                    pos_edge_index.append([i,j])
                else:
                    neg_edge_index.append([i,j])
    num_pos_edges = len(pos_edge_index)    
    num_neg_edges = len(neg_edge_index)
    pos_edge_index = np.array(pos_edge_index).T
    neg_edge_index = np.array(neg_edge_index)

    # 将unlabelled edge随机排序并分组
    ## 重复记录的两个index应分到同一组
    rand_order = np.random.permutation(np.arange(num_neg_edges))
    if num_neg_edges % num_pos_edges == 0:  # 如果unlabelled edge是positive edge整数倍
        num_group = num_neg_edges / num_pos_edges
        rand_order_grouped = rand_order.reshape((num_group, num_pos_edges))
    else:
        remainder = int(num_neg_edges % num_pos_edges)
        num_group = int(num_neg_edges // num_pos_edges + 1)
        rand_order_grouped = rand_order[:-remainder].reshape((num_group-1, num_pos_edges))
        rand_order_grouped = np.vstack((rand_order_grouped, rand_order[-num_pos_edges:]))
    neg_edge_indexs = []
    for i in range(num_group):
        order = rand_order_grouped[i]
        neg_ind = neg_edge_index[order]
        neg_edge_indexs.append(neg_ind.T.tolist())
    return torch.tensor(pos_edge_index,dtype=torch.long), torch.tensor(neg_edge_indexs,dtype=torch.long)
    
pos_edge_index, neg_edge_indexs = edgeIndex(A)
pos_edge_index = torch.cat((pos_edge_index, pos_edge_index[[1,0]]),dim=-1)
neg_edge_indexs = torch.cat((neg_edge_indexs, neg_edge_indexs[:,[1,0]]),dim=-1)

# %%

NUM_NODE = num_prot
NUM_POS_EDGE = int(pos_edge_index.shape[1] / 2)
X = torch.ones((NUM_NODE, NUM_FEATURE),dtype=torch.float) / NUM_FEATURE
EDGE_ATTR = torch.cat((torch.ones(NUM_POS_EDGE*2),torch.zeros(NUM_POS_EDGE*2)),dim=-1)
SCORE_MAT = np.zeros((NUM_NODE,NUM_NODE))
print(SCORE_MAT.shape)

#%%

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(NUM_FEATURE, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.conv4 = GCNConv(32, 16)

    def forward(self, data):
        x, edge_index, pos_edge_index, edge_attr = data.x, data.edge_index, data.edge_index[:,torch.cat((data.train_pos_edge_index,data.test_pos_edge_index))], data.edge_attr
        x = self.conv1(x, pos_edge_index)
        x = F.relu(x)
        x = self.conv2(x, pos_edge_index)
        x = F.relu(x)
        x = self.conv3(x, pos_edge_index)
        x = F.relu(x)
        x = self.conv4(x, pos_edge_index)
        return x

    def decode(self, z, data, istrain=True):
        if istrain:
            index = torch.cat((data.train_pos_edge_index, data.train_neg_edge_index))
        else:
            index = torch.cat((data.test_pos_edge_index, data.test_neg_edge_index))

        edge_label = data.edge_attr[index]
        edge_index = data.edge_index[:,index]
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_label

    def decode_all(self, z):
        
        for i in range(NUM_NODE):
            col = torch.sum(z[i] * z, dim=-1).sigmoid().cpu().detach().numpy()
            col = col.reshape((-1,1))
            if i == 0:
                score_mat = col
            else:
                score_mat = np.hstack((score_mat, col))
        score_mat = (score_mat - np.min(score_mat)) / (np.max(score_mat) - np.min(score_mat))
        global SCORE_MAT
        SCORE_MAT += score_mat
        print(score_mat.shape)
        print(SCORE_MAT.shape)




def splitTrainTest(train_ratio=0.7):
    arr = np.arange(NUM_POS_EDGE * 4)
    rand_order = np.random.permutation(np.arange(NUM_POS_EDGE))
    train = rand_order[:int(NUM_POS_EDGE*train_ratio)]
    train_pos_edge_index = np.concatenate((train, NUM_POS_EDGE+train))
    train_neg_edge_index = np.concatenate((2*NUM_POS_EDGE+train, 3*NUM_POS_EDGE+train))
    test = rand_order[int(NUM_POS_EDGE*train_ratio):]
    test_pos_edge_index = np.concatenate((test, NUM_POS_EDGE+test))
    test_neg_edge_index = np.concatenate((2*NUM_POS_EDGE+test, 3*NUM_POS_EDGE+test))
    return torch.tensor(train_pos_edge_index,dtype=torch.long),\
        torch.tensor(train_neg_edge_index,dtype=torch.long),torch.tensor(test_pos_edge_index,dtype=torch.long),torch.tensor(test_neg_edge_index,dtype=torch.long)


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#lossf = torch.nn.binary_cross_entropy_with_logits().to(device)
device = torch.device(DEVICE)


prob_scores = []
M = np.zeros((EPOCH, 4))
for neg_edge_index in neg_edge_indexs[:TIMES]:
    edge_index = torch.cat((pos_edge_index, neg_edge_index),dim=-1)
    data = Data(x=X, edge_index=edge_index, edge_attr=EDGE_ATTR)

    data.train_pos_edge_index, data.train_neg_edge_index, data.test_pos_edge_index, data.test_neg_edge_index = splitTrainTest()

    model = Net().to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    metrics = []
    for epoch in range(EPOCH):
        metric = []
        model.train()

        # 前向传播+计算损失函数
        z = model(data)
        logits, edge_label = model.decode(z, data)
        train_loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        metric.append(train_loss.item())

        probs = logits.sigmoid()
        y_real = edge_label.cpu()
        y_pred = torch.zeros((len(y_real)))
        y_pred[probs>THRESHOLD] = 1
        train_acc = int(torch.sum(y_pred==y_real)) / len(y_real)
        metric.append(train_acc)

        # 优化器初始化+反向传播+优化器迭代优化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        z = model(data)
        logits, edge_label = model.decode(z, data, istrain=False)
        test_loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        metric.append(test_loss.item())
        probs = logits.sigmoid()
        y_real = edge_label.cpu()
        y_pred = torch.zeros((len(y_real)))
        y_pred[probs>THRESHOLD] = 1
        test_acc = int(torch.sum(y_pred==y_real)) / len(y_real)
        metric.append(test_acc)
        auc = roc_auc_score(edge_label.cpu().detach().numpy(), probs.cpu().detach().numpy())
        print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f} test_loss: {:.4f} test_acc: {:.4f} test_auc: {:.4f}'.format(
            epoch, train_loss.item(), train_acc, test_loss.item(), test_acc, auc))
        metrics.append(metric)
    M += np.array(metrics)
    z = model(data)
    logits, edge_label = model.decode(z, data, istrain=False)
    probs = logits.sigmoid()
    prob_scores.append(probs.tolist())
    model.decode_all(z)
    
    del model

prob_scores = np.array(prob_scores).T

score_mat_df = pd.DataFrame(SCORE_MAT/TIMES, index=all_prot, columns=all_prot)
score_mat_df.to_csv(OUT_DIR+'/prob_score_mat.csv')

metric_df = pd.DataFrame(M / TIMES, columns=['train_loss', 'train_acc', 'test_loss', 'test_acc'])
metric_df.to_csv(OUT_DIR + '/logs.csv')

    
#%%
y = np.hstack((y_real.reshape((-1,1)), prob_scores))
y_df = pd.DataFrame(y)
y_df.to_csv(OUT_DIR+'/prob_scores.csv')





# %%

