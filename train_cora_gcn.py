import torch
from models.gcn_cora import GCNNet
import numpy as np
from datasets.coraDataset import CoraDataset
from utils.util import *
import torch.nn as nn
import torch.optim as optim

# 超参数定义
LEARNING_RATE = 0.1
WEIGHT_DACAY = 5e-4
EPOCHS = 200
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


# 测试函数
def test(model, adjacency, x, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adjacency, x)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, y[mask]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), y[mask].cpu().numpy()


def main():
    dataset = CoraDataset("./cora").data
    # 加载数据，并转换为torch.Tensor
    node_feature = dataset.x / dataset.x.sum(1, keepdims=True)  # 归一化数据，使得每一行和为1
    tensor_x = tensor_from_numpy(node_feature, DEVICE)
    tensor_y = tensor_from_numpy(dataset.y, DEVICE)
    tensor_train_mask = tensor_from_numpy(dataset.train_mask, DEVICE)
    tensor_val_mask = tensor_from_numpy(dataset.val_mask, DEVICE)
    tensor_test_mask = tensor_from_numpy(dataset.test_mask, DEVICE)
    normalize_adjacency = CoraDataset.normalization(dataset.adjacency)  # 规范化邻接矩阵

    num_nodes, input_dim = node_feature.shape
    indices = torch.from_numpy(np.asarray([normalize_adjacency.row,
                                           normalize_adjacency.col]).astype('int64')).long()
    values = torch.from_numpy(normalize_adjacency.data.astype(np.float32))
    tensor_adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes)).to(DEVICE)

    # 模型定义：Model, Loss, Optimizer
    model = GCNNet(input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DACAY)

    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    for epoch in range(EPOCHS):
        logits = model(tensor_adjacency, tensor_x)  # 前向传播
        train_mask_logits = logits[tensor_train_mask]  # 只选择训练节点进行监督
        loss = criterion(train_mask_logits, train_y)  # 计算损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        train_acc, _, _ = test(model, tensor_adjacency, tensor_x, tensor_y, tensor_train_mask)  # 计算当前模型训练集上的准确率
        val_acc, _, _ = test(model, tensor_adjacency, tensor_x, tensor_y, tensor_test_mask)  # 计算当前模型在验证集上的准确率
        # 记录训练过程中损失值和准确率的变化，用于画图
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))


if __name__ == '__main__':
    main()