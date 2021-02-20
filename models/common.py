import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_channels) + ' -> ' + str(self.out_channels) + ')'

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法

                Args:
                -------
                    adjacency: torch.sparse.FloatTensor
                        邻接矩阵
                    input_feature: torch.Tensor
                        输入特征
        """
        support = torch.mm(self.weights, input_feature)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output


if __name__ == '__main__':
    a = GraphConvolution(1, 2)
    print(a)



