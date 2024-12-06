import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, LeakyReLU
import numpy as np
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from math import pi as PI
EPS = 1e-6
# from utils.profile import lineprofile


class MessageModule(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = GVLinear(node_sca, node_vec, out_sca, out_vec) # GVLinear类用于实现一种特殊的线性变换，该变换处理标量和向量输入，并输出标量和向量。
        self.edge_gvp = GVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec) # GVPerceptronVN类是一个感知器，它处理标量和向量输入，并输出标量和向量。

        self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec) # 用于实现一种特殊的线性变换，该变换处理向量输入，并输出向量。

        self.out_gvlienar = GVLinear(out_sca, out_vec, out_sca, out_vec) # GVLinear类用于实现一种特殊的线性变换，该变换处理标量和向量输入，并输出标量和向量。

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=0, annealing=False):
        node_scalar, node_vector = self.node_gvlinear(node_features)
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]
        """输入的节点特征node_features通过node_gvlinear进行处理，得到node_scalar和node_vector。然后，根据edge_index_node选择对应的node_scalar和node_vector。"""
        
        edge_scalar, edge_vector = self.edge_gvp(edge_features)
        """输入的边特征edge_features通过edge_gvp进行处理，得到edge_scalar和edge_vector。"""

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        """计算y_scalar，y_node_vector和y_edge_vector。y_scalar是node_scalar和edge_scalar经过self.sca_linear处理后的乘积。
        y_node_vector是edge_scalar经过self.e2n_linear处理并增加一个维度后与node_vector的乘积。
        y_edge_vector是node_scalar经过self.n2e_linear处理并增加一个维度后与edge_vector经过self.edge_vnlinear处理后的乘积。"""
        
        y_vector = y_node_vector + y_edge_vector
        """将y_node_vector和y_edge_vector相加，得到y_vector。"""

        output = self.out_gvlienar((y_scalar, y_vector)) # 将y_scalar和y_vector作为输入，通过out_gvlienar进行处理，得到输出output。
        
        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # type: ignore # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]   # (A, 1)
        return output


class GVPerceptronVN(Module): #GVPerceptronVN类是一个感知器，它处理标量和向量输入，并输出标量和向量。
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = GVLinear(in_scalar, in_vector, out_scalar, out_vector) #用于处理标量和向量输入，并输出标量和向量。
        self.act_sca = LeakyReLU() #实现LeakyReLU激活函数，它可以在神经网络中用于增加非线性，解决梯度消失问题，从而提高模型的学习能力。
        self.act_vec = VNLeakyReLU(out_vector) #实现向量版本的LeakyReLU激活函数

    def forward(self, x):
        sca, vec = self.gv_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec
    """在forward方法中，首先通过self.gv_linear处理输入x，得到标量和向量。然后，分别通过self.act_vec和self.act_sca处理向量和标量，得到最终的输出。"""


class GVLinear(Module): # GVLinear类用于实现一种特殊的线性变换，该变换处理标量和向量输入，并输出标量和向量。
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vector = VNLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_vector = VNGroupLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_scalar = Conv1d(in_scalar + dim_hid, out_scalar, 1, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)
        """在__init__方法中，首先调用了父类Module的__init__方法，然后定义了一些类的属性。
        dim_hid是输入向量和输出向量中较大的一个。self.lin_vector和self.lin_vector2是VNLinear对象，用于处理向量输入。
        self.scalar_to_vector_gates和self.lin_scalar是Linear对象，用于处理标量输入。"""

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)  # type: ignore # (N_samples, dim_hid)
        feat_scalar_cat = torch.cat([feat_vector_norm, feat_scalar], dim=-1)  # (N_samples, dim_hid+in_scalar)

        out_scalar = self.lin_scalar(feat_scalar_cat)
        out_vector = self.lin_vector2(feat_vector_inter)

        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim = -1)
        out_vector = gating * out_vector
        return out_scalar, out_vector
        """在forward方法中，首先将输入特征分解为标量和向量。然后，通过self.lin_vector处理向量特征，并计算其二范数。
        接着，将计算得到的范数与标量特征拼接，然后通过self.lin_scalar处理得到标量输出。
        同时，通过self.lin_vector2处理向量特征得到向量输出。最后，使用sigmoid函数处理标量输出，并将其用于调整向量输出的大小。"""


class VNLinear(nn.Module): # VNLinear类用于实现一种特殊的线性变换，该变换处理向量输入，并输出向量。
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)
        """在__init__方法中，首先调用了父类nn.Module的__init__方法，然后创建了一个nn.Linear对象，并将其赋值给self.map_to_feat。
        nn.Linear是一个全连接层，用于实现线性变换。in_channels和out_channels是输入和输出特征的数量，*args和**kwargs是其他可选参数。"""
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x_out = self.map_to_feat(x.transpose(-2,-1)).transpose(-2,-1)
        return x_out
        """在forward方法中，输入数据x的最后两个维度会被交换，然后传递给self.map_to_feat进行处理，处理后的数据再次交换最后两个维度，然后返回。"""

class VNLeakyReLU(nn.Module): # VNLeakyReLU类用于实现一种特殊的LeakyReLU激活函数，该激活函数处理向量输入，并输出向量。
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    """在__init__方法中，首先调用了父类nn.Module的__init__方法，然后根据share_nonlinearity参数的值，初始化了self.map_to_dir。
    如果share_nonlinearity为True，则self.map_to_dir是一个nn.Linear对象，用于将输入的特征映射到一个维度；否则，self.map_to_dir是一个nn.Linear对象，用于将输入的特征映射到与输入相同的维度。
    此外，还初始化了self.negative_slope，用于控制LeakyReLU激活函数的负斜率。"""

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        d = self.map_to_dir(x.transpose(-2,-1)).transpose(-2,-1)  # (N_samples, N_feat, 3)
        dotprod = (x*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        x_out = (self.negative_slope * x +
                (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d)))
        return x_out
        """在forward方法中，首先通过self.map_to_dir处理输入x，得到d。然后，计算x和d的点积，得到dotprod。
        接着，创建一个掩码mask，用于标记dotprod中大于等于0的元素。然后，计算d的平方和，得到d_norm_sq。
        最后，根据LeakyReLU激活函数的公式，计算输出x_out。"""


# class VNAvgPool(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.pool = global_mean_pool

#     def forward(self, x, batch_idx):
#         '''
#         x: point features of shape [N_samples, N_feat, 3,]
#         '''
#         return self.pool(x, batch=batch_idx)  # (1, N_feat, 3)


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)
