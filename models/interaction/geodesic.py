# suppose: 
# node_sca dim=4, edge_sca_dim=6
# node_vec_dim = 3, edge_vec_dim = 3 
from torch import nn
from torch_scatter import scatter_sum
from math import pi 
from ..invariant import VNLinear, GVPerceptronVN, GVLinear
import torch
from ..model_utils import  GaussianSmearing

class EdgeMapping(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)
    """在__init__方法中，定义了一个nn.Linear层，输入特征的数量为1，输出特征的数量为edge_channels。
    这个线性层将用于将边的向量映射到一个更高维的空间。在forward方法中，首先对边的向量进行了归一化，即将向量除以其2范数（即向量的长度）。
    这是为了确保向量的长度为1，避免因向量的长度过大或过小而影响后续的计算。然后，使用定义的线性层对归一化后的边向量进行了映射，并将结果进行了转置。"""
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7) # type: ignore
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion
        
class Geodesic_GNN(nn.Module):
    def __init__(self, node_sca_dim=64, node_vec_dim=16, hid_dim=32, edge_dim=16, num_edge_types=2, \
        out_sca_dim=64, out_vec_dim=16, cutoff = 10.):
        super().__init__()
        self.cutoff = cutoff # self.cutoff：一个阈值，用于GaussianSmearing层。
        self.edge_expansion = EdgeMapping(edge_dim) # self.edge_expansion：一个EdgeMapping层，用于将边的特征映射到交互空间。
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim - num_edge_types) #./models/model_utils.py 用于处理边的距离特征。

        self.node_mapper = GVLinear(node_sca_dim,node_vec_dim,node_sca_dim,node_vec_dim) #./models/invariant.py 用于处理节点的特征映射。
        self.edge_mapper = GVLinear(edge_dim,edge_dim,node_sca_dim,node_vec_dim) #./models/invariant.py 用于处理边的特征映射。
        """GVLinear模块内部包含两个VNLinear层和两个nn.Linear层。VNLinear层用于处理向量输入，nn.Linear层用于处理标量输入。
        在GVLinear的forward方法中，首先将输入特征分解为标量和向量，然后分别通过相应的层进行处理，最后使用sigmoid函数处理标量输出，并将其用于调整向量输出的大小。"""

        self.edge_sca_sca = nn.Linear(node_sca_dim, hid_dim)
        self.node_sca_sca = nn.Linear(node_sca_dim, hid_dim)
        """这两行代码创建了两个nn.Linear对象，分别赋值给self.edge_sca_sca和self.node_sca_sca。
        self.edge_sca_sca和self.node_sca_sca用于处理图形数据，其中edge_sca_sca处理边的标量特征，node_sca_sca处理节点的标量特征。"""
        
        self.edge_sca_vec = nn.Linear(node_sca_dim, hid_dim)
        self.node_sca_vec = nn.Linear(node_sca_dim, hid_dim)
        """这两行代码创建了两个nn.Linear对象，分别赋值给self.edge_sca_vec和self.node_sca_vec。
        self.edge_sca_vec和self.node_sca_vec用于处理图形数据，其中edge_sca_vec处理边的向量特征，node_sca_vec处理节点的向量特征。"""
        
        self.edge_vec_vec = VNLinear(node_vec_dim, hid_dim) #./models/invariant.py 用于实现特定的线性变换。
        self.node_vec_vec = VNLinear(node_vec_dim, hid_dim) #./models/invariant.py 用于实现特定的线性变换。

        self.msg_out = GVLinear(hid_dim, hid_dim, out_sca_dim, out_vec_dim) #./models/invariant.py 用于处理标量和向量输入，并输出标量和向量
        """self.msg_out可能用于处理图形数据，其中msg_out处理节点或边的信息。这个模块将输入特征映射到一个隐藏空间，然后输出标量和向量特征。"""

        self.resi_connecter = GVLinear(node_sca_dim,node_vec_dim,node_sca_dim,node_vec_dim) #./models/invariant.py 用于处理标量和向量输入，并输出标量和向量
        self.aggr_out = GVLinear(node_sca_dim,node_vec_dim,node_sca_dim,node_vec_dim) #./models/invariant.py 用于处理标量和向量输入，并输出标量和向量
        """self.resi_connecter和self.aggr_out可能用于处理图形数据的不同阶段。resi_connecter可能用于实现残差连接，将输入特征与处理后的特征相加，以增强模型的表达能力。
        aggr_out可能用于聚合节点或边的信息，生成最终的输出。"""
    
    def forward(self, node_feats, node_pos, edge_feature, edge_index, gds_dist):
        
        num_nodes = node_feats[0].shape[0]
        edge_index_row = edge_index[0]
        edge_index_col = edge_index[1]
        edge_vector = node_pos[edge_index_row] - node_pos[edge_index_col]
        """首先，num_nodes = node_feats[0].shape[0]这行代码获取了图中节点的数量。
        node_feats[0]表示节点特征的第一个维度，即节点的数量，shape[0]则获取这个维度的大小。
        然后，edge_index_row = edge_index[0]和edge_index_col = edge_index[1]这两行代码获取了边索引的行和列。
        在图形数据中，边索引通常是一个2D张量，其中第一维表示源节点，第二维表示目标节点。
        最后，edge_vector = node_pos[edge_index_row] - node_pos[edge_index_col]这行代码计算了边向量。
        这是通过获取每条边的源节点和目标节点的位置，然后计算两者之间的差来实现的。这个差就是边向量，它表示了从源节点到目标节点的方向和距离。"""
        
        ## map edge_fetures: original space -> interaction space
        edge_sca_feat = torch.cat([self.distance_expansion(gds_dist), edge_feature], dim=-1)
        edge_vec_feat = self.edge_expansion(edge_vector) 
        """首先，edge_sca_feat = torch.cat([self.distance_expansion(gds_dist), edge_feature], dim=-1)这行代码将
        通过self.distance_expansion(gds_dist)得到的特征和edge_feature进行拼接，得到边的标量特征。
        torch.cat是PyTorch的一个函数，用于将多个张量按指定的维度拼接在一起。在这里，dim=-1表示按最后一个维度进行拼接。
        然后，edge_vec_feat = self.edge_expansion(edge_vector)这行代码通过self.edge_expansion(edge_vector)处理edge_vector，得到边的向量特征。
        self.edge_expansion可能是一个神经网络模块，用于将边的向量映射到一个更高维的空间。"""

        # Geodesic Message Passing 
        ## mapping the node and edge features to the same space 
        node_sca_feats, node_vec_feats = self.node_mapper(node_feats)
        edge_sca_feat, edge_vec_feat = self.edge_mapper([edge_sca_feat, edge_vec_feat])
        node_sca_feats, node_vec_feats = node_sca_feats[edge_index_row], node_vec_feats[edge_index_row]
        """首先，node_sca_feats, node_vec_feats = self.node_mapper(node_feats)这行代码调用了self.node_mapper对象的方法，并将结果赋值给node_sca_feats和node_vec_feats。
        self.node_mapper可能是一个神经网络模块，用于处理节点特征node_feats，并输出标量特征node_sca_feats和向量特征node_vec_feats。
        然后，edge_sca_feat, edge_vec_feat = self.edge_mapper([edge_sca_feat, edge_vec_feat])这行代码调用了self.edge_mapper对象的方法，并将结果赋值给edge_sca_feat和edge_vec_feat。
        self.edge_mapper可能是一个神经网络模块，用于处理边的标量特征edge_sca_feat和向量特征edge_vec_feat，并输出处理后的特征。
        最后，node_sca_feats, node_vec_feats = node_sca_feats[edge_index_row], node_vec_feats[edge_index_row]这行代码根据边索引edge_index_row选择了对应的节点特征。这是因为在图形数据中，每条边都连接两个节点，因此需要根据边索引选择对应的节点特征。"""
        
        ## geodesic coefficient 
        coeff = 0.5 * (torch.cos(gds_dist * pi / self.cutoff) + 1.0)
        coeff = coeff * (gds_dist <= self.cutoff) * (gds_dist >= 0.0)
        """首先，coeff = 0.5 * (torch.cos(gds_dist * pi / self.cutoff) + 1.0)这行代码计算了一个余弦值，然后将其范围从[-1,1]调整到[0,1]，并乘以0.5。
        这是通过将gds_dist乘以pi / self.cutoff，然后取余弦，再加1.0，最后乘以0.5来实现的。
        这样做的目的可能是为了将gds_dist映射到一个特定的范围，并使其在self.cutoff处平滑地接近0。
        然后，coeff = coeff * (gds_dist <= self.cutoff) * (gds_dist >= 0.0)这行代码进一步调整了coeff。
        这是通过将coeff乘以两个布尔张量来实现的，这两个布尔张量分别表示gds_dist是否小于等于self.cutoff和是否大于等于0.0。
        在PyTorch中，布尔张量可以与数值张量进行运算，True被视为1，False被视为0。因此，这行代码实际上是将coeff在gds_dist大于self.cutoff或小于0.0的位置设为0。"""
        
        ## compute the scalar message
        msg_sca_emb = self.node_sca_sca(node_sca_feats) * self.edge_sca_sca(edge_sca_feat)
        msg_sca_emb = msg_sca_emb * coeff.view(-1,1)
        """首先，msg_sca_emb = self.node_sca_sca(node_sca_feats) * self.edge_sca_sca(edge_sca_feat)这行代码调用了self.node_sca_sca和self.edge_sca_sca对象的方法，并将结果相乘，得到msg_sca_emb。
        self.node_sca_sca和self.edge_sca_sca用于处理节点的标量特征node_sca_feats和边的标量特征edge_sca_feat。
        这行代码通过将处理后的节点和边的标量特征相乘，得到了标量消息。然后，msg_sca_emb = msg_sca_emb * coeff.view(-1,1)这行代码将msg_sca_emb和coeff相乘，得到新的msg_sca_emb。
        coeff是一个系数，可能用于调整msg_sca_emb的值。coeff.view(-1,1)是将coeff的形状调整为二维，其中第二维的大小为1，这样就可以与msg_sca_emb进行广播运算。"""

        ## compute the vector message
        msg_vec_emb1 = self.node_vec_vec(node_vec_feats) * self.edge_sca_vec(edge_sca_feat).unsqueeze(-1)
        msg_vec_emb2 = self.node_sca_vec(node_sca_feats).unsqueeze(-1) * self.edge_vec_vec(edge_vec_feat)
        """调用了四个神经网络模块的方法，并将结果相乘，得到msg_vec_emb1和msg_vec_emb2。
        这四个神经网络模块用于处理节点和边的标量特征和向量特征。unsqueeze(-1)是用于增加一个维度，使得可以进行广播运算。"""
        
        msg_vec_emb = msg_vec_emb1 + msg_vec_emb2
        """将msg_vec_emb1和msg_vec_emb2相加，得到msg_vec_emb。这是通过将两个向量消息相加，得到最终的向量消息。"""
        
        msg_vec_emb = msg_vec_emb * coeff.view(-1,1,1)
        """msg_vec_emb和coeff相乘，得到新的msg_vec_emb。coeff是一个系数，可能用于调整msg_vec_emb的值。
        coeff.view(-1,1,1)是将coeff的形状调整为三维，其中后两维的大小为1，这样就可以与msg_vec_emb进行广播运算。"""
        
        ## message pssing mapping 
        msg_sca_emb, msg_vec_emb = self.msg_out([msg_sca_emb, msg_vec_emb])
        """调用self.msg_out对象的方法，并将结果赋值给msg_sca_emb和msg_vec_emb。"""

        ## aggregate the message 
        aggr_msg_sca = scatter_sum(msg_sca_emb, edge_index_row, dim=0, dim_size=num_nodes)
        aggr_msg_vec = scatter_sum(msg_vec_emb, edge_index_row, dim=0, dim_size=num_nodes)
        """使用scatter_sum函数对消息进行聚合。scatter_sum函数将源张量的元素按照索引张量的值相加，得到一个新的张量。"""

        ## residue connection
        resi_sca, resi_vec = self.resi_connecter(node_feats)
        # print(resi_sca.shape, aggr_msg_sca.shape)
        # print(resi_vec.shape, aggr_msg_vec.shape)
        out_sca = resi_sca + aggr_msg_sca
        out_vec = resi_vec + aggr_msg_vec
        """调用了self.resi_connecter对象的方法，并将结果赋值给resi_sca和resi_vec。self.resi_connecter是一个神经网络模块，用于进行残差连接。"""

        ## aggregation mapper
        out_sca, out_vec = self.aggr_out([out_sca, out_vec])
        """调用了self.aggr_out对象的方法，并将结果赋值给out_sca和out_vec。self.aggr_out用于处理聚合后的输出。"""
        
        return [out_sca, out_vec]



# class Geodesic_GNN(nn.Module):
#     def __init__(self, node_sca_dim=4, node_vec_dim=3, edge_sca_dim=6, edge_vec_dim=3, out_sca=16, out_vec=16, cutoff=10.):
#         super().__init__()
#         # To simplify the model, the out_feats_dim of edges and nodes are the same
        
#         self.self.edge_sca_sca = nn.Linear(edge_sca_dim, out_sca)
#         self.self.node_sca_sca = nn.Linear(node_sca_dim, out_sca)

#         self.self.edge_sca_vec = nn.Linear(edge_sca_dim, out_sca)
#         self.self.node_sca_vec = nn.Linear(node_sca_dim, out_sca)
#         self.self.edge_vec_vec = VNLinear(edge_vec_dim, out_vec)
#         self.self.node_vec_vec = VNLinear(node_vec_dim, out_vec)
        
#         self.encoder = GVPerceptronVN(out_sca,out_vec,out_sca,out_sca)
    
#     def forward(self, node_feats, edge_feats, edge_index, gds_dist):
#         edge_index_raw = edge_index[0]
#         num_nodes = node_feats[0].shape[0]
#         coeff = 0.5 * (torch.cos(gds_dist * pi / self.cutoff) + 1.0)
#         coeff = coeff * (gds_dist <= self.cutoff) * (gds_dist >= 0.0)

#         # compute the scalar message
#         msg_sca_emb = self.self.node_sca_sca(node_feats[0])[edge_index_raw] * self.self.edge_sca_sca(edge_feats[0])
#         msg_sca_emb = msg_sca_emb * coeff.view(-1,1)
        
#         # compute the vector message
#         msg_vec_emb1 = self.self.node_vec_vec(node_feats[1])[edge_index_raw] * self.self.edge_sca_vec(edge_feats[0]).unsqueeze(-1)
#         msg_vec_emb2 = self.self.node_sca_vec(node_feats[0])[edge_index_raw].unsqueeze(-1) * self.self.edge_vec_vec(edge_feats[1])
#         msg_vec_emb = msg_vec_emb1 + msg_vec_emb2
#         msg_vec_emb = msg_vec_emb * coeff.view(-1,1,1)

#         # arrgrate the message 
#         aggr_msg_sca = scatter_sum(msg_sca_emb, edge_index_raw, dim=0, dim_size=num_nodes)
#         aggr_msg_vec = scatter_sum(msg_vec_emb, edge_index_raw, dim=0, dim_size=num_nodes)
        
#         # then encode the geodesic feates 
#         node_aggr_sca, node_aggr_vec = self.encoder((aggr_msg_sca,aggr_msg_vec))
        
#         return node_aggr_sca, node_aggr_vec