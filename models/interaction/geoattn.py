# supnode_pose: 
# node_sca dim=4, edge_sca_dim=6
# node_vec_dim = 3, edge_vec_dim = 3 
from torch import nn
from torch_scatter import scatter_sum
from ..invariant import VNLinear, GVPerceptronVN, GVLinear
import torch
from torch_scatter import scatter_softmax
from torch.nn import Sigmoid
from ..model_utils import  GaussianSmearing

class EdgeMapping(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)
        """在 __init__ 方法中，我们初始化了一个线性层 self.nn，它的输入特征数为 1，输出特征数为 edge_channels。
        这个线性层没有偏置项（bias term），这意味着它只会对输入进行线性变换，而不会添加任何常数偏置。"""
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7) # type: ignore
        """在 forward 方法中，我们首先对 edge_vector 进行了归一化处理。这是通过 edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7) 实现的，
        其中 torch.norm 函数用于计算 edge_vector 的 2-范数（也就是其长度），然后我们将 edge_vector 除以其长度，得到一个单位长度的向量。
        这样做的目的是为了消除 edge_vector 的长度对后续计算的影响。"""
        
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        """我们将归一化后的 edge_vector 通过 self.nn 线性层进行变换，然后对结果进行维度变换。这是通过 self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1) 实现的。
        unsqueeze(-1) 是将 edge_vector 在最后一个维度上增加一个维度，transpose(1, -1) 是将第二个维度和最后一个维度进行交换。这样做的目的是为了让输出的形状与预期的形状一致。"""
        return expansion

class Geoattn_GNN(nn.Module):
    def __init__(self, node_sca_dim=64,node_vec_dim=16, num_edge_types=4, edge_dim=16, hid_dim=32,\
        out_sca_dim=64, out_vec_dim=16, cutoff=10):
        super().__init__()
        
        self.cutoff = cutoff # self.cutoff：一个阈值，用于GaussianSmearing层。
        self.edge_expansion = EdgeMapping(edge_dim) # self.edge_expansion：一个EdgeMapping层，用于将边的特征映射到交互空间
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim - num_edge_types) #./models/model_utils.py 用于处理边的距离特征。
        
        self.node_mapper = GVLinear(node_sca_dim,node_vec_dim,node_sca_dim,node_vec_dim) #./models/invariant.py 用于处理节点的特征映射。
        """创建了一个GVLinear的实例，并将其赋值给self.node_mapper。在创建实例时，传入了四个参数：node_sca_dim、node_vec_dim、node_sca_dim和node_vec_dim。
        这四个参数分别表示节点的标量输入维度、向量输入维度、标量输出维度和向量输出维度。"""
        self.edge_mapper = GVLinear(edge_dim,edge_dim,node_sca_dim,node_vec_dim) #./models/invariant.py 用于处理节点的特征映射。
        """创建了一个GVLinear的实例，并将其赋值给self.edge_mapper。在创建实例时，传入了四个参数：edge_dim、edge_dim、node_sca_dim和node_vec_dim。
        这四个参数分别表示边的标量输入维度、向量输入维度、标量输出维度和向量输出维度。"""
        """GVLinear类是一个神经网络模块，用于实现一种特殊的线性变换，该变换处理标量和向量输入，并输出标量和向量。
        在GVLinear类的forward方法中，首先将输入特征分解为标量和向量，然后通过一系列的神经网络层处理这些特征，最后得到标量和向量的输出。"""

        self.edge_net = nn.Linear(node_sca_dim, hid_dim) 
        self.node_net = nn.Linear(node_sca_dim, hid_dim)
        """self.edge_sca_net = nn.Linear(node_sca_dim, hid_dim)这行代码创建了一个nn.Linear的实例，并将其赋值给self.edge_sca_net。
        nn.Linear是PyTorch中的一个类，用于实现线性变换。
        在创建实例时，传入了两个参数：node_sca_dim和hid_dim。这两个参数分别表示输入特征的维度和输出特征的维度。
        然后，self.node_sca_net = nn.Linear(node_sca_dim, hid_dim)这行代码创建了另一个nn.Linear的实例，并将其赋值给self.node_sca_net。
        这个实例的输入和输出特征的维度与self.edge_sca_net相同。"""

        self.edge_sca_net = nn.Linear(node_sca_dim, hid_dim)
        self.node_sca_net = nn.Linear(node_sca_dim, hid_dim)
        """两行代码创建了两个nn.Linear的实例，分别赋值给self.edge_sca_net和self.node_sca_net。
        nn.Linear是PyTorch中的一个类，用于实现线性变换。
        在创建实例时，传入了两个参数：node_sca_dim和hid_dim。
        这两个参数分别表示输入特征的维度和输出特征的维度"""
        
        self.edge_vec_net = VNLinear(node_vec_dim, hid_dim)
        self.node_vec_net = VNLinear(node_vec_dim, hid_dim)
        """创建了两个VNLinear的实例，分别赋值给self.edge_vec_net和self.node_vec_net。
        VNLinear是一个自定义的神经网络模块，用于实现一种特殊的线性变换，该变换处理向量输入，并输出向量。
        在创建实例时，传入了两个参数：node_vec_dim和hid_dim。这两个参数分别表示输入特征的维度和输出特征的维度。
        在VNLinear类的forward方法中，输入数据的最后两个维度会被交换，然后传递给self.map_to_feat进行处理，处理后的数据再次交换最后两个维度，然后返回。"""

        self.sca_attn_net = nn.Linear(node_sca_dim*2+1, hid_dim)
        self.vec_attn_net = VNLinear(node_vec_dim, hid_dim)
        
        self.softmax = scatter_softmax  
        """将scatter_softmax函数赋值给self.softmax。scatter_softmax是一个特殊的softmax函数，它可以在scatter操作的维度上进行softmax计算。
        这种操作在图神经网络中很常见，因为我们经常需要在节点的邻居上进行softmax操作。这样，每个节点的邻居的权重之和就会等于1，这对于许多图神经网络的注意力机制来说是必要的。"""
        self.sigmoid = nn.Sigmoid()
        """self.sigmoid = nn.Sigmoid()这行代码创建了一个nn.Sigmoid的实例，并将其赋值给self.sigmoid。nn.Sigmoid是PyTorch中的一个类，用于实现sigmoid函数。
        sigmoid函数可以将任何实数映射到(0,1)区间，常用于二分类问题的输出层，或者需要输出在(0,1)区间的情况。"""

        self.msg_out = GVLinear(hid_dim, hid_dim, out_sca_dim, out_vec_dim)
        

        self.resi_connecter = GVLinear(node_sca_dim,node_vec_dim,node_sca_dim,node_vec_dim)
        self.aggr_out = GVPerceptronVN(node_sca_dim,node_vec_dim,node_sca_dim,node_vec_dim) #./models/invariant.py GVPerceptronVN类是一个感知器，它处理标量和向量输入，并输出标量和向量。
    
    def forward(self, node_feats, node_pos, edge_feature, edge_index):
        num_nodes = node_feats[0].shape[0] # 获取了节点特征矩阵的第一维度的大小，也就是节点的数量。
        edge_index_row = edge_index[0] 
        edge_index_col = edge_index[1] # 获取了边索引的第一行和第二行。在图神经网络中，边索引通常是一个2维的张量，其中第一行表示边的源节点，第二行表示边的目标节点。
        edge_vector = node_pos[edge_index_row] - node_pos[edge_index_col] # 计算了每条边的向量。这里，node_pos是一个矩阵，其中每一行表示一个节点的位置。通过索引操作，我们可以获取每条边的源节点和目标节点的位置，然后通过减法操作，我们可以得到每条边的向量。

        edge_dist = torch.norm(edge_vector, dim=-1)
        ## map edge_features: original space -> interation space
        edge_dist = torch.norm(edge_vector, dim=-1, p=2) # type: ignore 
        """edge_vector是一个张量，其中每一行表示一条边的向量。
        torch.norm(edge_vector, dim=-1)和torch.norm(edge_vector, dim=-1, p=2)这两个函数都计算了edge_vector在最后一个维度（即每一行）上的L2范数，也就是每条边向量的长度。
        这里，参数p=2表示使用L2范数，也就是欧几里得范数。需要注意的是，这段代码中的两行代码实际上是重复的，它们都计算了每条边向量的长度。
        在实际使用中，我们只需要其中的一行代码就可以了。"""
        
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        """torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1) 将 self.distance_expansion(edge_dist) 和 edge_feature 两个张量在最后一个维度上拼接起来。
        这意味着，如果 self.distance_expansion(edge_dist) 的形状是 (a, b, c)，edge_feature 的形状是 (a, b, d)，那么拼接后的张量 edge_sca_feat 的形状就会是 (a, b, c+d)。"""
        edge_vec_feat = self.edge_expansion(edge_vector) 
        """self.edge_expansion 是一个方法，它对输入的 edge_vector 进行某种形式的扩展。具体的扩展方式取决于该方法的实现，但通常这种扩展会增加输入张量的维度或者改变其形状。"""

        # message passing framework
        ## extract edge and node features in interaction space
        node_sca_feats, node_vec_feats = self.node_mapper(node_feats) # 调用了 self.node_mapper 方法对 node_feats 进行映射，然后将映射后的结果分别赋值给 node_sca_feats 和 node_vec_feats。
        edge_sca_feat, edge_vec_feat = self.edge_mapper([edge_sca_feat, edge_vec_feat]) # 调用了 self.edge_mapper 方法对 edge_sca_feat 和 edge_vec_feat 进行映射，然后将映射后的结果分别赋值给 edge_sca_feat 和 edge_vec_feat。
        node_sca_feats, node_vec_feats = node_sca_feats[edge_index_row], node_vec_feats[edge_index_row] # 根据 edge_index_row 的值选择 node_sca_feats 和 node_vec_feats 的子集。
 
        ## compute the attention score \alpha_ij and A_ij
        alpha_sca = torch.cat([node_sca_feats[edge_index[0]], node_sca_feats[edge_index[1]], edge_dist.unsqueeze(-1)], dim=-1)
        """使用了 torch.cat 函数来拼接三个张量：node_sca_feats[edge_index[0]]，node_sca_feats[edge_index[1]] 和 edge_dist.unsqueeze(-1)。"""
        alpha_sca = self.sca_attn_net(alpha_sca)
        alpha_sca = self.softmax(alpha_sca,edge_index_row,dim=0)
        """使用了 self.sca_attn_net 对拼接后的张量进行处理，然后使用了 self.softmax 函数对处理后的张量进行 softmax 操作。"""

        alpha_vec_hid = self.vec_attn_net(node_vec_feats)
        alpha_vec = (alpha_vec_hid[edge_index[0]] * alpha_vec_hid[edge_index[1]]).sum(-1).sum(-1)
        alpha_vec = self.sigmoid(alpha_vec)
        """使用了 self.vec_attn_net 对 node_vec_feats 进行处理，然后将处理后的结果分别赋值给 alpha_vec_hid 和 alpha_vec。"""
    
        ## message: the scalar feats
        node_sca_feat =  self.node_net(node_sca_feats)[edge_index_row] * self.edge_net(edge_sca_feat) 
        """将 node_sca_feats 作为输入传递给 self.node_net 网络，然后根据 edge_index_row 的值选择结果的子集, 然后进行元素级别（element-wise）的乘法运算。"""
        
        ## message: the equivariant interaction between node feature and edge feature
        node_sca_hid = self.node_sca_net(node_sca_feats)[edge_index_row].unsqueeze(-1)
        edge_vec_hid = self.edge_vec_net(edge_vec_feat)
        node_vec_hid = self.node_vec_net(node_vec_feats)[edge_index_row]
        edge_sca_hid =  self.edge_sca_net(edge_sca_feat).unsqueeze(-1)
        """我们有四个隐藏层的特征：node_sca_hid、edge_vec_hid、node_vec_hid 和 edge_sca_hid。
        这些特征是通过将相应的输入特征（node_sca_feats、edge_vec_feat、node_vec_feats 和 edge_sca_feat）传递给预定义的网络
        （self.node_sca_net、self.edge_vec_net、self.node_vec_net 和 self.edge_sca_net）并进行处理得到的。"""
        
        msg_sca = node_sca_feat * alpha_sca 
        msg_vec = (node_sca_hid * edge_vec_hid + node_vec_hid*edge_sca_hid)*alpha_vec.unsqueeze(-1).unsqueeze(-1)
        """定义了两个消息变量：msg_sca和msg_vec。这两个变量是通过对节点和边的特征进行一些操作得到的，这些操作可能是图形神经网络中的消息传递步骤。
        具体来说，msg_sca是通过将node_sca_feat和alpha_sca相乘得到的，而msg_vec是通过将node_sca_hid和edge_vec_hid相乘，
        然后加上node_vec_hid和edge_sca_hid相乘，最后乘以alpha_vec.unsqueeze(-1).unsqueeze(-1)得到的。
        这里的alpha_sca和alpha_vec可能是一些权重参数，用于调整消息的重要性。"""
        
        msg_sca,msg_vec = self.msg_out([msg_sca,msg_vec])
        """msg_sca和msg_vec是之前计算出来的消息，它们可能包含了节点和边的特征信息。这些消息被传递给self.msg_out函数进行处理。
        处理后的消息被赋值给原来的msg_sca和msg_vec，这意味着原来的消息被更新了。这些更新后的消息可能会在后续的步骤中用于更新节点和边的特征。"""
        
        ## aggregate the message 
        aggr_msg_sca = scatter_sum(msg_sca, edge_index_row, dim=0, dim_size=num_nodes)
        aggr_msg_vec = scatter_sum(msg_vec, edge_index_row, dim=0, dim_size=num_nodes)
        """在这两行代码中，scatter_sum函数被用于实现这个聚合过程。scatter_sum是一个常用的函数，
        它可以将一个源张量（在这里是msg_sca和msg_vec）中的元素按照索引（在这里是edge_index_row）进行聚合。
        具体来说，它会将源张量中具有相同索引的元素相加，然后将结果存储在目标张量中。
        dim=0参数指定了聚合操作在源张量的哪一个维度上进行，dim_size=num_nodes参数指定了目标张量的大小，这里是节点的数量。
        aggr_msg_sca和aggr_msg_vec是聚合后的消息，它们包含了每个节点的所有邻居节点的信息。这些聚合后的消息可能会在后续的步骤中用于更新节点的特征。"""

        ## residue connection 
        resi_sca, resi_vec = self.resi_connecter(node_feats)
        out_sca = resi_sca + aggr_msg_sca
        out_vec = resi_vec + aggr_msg_vec
        """代码通过self.resi_connecter(node_feats)获取了两个残差连接的输出resi_sca和resi_vec。
        self.resi_connecter可能是一个神经网络层或者一个函数，它接收节点特征node_feats作为输入，然后返回处理后的特征。
        代码通过将残差连接的输出和之前聚合的消息进行相加，得到了最终的输出out_sca和out_vec。
        这里的aggr_msg_sca和aggr_msg_vec是之前通过scatter_sum函数聚合得到的消息。"""

        ## map the aggregated feature
        out_sca, out_vec = self.aggr_out([out_sca, out_vec]) # 聚合特征被传递给self.aggr_out函数进行处理，处理后的结果被赋值给out_sca和out_vec。

        return [out_sca, out_vec]
