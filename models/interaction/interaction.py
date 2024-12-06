import numpy as np
import torch
from torch.nn import Module, ModuleList, LeakyReLU, LayerNorm
from torch_scatter import scatter_sum, scatter
from math import pi as PI

from ..model_utils import GaussianSmearing, EdgeExpansion
from ..invariant import GVLinear, VNLeakyReLU, MessageModule
from .geodesic import Geodesic_GNN
from .geoattn import Geoattn_GNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class InteractionModule(Module):
    def __init__(self, node_sca_dim=64, node_vec_dim=16, edge_dim=16, hid_dim=32, num_geodesic=2, \
                 num_geoattn=4, k=24, cutoff=10.):

        super().__init__()

        self.node_sca_dim = node_sca_dim
        self.node_vec_dim = node_vec_dim
        self.edge_dim = edge_dim
        self.hid_dim = hid_dim
        self.num_geodesic = num_geodesic
        self.num_geoattn = num_geoattn
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_geodesic):
            block = Geodesic_GNN(
                node_sca_dim=node_sca_dim,
                node_vec_dim=node_vec_dim,
                hid_dim=hid_dim,
                edge_dim=edge_dim,
                num_edge_types=2,
                out_sca_dim=node_sca_dim,
                out_vec_dim=node_vec_dim,
                cutoff=cutoff
            )
            self.interactions.append(block)

        for _ in range(num_geoattn):
            block = Geoattn_GNN(
                node_sca_dim=node_sca_dim,
                node_vec_dim=node_vec_dim,
                hid_dim=hid_dim,
                edge_dim=edge_dim,
                num_edge_types=4,
                out_sca_dim=node_sca_dim,
                out_vec_dim=node_vec_dim,
                cutoff=cutoff
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]

    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, idx_ligand, idx_surface, gds_edge_index, gds_edge_feature, gds_dis,
                geom_edge_index, geom_edge_feature):

        h_surface_sca = node_attr[0][idx_surface]
        h_surface_vec = node_attr[1][idx_surface]
        gds_edge_vec = pos[idx_ligand][gds_edge_index[0]] - pos[idx_ligand][gds_edge_index[1]]

        for geodesic_block in self.interactions[:self.num_geodesic]:
            delta_h = geodesic_block([h_surface_sca, h_surface_vec], gds_edge_feature, gds_edge_vec, gds_edge_index,
                                     gds_dis)
            h_surface_sca = h_surface_sca + delta_h[0]
            h_surface_vec = h_surface_vec + delta_h[1]

        h_ligpkt_sca = torch.cat([node_attr[0][idx_ligand], h_surface_sca], dim=0)
        h_ligpkt_vec = torch.cat([node_attr[1][idx_ligand], h_surface_vec], dim=0)
        geom_edge_vec = pos[geom_edge_index[0]] - pos[geom_edge_index[1]]

        for geoattn_block in self.interactions[self.num_geoattn:]:
            delta_h = geoattn_block([h_ligpkt_sca, h_ligpkt_vec], geom_edge_feature, geom_edge_vec, geom_edge_index)
            h_ligpkt_sca = h_ligpkt_sca + delta_h[0]
            h_ligpkt_vec = h_ligpkt_vec + delta_h[1]

        return [h_ligpkt_sca, h_ligpkt_vec]


##############################################################################################################
class TransformerFeatureMixer(Module):

    def __init__(self, hidden_channels=[64, 16], edge_channels=16, num_edge_types=4, key_channels=32, num_heads=1,
                 num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff
        self.exphormer_attention = ExphormerAttention(self.hidden_channels[0], self.hidden_channels[0], num_heads)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(hidden_channels=hidden_channels, edge_channels=edge_channels,
                                                num_edge_types=num_edge_types, key_channels=key_channels,
                                                num_heads=num_heads, cutoff=cutoff)
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]

    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):

        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        edge_vector = self.exphormer_attention(node_attr, edge_index, edge_feature, edge_vector)

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h

class TransformerFeatureMixerForPos(Module):

    def __init__(self, hidden_channels=[64, 16], edge_channels=16, num_edge_types=4, key_channels=32, num_heads=1,
                 num_interactions=6, k=32, cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff
        self.exphormer_attention = ExphormerAttention(self.hidden_channels[0], self.hidden_channels[0], num_heads)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(hidden_channels=hidden_channels, edge_channels=edge_channels,
                                                num_edge_types=num_edge_types, key_channels=key_channels,
                                                num_heads=num_heads, cutoff=cutoff)
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]

    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):

        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h


class AttentionInteractionBlockVN(Module):

    def __init__(self, hidden_channels, edge_channels, num_edge_types, key_channels, num_heads=1, dropout=0.1,
                 alpha=0.2, cutoff=10.):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        # edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels - num_edge_types)
        self.vector_expansion = EdgeExpansion(
            edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(hidden_channels[0], hidden_channels[1], edge_channels, edge_channels,
                                            hidden_channels[0], hidden_channels[1], cutoff)
        self.GAT_attention = [
            GATLayer(hidden_channels[0], hidden_channels[0], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(num_heads)]
        self.out_att = GATLayer(hidden_channels[0] * num_heads, hidden_channels[0], dropout=dropout, alpha=alpha,
                                concat=False)

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index  # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector)

        # edge_sca_feat, edge_vec_feat = self.exphormer_attention((edge_sca_feat, edge_vec_feat))

        # edge_sca_feat = torch.nn.functional.dropout(edge_sca_feat, self.dropout)
        # edge_vec_feat = torch.nn.functional.dropout(edge_vec_feat, self.dropout)
        # edge_sca_feat = torch.cat([attention(edge_sca_feat) for attention in self.attention])
        # edge_vec_feat = torch.cat([attention(edge_vec_feat) for attention in self.attention])
        # edge_sca_feat = torch.nn.functional.dropout(edge_sca_feat, self.dropout)
        # edge_vec_feat = torch.nn.functional.dropout(edge_vec_feat, self.dropout)
        # edge_sca_feat = torch.nn.functional.elu(self.out_att(edge_sca_feat, self.alpha))
        # edge_vec_feat = torch.nn.functional.elu(self.out_att(edge_vec_feat, self.alpha))

        # Compute messages
        msg_j_sca, msg_j_vec = self.message_module(x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  # .view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  # .view(N, -1, 3) # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class ExphormerAttention(Module):

    def __init__(self, in_dim, out_dim, num_heads=4, use_bias=True, dim_edge=None, use_virt_nodes=False):
        super().__init__()

        if out_dim % num_heads != 0:
            raise ValueError('hidden dimension is not dividable by the number of heads')
        self.out_dim = out_dim // num_heads
        self.num_heads = num_heads
        self.use_virt_nodes = use_virt_nodes
        self.use_bias = use_bias

        if dim_edge is None:
            dim_edge = in_dim



        self.linear = torch.nn.Linear(16, 64, bias=use_bias)
        self.linear1 = torch.nn.Linear(1, 4, bias=use_bias)
        self.linear2 = torch.nn.Linear(4, 16, bias=use_bias)
        self.linear_out = torch.nn.Linear(64, 16, bias=use_bias)
        self.linear_out1 = torch.nn.Linear(16, 1, bias=use_bias)

        self.elu = torch.nn.functional.elu
        self.dropout = torch.nn.functional.dropout
        

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     xavier_uniform_(self.Q)
    #     xavier_uniform_(self.K)
    #     xavier_uniform_(self.V)
    #     xavier_uniform_(self.E)

    def propagate_attention(self, Q_h, K_h, E, V_h, edge_index):
        src = K_h[edge_index[0].to(torch.long)]  # (num edges) x num_heads x out_dim
        dest = Q_h[edge_index[1].to(torch.long)]  # (num edges) x num_heads x out_dim
        
        E = E.reshape(-1, 16)
        # E = self.linear(E)
        # E = E.reshape(-1, 16)
        # E = torch.nn.functional.elu(E)
        # E = torch.nn.functional.dropout(E, 0.1)
        # E = self.linear(E)
        # E = torch.nn.functional.elu(E)
        
        
        
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / np.sqrt(self.out_dim)
        
        E = E.reshape(-1, score.shape[1], score.shape[2])
        
        # Use available edge features to modify the scores for edges
        score = torch.mul(score, E)  # (num real edges) x num_heads x out_dim
        score = torch.exp(score.sum(-1, keepdim=True).clamp(-5, 5))  # (num real edges) x num_heads x 1

        # Apply attention score to each source node to create edge messages
        msg = V_h[edge_index[0].to(torch.long)] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        wV = torch.zeros_like(V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=wV, dim_size=V_h.size(0), reduce='add')

        # Compute attention normalization coefficient
        Z = score.new_zeros(V_h.size(0), self.num_heads, 1)  # (num nodes in batch) x num_heads x 1
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')
        return wV, Z

    def forward(self, node_attr, edge_index, edge_feature, edge_vector):
        edge_attr = edge_feature
        edge_index = edge_index
        h = edge_vector
        num_node = node_attr[0].shape[0]
        # if self.use_virt_nodes:
        #     h = torch.cat([h, batch.virt_h], dim=0)
        #     edge_index = torch.cat([edge_index, batch.virt_edge_index], dim=1)
        #     edge_attr = torch.cat([edge_attr, batch.virt_edge_attr], dim=0)

        # num_elements = h.numel()
        # padding = self.out_dim * self.num_heads - (num_elements % (self.out_dim * self.num_heads))
        # h = h.reshape(-1)
        # h = torch.nn.functional.pad(h, (0, padding))
        h = h.reshape(-1, 1)
        h = self.linear1(h)
        h = self.elu(h)
        h = self.dropout(h, 0.1)
        h = self.linear2(h)
        h = h.view(-1, self.out_dim).float()
        
        Q_h = self.linear(h)
        K_h = self.linear(h)
        V_h = self.linear(h)
        
        # num_elements = edge_attr.numel()
        # padding = (self.out_dim * self.num_heads) - (num_elements % (self.out_dim * self.num_heads))
        # edge_attr = edge_attr.reshape(-1)
        # edge_attr = torch.nn.functional.pad(edge_attr, (0, padding))

        edge_attr = self.linear2(edge_attr.float())
        edge_attr = self.elu(edge_attr)
        edge_attr = self.dropout(edge_attr, 0.1)
        E = self.linear(edge_attr)
        
        E = E.reshape(-1, self.out_dim)
        
        
        # num_elements = h_vec.numel()
        # padding = 64 - (num_elements % 64)

        # # 对h_vec进行补零
        # h_vec = h_vec.reshape(-1)
        # h_vec = torch.nn.functional.pad(h_vec, (0, padding))

        # h_vec = h_vec.reshape(-1, 64)
        # Q_h_vec = self.Q(h_vec)
        # K_h_vec = self.K(h_vec)
        # V_h_vec = self.V(h_vec)
        
        # edge_attr = edge_attr.float().reshape(-1, 64) 
        # E = self.E(edge_attr)
        
        
        

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        
        # Q_h_vec = Q_h_vec.view(-1, self.num_heads, self.out_dim)
        # K_h_vec = K_h_vec.view(-1, self.num_heads, self.out_dim)
        # V_h_vec = V_h_vec.view(-1, self.num_heads, self.out_dim)
        
        E = E.view(-1, self.num_heads, self.out_dim)

        wV, Z = self.propagate_attention(Q_h, K_h, E, V_h, edge_index)
        # wV_vec, Z_vec = self.propagate_attention(Q_h_vec, K_h_vec, E, V_h_vec, edge_index)

        h_out = wV / (Z + 1e-6)
        # h_out_vec = wV_vec / (Z_vec + 1e-6)

        h_out = h_out.view(-1, self.out_dim * self.num_heads)
        # h_out_vec = h_out_vec.view(-1, self.out_dim * self.num_heads)

        h_out = self.linear_out(h_out)
        h_out = torch.nn.functional.elu(h_out)
        h_out = torch.nn.functional.dropout(h_out, 0.1)
        h_out = self.linear_out1(h_out)

        h_out = h_out.view(-1, 3)
        # h_out_vec = h_out_vec.view(-1, num_node, 3)
        
        return h_out


class GATLayer(Module):
    """GAT层"""

    def __init__(self, input_feature, output_feature, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.alpha = alpha
        self.dropout = dropout
        self.concat = concat
        self.a = torch.nn.Parameter(torch.empty(size=(2 * output_feature, 1))).to(device)
        self.w = torch.nn.Parameter(torch.empty(size=(input_feature, output_feature))).to(device)
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w.detach(), gain=1.414)
        torch.nn.init.xavier_uniform_(self.a.detach(), gain=1.414)

    def forward(self, batch):
        h = batch.reshape(-1, self.input_feature)
        self.w = self.w.reshape(self.input_feature, self.output_feature)
        Wh = torch.mm(h, self.w)
        attention = self._prepare_attentional_mechanism_input(Wh)
        attention = torch.nn.functional.softmax(attention, dim=1)  # 每行做Softmax，相当于每个节点做softmax
        attention = torch.nn.functional.dropout(attention, self.dropout)
        h_prime = torch.mm(attention, Wh)  # 得到下一层的输入

        if self.concat:
            return torch.nn.functional.elu(h_prime)  # 激活
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        Wh1 = torch.matmul(Wh, self.a[:self.output_feature, :])  # N*out_size @ out_size*1 = N*1

        Wh2 = torch.matmul(Wh, self.a[self.output_feature:, :])  # N*1

        e = Wh1 + Wh2.T  # Wh1的每个原始与Wh2的所有元素相加，生成N*N的矩阵
        return self.leakyrelu(e)
