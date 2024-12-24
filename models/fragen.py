import torch
from torch.nn import Module
from torch.nn import functional as F

from .interaction import get_encoder_vn, get_encoder_vn_for_pos
from .interaction.geoattn import Geoattn_GNN
from .interaction.geodesic import Geodesic_GNN
from .model_utils import *
from .embedding import AtomEmbedding
from .generation import get_field_vn  # topology generation
from .generation import FrontierLayerVN  # attachment
from .generation import PositionPredictor # geometry generation
from utils.misc import unique

class Fragen(Module):

    def __init__(self, config, num_classes, num_bond_types, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config
        self.num_bond_types = num_bond_types
        
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec] # 创建了一个列表self.emb_dim，包含了隐藏通道的数量和隐藏向量的数量。
        self.protein_atom_emb = AtomEmbedding(protein_atom_feature_dim, 1, *self.emb_dim) 
        self.ligand_atom_emb = AtomEmbedding(ligand_atom_feature_dim, 1, *self.emb_dim)
        """AtomEmbedding是一个用于处理原子特征的类，它的主要功能是将输入的标量和向量特征转换为嵌入特征。
        这个类的构造函数接受输入标量、输入向量、输出标量、输出向量和向量标准化器作为参数。
        在这段代码中，protein_atom_feature_dim和ligand_atom_feature_dim分别是蛋白质原子特征的维度和配体原子特征的维度，1是输入向量的维度，*self.emb_dim是输出标量和输出向量的维度。
        self.protein_atom_emb和self.ligand_atom_emb在后续的操作中可能会被用来处理蛋白质原子和配体原子的特征，将它们转换为嵌入特征。"""
        
        self.topologic_learner = Geodesic_GNN(node_sca_dim=self.emb_dim[0], node_vec_dim=self.emb_dim[1])
        """创建一个Geodesic_GNN类的实例，名为self.topologic_learner。Geodesic_GNN是一个地理信息网络（Geodesic Graph Neural Network）的实现，用于处理图结构数据。
        在创建Geodesic_GNN实例时，传入了两个参数：node_sca_dim和node_vec_dim。这两个参数分别对应于节点的标量维度和向量维度，它们的值从self.emb_dim中获取。
        self.emb_dim是一个包含两个元素的列表，第一个元素是节点的标量维度，第二个元素是节点的向量维度。"""
        
        self.geometry_learner = Geoattn_GNN(node_sca_dim=self.emb_dim[0], node_vec_dim=self.emb_dim[1])
        """self.geometry_learner = Geoattn_GNN(node_sca_dim=self.emb_dim[0], node_vec_dim=self.emb_dim[1])这行代码创建了一个Geoattn_GNN的实例，并将其赋值给self.geometry_learner。
        在创建实例时，传入了两个参数：node_sca_dim和node_vec_dim。这两个参数分别表示节点的标量维度和向量维度，它们的值从self.emb_dim中获取，
        self.emb_dim是一个包含两个元素的列表，第一个元素是标量维度，第二个元素是向量维度。
        Geoattn_GNN类的定义中包含了许多神经网络层和函数，用于处理节点和边的特性，计算注意力分数，传递和聚合消息，以及进行残差连接和特性映射。
        这个类的forward方法定义了如何使用这些层和函数来进行前向传播。"""
        
        self.encoder = get_encoder_vn(config.encoder)
        """get_encoder_vn函数被用于创建编码器对象。这个函数接收一个配置对象config.encoder作为输入，然后根据配置对象的属性来创建并返回一个特定的编码器。
        具体来说，如果配置对象的name属性值为'cftfm'，则函数会创建一个TransformerFeatureMixer对象。
        否则，函数会抛出一个NotImplementedError异常，表示请求的操作尚未实现。"""
        
        self.encoder_for_pos = get_encoder_vn_for_pos(config.encoder)
        
        in_sca, in_vec = self.encoder.out_sca, self.encoder.out_vec 
        """out_sca属性返回./models/interaction/interaction.py/TransformerFeatureMixer中self.hidden_channels[0]的值。
        out_vec属性返回./models/interaction/interaction.py/TransformerFeatureMixer中self.hidden_channels[1]的值。"""
        
        self.field = get_field_vn(config.field, num_classes=num_classes, num_bond_types=num_bond_types, 
                                             in_sca=in_sca, in_vec=in_vec)
        
        self.frontier_pred = FrontierLayerVN(in_sca=in_sca, in_vec=in_vec,
                                                                            hidden_dim_sca=128, hidden_dim_vec=32)
        # self.protein_frontier_pred = FrontierLayerVN(in_sca=in_sca, in_vec=in_vec,
        #                                                                     hidden_dim_sca=128, hidden_dim_vec=32)
        self.pos_predictor = PositionPredictor(in_sca=in_sca, in_vec=in_vec,
                                                                            num_filters=[config.position.num_filters]*2, n_component=config.position.n_component)

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction='mean', smoothing=0.1)
        self.bceloss_with_logits = nn.BCEWithLogitsLoss()

        # 键长约束 [新增代码]
        self.bond_length_range = (1.0, 2.0)

    def sample_init(self,
        compose_feature,
        compose_pos,
        idx_protein,
        gds_edge_sca, gds_knn_edge_index, gds_dist,
        compose_knn_edge_index,compose_knn_edge_feature,
        n_samples_pos=-1,
        n_samples_atom=-1
        ):
        idx_ligand = torch.empty(0).to(idx_protein)  # fake index of ligand
        focal_resutls = self.sample_focal(compose_feature, compose_pos, idx_ligand, idx_protein, gds_edge_sca, gds_knn_edge_index, gds_dist, 
                                                compose_knn_edge_index, compose_knn_edge_feature)
        if focal_resutls[0]:  # has frontiers
            has_frontier, idx_frontier, p_frontier, idx_focal_in_compose, p_focal, h_compose = focal_resutls # type: ignore
            pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi = self.sample_position(
                h_compose, compose_pos, idx_focal_in_compose, n_samples=n_samples_pos,
            )
            idx_focal_in_compose, p_focal = idx_focal_in_compose[idx_parent], p_focal[idx_parent]
            element_pred, element_prob, has_atom_prob, idx_parent = self.sample_init_element(
                pos_generated, h_compose, compose_pos, n_samples=n_samples_atom
            )
            pos_generated, pdf_pos, idx_focal_in_compose, p_focal = pos_generated[idx_parent], pdf_pos[idx_parent], idx_focal_in_compose[idx_parent], p_focal[idx_parent]
            return (has_frontier, idx_frontier, p_frontier,  # frontier
                        idx_focal_in_compose, p_focal,  # focal
                        pos_generated, pdf_pos, abs_pos_mu, pos_sigma, pos_pi,  # positions
                        element_pred, element_prob, has_atom_prob,  # element
                        )
        else:
            return (False, )

    def sample(self,
        compose_feature,
        compose_pos,
        idx_ligand,
        idx_protein,
        gds_edge_sca, gds_knn_edge_index, gds_dist,
        compose_knn_edge_index, compose_knn_edge_feature,
        ligand_context_bond_index,
        ligand_context_bond_type,
        n_samples_pos=-1,
        n_samples_atom=-1,
        # n_samples=5,
        frontier_threshold=0,
        freeze=None,
        anchor=None,
        ):
        focal_resutls = self.sample_focal(compose_feature, compose_pos, idx_ligand, idx_protein, gds_edge_sca, gds_knn_edge_index, gds_dist,
                                          compose_knn_edge_index, compose_knn_edge_feature, frontier_threshold=frontier_threshold, freeze=freeze,anchor=anchor)
        if focal_resutls[0]:  # has frontiers
            has_frontier, idx_frontier, p_frontier, idx_focal_in_compose, p_focal, h_compose = focal_resutls # type: ignore
            pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi = self.sample_position(
                h_compose, compose_pos, idx_focal_in_compose, n_samples=n_samples_pos
            )
            idx_focal_in_compose, p_focal = idx_focal_in_compose[idx_parent], p_focal[idx_parent]
            element_pred, element_prob, has_atom_prob, idx_parent, bond_index, bond_type, bond_prob = self.sample_element_and_bond(
                pos_generated, h_compose, compose_pos, idx_ligand, ligand_context_bond_index, ligand_context_bond_type, n_samples=n_samples_atom
            )
            pos_generated, pdf_pos, idx_focal_in_compose, p_focal = pos_generated[idx_parent], pdf_pos[idx_parent], idx_focal_in_compose[idx_parent], p_focal[idx_parent]
            return (has_frontier, idx_frontier, p_frontier,  # frontier
                        idx_focal_in_compose, p_focal,  # focal
                        pos_generated, pdf_pos, abs_pos_mu, pos_sigma, pos_pi,  # positions
                        element_pred, element_prob, has_atom_prob,  # element
                        bond_index, bond_type, bond_prob  # bond
                        )
        else:
            return (False, )

    def sample_focal(self,
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            gds_edge_sca, gds_knn_edge_index, gds_dist,
            compose_knn_edge_index,compose_knn_edge_feature,
            n_samples=-1,
            frontier_threshold=0,
            anchor = None,
            freeze = None,
            force_search = None,
        ):
        '''
        Optional:
        force_search: a integral to force the focal atom selection. E.g., force_search=3 means 
            select 3 candidates for focal atom determination 
        '''
        # # 0: encode 
        h_compose = interaction_embed(compose_feature, compose_pos, idx_ligand, idx_protein,
                                    gds_edge_sca, gds_knn_edge_index, gds_dist,
                                    compose_knn_edge_feature, compose_knn_edge_index,
                                        self.ligand_atom_emb, self.protein_atom_emb, self.topologic_learner, self.geometry_learner, self.emb_dim)
        
        h_compose_for_pos = self.encoder_for_pos(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        )
        
        h_compose = self.encoder(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        ) 
        
        h_compose[0] = h_compose[0] * h_compose_for_pos[0]
        h_compose[1] = h_compose[1] * h_compose_for_pos[1]
        
        # # For the initial atom
        if len(idx_ligand) == 0:
            idx_ligand = idx_protein
        # # 1: predict frontier
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_ligand,
        )[:, 0]

        # make the constraint, freeze refers to frozen atoms having no choice to be selected
        if freeze is not None:
            y_frontier_pred[freeze] = -torch.ones(y_frontier_pred[freeze].shape[0], device=y_frontier_pred.device) + frontier_threshold
        # anchor refers to the atoms which having probability to be selected
        if anchor is not None:
            y_frontier_pred[anchor] = torch.ones(y_frontier_pred[anchor].shape[0],device=y_frontier_pred.device) + frontier_threshold

        ind_frontier = (y_frontier_pred > frontier_threshold)
        if force_search is not None:
            _ , idx_tmp = torch.sort(y_frontier_pred, descending=True) #descending为alse，升序，为True，降序
            idx = idx_tmp[:force_search]
            y_frontier_pred_bool = torch.zeros(y_frontier_pred.shape[0], device=y_frontier_pred.device)
            for i in idx:
                y_frontier_pred_bool[i] = 1
            ind_frontier = (y_frontier_pred_bool==1)

        has_frontier = torch.sum(ind_frontier) > 0
        frontier_scale = 1
        if has_frontier:
            # # 2: sample focal from frontiers
            idx_frontier = idx_ligand[ind_frontier]
            p_frontier = torch.sigmoid(y_frontier_pred[ind_frontier])
            if n_samples > 0:  # sample from frontiers
                p_frontier_in_compose = torch.zeros(len(compose_pos), dtype=torch.float32, device=compose_pos.device)
                p_frontier_in_compose_sf = torch.zeros_like(p_frontier_in_compose)
                p_frontier_in_compose_sf[idx_frontier] = F.softmax(p_frontier / frontier_scale, dim=0)
                p_frontier_in_compose[idx_frontier] = p_frontier
                idx_focal_in_compose = p_frontier_in_compose_sf.multinomial(num_samples=n_samples, replacement=True)
                p_focal = p_frontier_in_compose[idx_focal_in_compose]
            else:  # get all frontiers as focal
                idx_focal_in_compose = torch.nonzero(ind_frontier)[:, 0]
                p_focal = p_frontier

            return (has_frontier, idx_frontier, p_frontier,  # frontier
                        idx_focal_in_compose, p_focal,  # focal
                        h_compose)
        else:
            return (has_frontier, h_compose)

    def sample_focal_constriant(self,
            compose_feature,
            compose_pos,
            idx_ligand,
            idx_protein,
            gds_edge_sca, gds_knn_edge_index, gds_dist,
            compose_knn_edge_index,compose_knn_edge_feature,
            n_samples=-1,
            frontier_threshold=0,
            anchor = None,
            freeze = None
        ):
        # # 0: encode 
        h_compose = interaction_embed(compose_feature, compose_pos, idx_ligand, idx_protein,
                                    gds_edge_sca, gds_knn_edge_index, gds_dist,
                                    compose_knn_edge_feature, compose_knn_edge_index,
         
                                        self.ligand_atom_emb, self.protein_atom_emb, self.topologic_learner, self.geometry_learner, self.emb_dim)
        # # Encode compose
        h_compose_for_pos = self.encoder_for_pos(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        )
        
        h_compose = self.encoder(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        )
        
        h_compose[0] = h_compose[0] * h_compose_for_pos[0]
        h_compose[1] = h_compose[1] * h_compose_for_pos[1]
        
        # # For the initial atom
        if len(idx_ligand) == 0:
            idx_ligand = idx_protein
        # # 1: predict frontier
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_ligand,
        )[:, 0]

        # make the constraint, freeze refers to frozen atoms having no choice to be selected
        if freeze is not None:
            y_frontier_pred[freeze] = -torch.ones(y_frontier_pred[freeze].shape[0])
        # anchor refers to the atoms which having probability to be selected
        if anchor is not None:
            y_frontier_pred[~anchor] = -torch.ones(y_frontier_pred[~anchor].shape[0])

        ind_frontier = (y_frontier_pred > frontier_threshold)
        has_frontier = torch.sum(ind_frontier) > 0
        frontier_scale = 1
        
        if has_frontier:
            # # 2: sample focal from frontiers
            idx_frontier = idx_ligand[ind_frontier]
            p_frontier = torch.sigmoid(y_frontier_pred[ind_frontier])
            if n_samples > 0:  # sample from frontiers
                p_frontier_in_compose = torch.zeros(len(compose_pos), dtype=torch.float32, device=compose_pos.device)
                p_frontier_in_compose_sf = torch.zeros_like(p_frontier_in_compose)
                p_frontier_in_compose_sf[idx_frontier] = F.softmax(p_frontier / frontier_scale, dim=0)
                p_frontier_in_compose[idx_frontier] = p_frontier
                idx_focal_in_compose = p_frontier_in_compose_sf.multinomial(num_samples=n_samples, replacement=True)
                p_focal = p_frontier_in_compose[idx_focal_in_compose]
            else:  # get all frontiers as focal
                idx_focal_in_compose = torch.nonzero(ind_frontier)[:, 0]
                p_focal = p_frontier

            return (has_frontier, idx_frontier, p_frontier,  # frontier
                        idx_focal_in_compose, p_focal,  # focal
                        h_compose)
        else:
            return (has_frontier, h_compose)

    def sample_position(self,
        h_compose,
        compose_pos,
        idx_focal_in_compose,
        n_samples=-1,
        ):
        n_focals = len(idx_focal_in_compose)
        # # 3: get position distributions and sample positions
        relative_pos_mu, abs_pos_mu, pos_sigma, pos_pi  = self.pos_predictor(
            h_compose,
            idx_focal_in_compose,
            compose_pos,
        )
        if n_samples < 0:
            pos_generated = self.pos_predictor.get_maximum(abs_pos_mu, pos_sigma, pos_pi,)  # n_focals, n_per_pos, 3
            n_candidate_samples = pos_generated.size(1)
            pos_generated = torch.reshape(pos_generated, [-1, 3])
            pdf_pos = self.pos_predictor.get_mdn_probability(
                mu=torch.repeat_interleave(abs_pos_mu, repeats=n_candidate_samples, dim=0),
                sigma=torch.repeat_interleave(pos_sigma, repeats=n_candidate_samples, dim=0),
                pi=torch.repeat_interleave(pos_pi, repeats=n_candidate_samples, dim=0),
                pos_target=pos_generated
            )
            idx_parent = torch.repeat_interleave(torch.arange(n_focals), repeats=n_candidate_samples, dim=0).to(compose_pos.device)

        return (pos_generated, pdf_pos, idx_parent, abs_pos_mu, pos_sigma, pos_pi)  # position

    def sample_element_and_bond(self, 
        pos_generated,
        h_compose,
        compose_pos,
        idx_ligand,
        ligand_bond_index,
        ligand_bond_type,
        n_samples
        ):
        # # 4: query positions 
        #NOTE: Only one parent batch (one compose graph) at a time (i.e. batch size = 1)
        n_query = len(pos_generated)
        n_context = len(idx_ligand)
        y_query_pred, edge_pred = self.query_position(
            pos_query = pos_generated,
            h_compose = h_compose,
            compose_pos = compose_pos,
            idx_ligand = idx_ligand,
            ligand_bond_index = ligand_bond_index,
            ligand_bond_type = ligand_bond_type
        )
        if n_samples < 0: #
            # raise NotImplementedError('The following is not fixed (and for/ bond)')
            has_atom_prob =  1 - 1 / (1 + torch.exp(y_query_pred).sum(-1))
            y_query_pred = F.softmax(y_query_pred, dim=-1)
            element_pred = y_query_pred.argmax(dim=-1)  # multinomial(1)[:, 0]
            element_prob = y_query_pred[torch.arange(len(y_query_pred)), element_pred]
            idx_parent = torch.arange(n_query)
        else:
            has_atom_prob =  (1 - 1 / (1 + torch.exp(y_query_pred).sum(-1)))
            has_atom_prob = torch.repeat_interleave(has_atom_prob, n_samples, dim=0)  # n_query * n_samples
            y_query_pred = F.softmax(y_query_pred, dim=-1)
            element_pred = y_query_pred.multinomial(n_samples, replacement=True).reshape(-1)  # n_query * n_samples
            idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(compose_pos.device)
            element_prob = y_query_pred[idx_parent, element_pred]
        # # 5: determine bonds
        if n_samples < 0:
            all_edge_type = torch.argmax(edge_pred, dim=-1) # (num_generated, num_ligand_context)
            bond_index = torch.stack(torch.where(
                all_edge_type > 0,
            ), dim=0)
            bond_type = all_edge_type[bond_index[0], bond_index[1]]
            bond_prob = F.softmax(edge_pred, dim=-1)[bond_index[0], bond_index[1], bond_type]
        else:
            edge_pred = F.softmax(edge_pred, dim=-1)  # (num_query, num_context, 4)
            edge_pred_flat = edge_pred.reshape([n_query * n_context, -1])  # (num_query * num_context, 4)
            all_edge_type = edge_pred_flat.multinomial(n_samples, replacement=True)  # (num_query * num_context, n_samples)
            all_edge_type = all_edge_type.reshape([n_query, n_context, n_samples])  # (num_query, num_context, n_samples)
            all_edge_type = all_edge_type.transpose(1, 2)  # (num_query, n_samples, num_context)
            all_edge_type = all_edge_type.reshape([n_query * n_samples, n_context]) # (num_generated * n_samples, num_ligand_context)
            
            # drop duplicates
            id_element_and_bond = torch.cat([idx_parent.unsqueeze(-1), element_pred.unsqueeze(-1), all_edge_type], dim=1)
            id_element_and_bond, index_unique = unique(id_element_and_bond, dim=0) 
            # all_edge_type = all_edge_type[index_unique]
            element_pred, element_prob, has_atom_prob, idx_parent = element_pred[index_unique], element_prob[index_unique], has_atom_prob[index_unique], idx_parent[index_unique]

            # get bond index
            all_edge_type = all_edge_type[index_unique]
            bond_index = torch.stack(torch.where(
                all_edge_type > 0,
            ), dim=0)
            bond_type = all_edge_type[bond_index[0], bond_index[1]]
            bond_prob = edge_pred[idx_parent[bond_index[0]], bond_index[1], bond_type]
        
            
        return (element_pred, element_prob, has_atom_prob, idx_parent, # element
                    bond_index, bond_type, bond_prob  # bond
                    )
    
    # def sample_element_and_bond(self, 
        #     pos_generated,  # 新生成的原子位置
        #     h_compose,  # 复合物的上下文特征
        #     compose_pos,  # 复合物中原子的位置信息
        #     idx_ligand,  # 配体原子的索引
        #     ligand_bond_index,  # 配体中化学键的索引
        #     ligand_bond_type,  # 配体中化学键的类型
        #     n_samples  # 采样的数量，决定是否进行多样性采样
        #     ):
        # """
        # 函数功能：对生成的原子位置进行化学元素和化学键的采样，并返回采样结果。
        # 参数说明：
        # - pos_generated: 新生成的原子位置。
        # - h_compose: 复合物的上下文特征（例如原子类型嵌入）。
        # - compose_pos: 复合物中所有原子的位置信息。
        # - idx_ligand: 配体中原子的索引列表。
        # - ligand_bond_index: 配体中化学键的索引。
        # - ligand_bond_type: 配体中化学键的类型。
        # - n_samples: 采样的数量，`n_samples < 0` 表示不进行多样性采样，只选择一个结果。
        # """

        # # 第一步：获取生成原子位置与复合物的交互预测结果
        # # n_query 是生成原子的数量，n_context 是配体原子的数量
        # n_query = len(pos_generated)  # 查询的生成原子数量
        # n_context = len(idx_ligand)  # 配体的原子数量

        # # 调用 query_position 函数，预测生成原子所属的化学元素和化学键的类型
        # y_query_pred, edge_pred = self.query_position(
        #     pos_query = pos_generated,  # 新生成的原子位置
        #     h_compose = h_compose,  # 复合物的上下文特征
        #     compose_pos = compose_pos,  # 复合物中所有原子的位置
        #     idx_ligand = idx_ligand,  # 配体中原子的索引
        #     ligand_bond_index = ligand_bond_index,  # 配体中化学键的索引
        #     ligand_bond_type = ligand_bond_type  # 配体中化学键的类型
        # )

        # # 如果不进行多样性采样（n_samples < 0）
        # if n_samples < 0: 
        #     # 计算生成原子是否为真实原子的概率
        #     has_atom_prob = 1 - 1 / (1 + torch.exp(y_query_pred).sum(-1))  # 使用 sigmoid 函数计算概率
        #     y_query_pred = F.softmax(y_query_pred, dim=-1)  # 对预测的化学元素概率进行 softmax 归一化
        #     element_pred = y_query_pred.argmax(dim=-1)  # 选取概率最大的化学元素作为预测结果
        #     element_prob = y_query_pred[torch.arange(len(y_query_pred)), element_pred]  # 获取对应元素的概率
        #     idx_parent = torch.arange(n_query)  # 每个生成原子对应的索引
        # else:  # 如果进行多样性采样（n_samples >= 0）
        #     # 计算生成原子是否为真实原子的概率，并进行重复采样
        #     has_atom_prob = (1 - 1 / (1 + torch.exp(y_query_pred).sum(-1)))  # 同样计算是否为真实原子的概率
        #     has_atom_prob = torch.repeat_interleave(has_atom_prob, n_samples, dim=0)  # 重复采样 n_samples 次
            
        #     # 对化学元素的概率进行 softmax 归一化
        #     y_query_pred = F.softmax(y_query_pred, dim=-1)

        #     # 多样性采样：对每个生成原子的位置，采样 n_samples 次化学元素
        #     element_pred = y_query_pred.multinomial(n_samples, replacement=True).reshape(-1)  # 采样结果展平
        #     idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(compose_pos.device)  # 每个生成原子重复 n_samples 次
        #     element_prob = y_query_pred[idx_parent, element_pred]  # 获取采样结果对应的概率

        # # 第二步：确定化学键
        # if n_samples < 0:  # 如果不进行多样性采样
        #     # 获取化学键类型的预测结果
        #     all_edge_type = torch.argmax(edge_pred, dim=-1)  # 对每个生成原子与配体原子，选取概率最大的键类型
        #     bond_index = torch.stack(torch.where(
        #         all_edge_type > 0,  # 筛选出预测为有化学键的边
        #     ), dim=0)  # 获取化学键的索引
        #     bond_type = all_edge_type[bond_index[0], bond_index[1]]  # 获取化学键的类型
        #     bond_prob = F.softmax(edge_pred, dim=-1)[bond_index[0], bond_index[1], bond_type]  # 获取化学键的概率
        # else:  # 如果进行多样性采样
        #     # 对化学键的概率进行 softmax 归一化
        #     edge_pred = F.softmax(edge_pred, dim=-1)  # (num_query, num_context, 4)
        #     edge_pred_flat = edge_pred.reshape([n_query * n_context, -1])  # 将化学键的维度展平
        #     all_edge_type = edge_pred_flat.multinomial(n_samples, replacement=True)  # 对每个（生成原子, 配体原子）采样 n_samples 次键类型
        #     all_edge_type = all_edge_type.reshape([n_query, n_context, n_samples])  # 恢复维度
        #     all_edge_type = all_edge_type.transpose(1, 2)  # 转置以方便后续操作
        #     all_edge_type = all_edge_type.reshape([n_query * n_samples, n_context])  # 展平为 (num_generated * n_samples, num_ligand_context)

        #     # 去除重复的采样结果
        #     id_element_and_bond = torch.cat([idx_parent.unsqueeze(-1), element_pred.unsqueeze(-1), all_edge_type], dim=1)  # 将化学元素和化学键拼接
        #     id_element_and_bond, index_unique = unique(id_element_and_bond, dim=0)  # 去重
        #     # 更新采样结果为去重后的结果
        #     element_pred, element_prob, has_atom_prob, idx_parent = element_pred[index_unique], element_prob[index_unique], has_atom_prob[index_unique], idx_parent[index_unique]

        #     # 获取去重后的化学键
        #     all_edge_type = all_edge_type[index_unique]
        #     bond_index = torch.stack(torch.where(
        #         all_edge_type > 0,  # 筛选出预测为有化学键的边
        #     ), dim=0)  # 获取化学键的索引
        #     bond_type = all_edge_type[bond_index[0], bond_index[1]]  # 获取化学键的类型
        #     bond_prob = edge_pred[idx_parent[bond_index[0]], bond_index[1], bond_type]  # 获取化学键的概率

        # # 返回化学元素和化学键的采样结果
        # return (element_pred,  # 采样的化学元素
        #         element_prob,  # 化学元素的概率
        #         has_atom_prob,  # 是否为真实原子的概率
        #         idx_parent,  # 每个生成原子的父索引
        #         bond_index,  # 化学键的索引
        #         bond_type,  # 化学键的类型
        #         bond_prob)  # 化学键的概率

    # def sample_element_and_bond(self, 
    #         pos_generated,  # 新生成的原子位置
    #         h_compose,  # 复合物的上下文特征
    #         compose_pos,  # 复合物中原子的位置信息
    #         idx_ligand,  # 配体原子的索引
    #         ligand_bond_index,  # 配体中化学键的索引
    #         ligand_bond_type,  # 配体中化学键的类型
    #         n_samples  # 采样的数量，决定是否进行多样性采样
    #         ):
    #     """
    #     函数功能：对生成的原子位置进行化学元素和化学键的采样，并返回采样结果。
    #     参数说明：
    #     - pos_generated: 新生成的原子位置。
    #     - h_compose: 复合物的上下文特征（例如原子类型嵌入）。
    #     - compose_pos: 复合物中所有原子的位置信息。
    #     - idx_ligand: 配体中原子的索引列表。
    #     - ligand_bond_index: 配体中化学键的索引。
    #     - ligand_bond_type: 配体中化学键的类型。
    #     - n_samples: 采样的数量，`n_samples < 0` 表示不进行多样性采样，只选择一个结果。
    #     """

    #     # 第一步：获取生成原子位置与复合物的交互预测结果
    #     # n_query 是生成原子的数量，n_context 是配体原子的数量
    #     n_query = len(pos_generated)  # 查询的生成原子数量
    #     n_context = len(idx_ligand)  # 配体的原子数量

    #     # 调用 query_position 函数，预测生成原子所属的化学元素和化学键的类型
    #     y_query_pred, edge_pred = self.query_position(
    #         pos_query = pos_generated,  # 新生成的原子位置
    #         h_compose = h_compose,  # 复合物的上下文特征
    #         compose_pos = compose_pos,  # 复合物中所有原子的位置
    #         idx_ligand = idx_ligand,  # 配体中原子的索引
    #         ligand_bond_index = ligand_bond_index,  # 配体中化学键的索引
    #         ligand_bond_type = ligand_bond_type  # 配体中化学键的类型
    #     )

    #     # 如果不进行多样性采样（n_samples < 0）
    #     if n_samples < 0: 
    #         # 计算生成原子是否为真实原子的概率
    #         has_atom_prob = 1 - 1 / (1 + torch.exp(y_query_pred).sum(-1))  # 使用 sigmoid 函数计算概率
    #         y_query_pred = F.softmax(y_query_pred, dim=-1)  # 对预测的化学元素概率进行 softmax 归一化
    #         element_pred = y_query_pred.argmax(dim=-1)  # 选取概率最大的化学元素作为预测结果
    #         element_prob = y_query_pred[torch.arange(len(y_query_pred)), element_pred]  # 获取对应元素的概率
    #         idx_parent = torch.arange(n_query)  # 每个生成原子对应的索引
    #     else:  # 如果进行多样性采样（n_samples >= 0）
    #         # 计算生成原子是否为真实原子的概率，并进行重复采样
    #         has_atom_prob = (1 - 1 / (1 + torch.exp(y_query_pred).sum(-1)))  # 同样计算是否为真实原子的概率
    #         has_atom_prob = torch.repeat_interleave(has_atom_prob, n_samples, dim=0)  # 重复采样 n_samples 次
            
    #         # 对化学元素的概率进行 softmax 归一化
    #         y_query_pred = F.softmax(y_query_pred, dim=-1)

    #         # 多样性采样：对每个生成原子的位置，采样 n_samples 次化学元素
    #         element_pred = y_query_pred.multinomial(n_samples, replacement=True).reshape(-1)  # 采样结果展平
    #         idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(compose_pos.device)  # 每个生成原子重复 n_samples 次
    #         element_prob = y_query_pred[idx_parent, element_pred]  # 获取采样结果对应的概率

    #     # 第二步：确定化学键
    #     if n_samples < 0:  # 如果不进行多样性采样
    #         # 获取化学键类型的预测结果
    #         all_edge_type = torch.argmax(edge_pred, dim=-1)  # 对每个生成原子与配体原子，选取概率最大的键类型
    #         bond_index = torch.stack(torch.where(
    #             all_edge_type > 0,  # 筛选出预测为有化学键的边
    #         ), dim=0)  # 获取化学键的索引
    #         bond_type = all_edge_type[bond_index[0], bond_index[1]]  # 获取化学键的类型
    #         bond_prob = F.softmax(edge_pred, dim=-1)[bond_index[0], bond_index[1], bond_type]  # 获取化学键的概率
    #     else:  # 如果进行多样性采样
    #         # 对化学键的概率进行 softmax 归一化
    #         edge_pred = F.softmax(edge_pred, dim=-1)  # (num_query, num_context, 4)
    #         edge_pred_flat = edge_pred.reshape([n_query * n_context, -1])  # 将化学键的维度展平
    #         all_edge_type = edge_pred_flat.multinomial(n_samples, replacement=True)  # 对每个（生成原子, 配体原子）采样 n_samples 次键类型
    #         all_edge_type = all_edge_type.reshape([n_query, n_context, n_samples])  # 恢复维度
    #         all_edge_type = all_edge_type.transpose(1, 2)  # 转置以方便后续操作
    #         all_edge_type = all_edge_type.reshape([n_query * n_samples, n_context])  # 展平为 (num_generated * n_samples, num_ligand_context)

    #         # 去除重复的采样结果
    #         id_element_and_bond = torch.cat([idx_parent.unsqueeze(-1), element_pred.unsqueeze(-1), all_edge_type], dim=1)  # 将化学元素和化学键拼接
    #         id_element_and_bond, index_unique = unique(id_element_and_bond, dim=0)  # 去重
    #         # 更新采样结果为去重后的结果
    #         element_pred, element_prob, has_atom_prob, idx_parent = element_pred[index_unique], element_prob[index_unique], has_atom_prob[index_unique], idx_parent[index_unique]

    #         # 获取去重后的化学键
    #         all_edge_type = all_edge_type[index_unique]
    #         bond_index = torch.stack(torch.where(
    #             all_edge_type > 0,  # 筛选出预测为有化学键的边
    #         ), dim=0)  # 获取化学键的索引
    #         bond_type = all_edge_type[bond_index[0], bond_index[1]]  # 获取化学键的类型
    #         bond_prob = edge_pred[idx_parent[bond_index[0]], bond_index[1], bond_type]  # 获取化学键的概率

    #     # 返回化学元素和化学键的采样结果
    #     return (element_pred,  # 采样的化学元素
    #             element_prob,  # 化学元素的概率
    #             has_atom_prob,  # 是否为真实原子的概率
    #             idx_parent,  # 每个生成原子的父索引
    #             bond_index,  # 化学键的索引
    #             bond_type,  # 化学键的类型
    #             bond_prob)  # 化学键的概率

    def sample_element_and_bond(self, 
        pos_generated,
        h_compose,
        compose_pos,
        idx_ligand,
        ligand_bond_index,
        ligand_bond_type,
        n_samples
        ):
        # # 4: query positions 
        #NOTE: Only one parent batch (one compose graph) at a time (i.e. batch size = 1)
        n_query = len(pos_generated)
        n_context = len(idx_ligand)
        y_query_pred, edge_pred = self.query_position(
            pos_query = pos_generated,
            h_compose = h_compose,
            compose_pos = compose_pos,
            idx_ligand = idx_ligand,
            ligand_bond_index = ligand_bond_index,
            ligand_bond_type = ligand_bond_type
        )

        # 处理原子类型预测
        if n_samples < 0:
            has_atom_prob =  1 - 1 / (1 + torch.exp(y_query_pred).sum(-1))
            y_query_pred = F.softmax(y_query_pred, dim=-1)
            element_pred = y_query_pred.argmax(dim=-1)
            element_prob = y_query_pred[torch.arange(len(y_query_pred)), element_pred]
            idx_parent = torch.arange(n_query)
        else:
            has_atom_prob =  (1 - 1 / (1 + torch.exp(y_query_pred).sum(-1)))
            has_atom_prob = torch.repeat_interleave(has_atom_prob, n_samples, dim=0)
            y_query_pred = F.softmax(y_query_pred, dim=-1)
            element_pred = y_query_pred.multinomial(n_samples, replacement=True).reshape(-1)
            idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(compose_pos.device)
            element_prob = y_query_pred[idx_parent, element_pred]

        # 处理化学键预测
        if n_samples < 0:
            # 计算所有原子对之间的距离
            dist_matrix = torch.cdist(pos_generated[idx_parent], compose_pos[idx_ligand])
            # 创建键长掩码
            bond_length_mask = (dist_matrix >= 1.0) & (dist_matrix <= 2.0)
            
            all_edge_type = torch.argmax(edge_pred, dim=-1)
            # 应用键长掩码：如果距离不在有效范围内，将边类型设为0（无键）
            all_edge_type = torch.where(bond_length_mask, all_edge_type, torch.zeros_like(all_edge_type))
            
            bond_index = torch.stack(torch.where(all_edge_type > 0), dim=0)
            bond_type = all_edge_type[bond_index[0], bond_index[1]]
            bond_prob = F.softmax(edge_pred, dim=-1)[bond_index[0], bond_index[1], bond_type]
        else:
            # 计算所有原子对之间的距离
            dist_matrix = torch.cdist(pos_generated, compose_pos[idx_ligand])
            # 创建键长掩码
            bond_length_mask = (dist_matrix >= 1.0) & (dist_matrix <= 2.0)
            # 重复掩码以匹配采样次数
            bond_length_mask = bond_length_mask.unsqueeze(-1).expand(-1, -1, n_samples)
            bond_length_mask = bond_length_mask.transpose(1, 2)
            bond_length_mask = bond_length_mask.reshape(n_query * n_samples, n_context)
            
            edge_pred = F.softmax(edge_pred, dim=-1)
            edge_pred_flat = edge_pred.reshape([n_query * n_context, -1])
            all_edge_type = edge_pred_flat.multinomial(n_samples, replacement=True)
            all_edge_type = all_edge_type.reshape([n_query, n_context, n_samples])
            all_edge_type = all_edge_type.transpose(1, 2)
            all_edge_type = all_edge_type.reshape([n_query * n_samples, n_context])
            
            # 应用键长掩码
            all_edge_type = torch.where(bond_length_mask, all_edge_type, torch.zeros_like(all_edge_type))
            
            # 去除重复
            id_element_and_bond = torch.cat([idx_parent.unsqueeze(-1), element_pred.unsqueeze(-1), all_edge_type], dim=1)
            id_element_and_bond, index_unique = unique(id_element_and_bond, dim=0)
            element_pred, element_prob, has_atom_prob, idx_parent = element_pred[index_unique], element_prob[index_unique], has_atom_prob[index_unique], idx_parent[index_unique]

            # 获取键信息
            all_edge_type = all_edge_type[index_unique]
            bond_index = torch.stack(torch.where(all_edge_type > 0), dim=0)
            bond_type = all_edge_type[bond_index[0], bond_index[1]]
            bond_prob = edge_pred[idx_parent[bond_index[0]], bond_index[1], bond_type]
                
        return (element_pred, element_prob, has_atom_prob, idx_parent,  # element
                bond_index, bond_type, bond_prob  # bond
                )

    def sample_init_element(self, 
        pos_generated,
        h_compose,
        compose_pos,
        n_samples,
        ):
        # # 4: query positions 
        #NOTE: Only one parent batch (one compose graph) at a time (i.e. batch size = 1)
        n_query = len(pos_generated)
        query_compose_knn_edge_index = knn(x=compose_pos, y=pos_generated, k=self.config.field.knn, num_workers=16)
        y_query_pred, _ = self.field(
            pos_query = pos_generated,
            edge_index_query = [],
            pos_compose = compose_pos,
            node_attr_compose = h_compose,
            edge_index_q_cps_knn = query_compose_knn_edge_index,
        )
        if n_samples < 0:
            # raise NotImplementedError('The following is not fixed')
            has_atom_prob =  1 - 1 / (1 + torch.exp(y_query_pred).sum(-1))
            y_query_pred = F.softmax(y_query_pred, dim=-1)
            element_pred = y_query_pred.argmax(dim=-1)
            element_prob = y_query_pred[torch.arange(len(y_query_pred)), element_pred]
            idx_parent = torch.arange(n_query).to(compose_pos.device)
        else:
            has_atom_prob =  (1 - 1 / (1 + torch.exp(y_query_pred).sum(-1)))
            has_atom_prob = torch.repeat_interleave(has_atom_prob, n_samples, dim=0)  # n_query * n_samples
            y_query_pred = F.softmax(y_query_pred, dim=-1)
            element_pred = y_query_pred.multinomial(n_samples, replacement=True).reshape(-1)  # n_query, n_samples
            idx_parent = torch.repeat_interleave(torch.arange(n_query), n_samples, dim=0).to(compose_pos.device)
            element_prob = y_query_pred[idx_parent, element_pred]
            # drop duplicates
            identifier = torch.stack([idx_parent, element_pred], dim=1)
            identifier, index_unique = unique(identifier, dim=0)

            element_pred, element_prob, has_atom_prob, idx_parent = element_pred[index_unique], element_prob[index_unique], has_atom_prob[index_unique], idx_parent[index_unique]

        return (element_pred, element_prob, has_atom_prob, idx_parent) # element

    def get_loss(self, pos_real, y_real, pos_fake,  # query real positions,
                          index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat,  # for edge attention
                          edge_index_real, edge_label,  # edges to predict
                          compose_feature, compose_pos, idx_ligand, idx_protein,  # compose (protein and context ligand) atoms
                          y_frontier,  # frontier labels
                          idx_focal,  pos_generate,  # focal and generated positions  #NOTE: idx are in comopse
                          idx_protein_all_mask, y_protein_frontier,  # surface of protein
                          gds_edge_sca, gds_knn_edge_index, gds_dist,
                          compose_knn_edge_index, compose_knn_edge_feature, real_compose_knn_edge_index,  fake_compose_knn_edge_index  # edges in compose, query-compose
        ):

        # # emebedding
        h_compose = interaction_embed(compose_feature, compose_pos, idx_ligand, idx_protein,
                                    gds_edge_sca, gds_knn_edge_index, gds_dist,
                                    compose_knn_edge_feature, compose_knn_edge_index,
                                        self.ligand_atom_emb, self.protein_atom_emb, self.topologic_learner, self.geometry_learner, self.emb_dim)
        # # Encode compose
        h_compose_for_pos = self.encoder_for_pos(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        )
        
        h_compose = self.encoder(
            node_attr = h_compose,
            pos = compose_pos,
            edge_index = compose_knn_edge_index,
            edge_feature = compose_knn_edge_feature,
        )   # (N_p+N_l, H)    
                
        h_compose[0] = h_compose[0] * h_compose_for_pos[0]
        h_compose[1] = h_compose[1] * h_compose_for_pos[1]
        
        # # 0: frontier atoms of protein
        y_protein_frontier_pred = self.frontier_pred(
            h_compose,
            idx_protein_all_mask
        )
        # # 1: Fontier atoms
        y_frontier_pred = self.frontier_pred(
            h_compose,
            idx_ligand,
        )
        # # 2: Positions relative to focal atoms  `idx_focal`
        relative_pos_mu, abs_pos_mu, pos_sigma, pos_pi  = self.pos_predictor(
            h_compose,
            idx_focal,
            compose_pos,
        )

        # # 3: Element and bonds of the new position atoms
        y_real_pred, edge_pred = self.field(
            pos_query = pos_real,
            edge_index_query = edge_index_real,
            pos_compose = compose_pos,
            node_attr_compose = h_compose,
            edge_index_q_cps_knn = real_compose_knn_edge_index,

            index_real_cps_edge_for_atten = index_real_cps_edge_for_atten,
            tri_edge_index = tri_edge_index,
            tri_edge_feat = tri_edge_feat
        )   # (N_real, num_classes)

        # # fake positions
        y_fake_pred,  _ = self.field(
            pos_query = pos_fake,
            edge_index_query = [],
            pos_compose = compose_pos,
            node_attr_compose = h_compose,
            edge_index_q_cps_knn = fake_compose_knn_edge_index,
        )   # (N_fake, num_classes)

        # # loss
        loss_surf = F.binary_cross_entropy_with_logits(
            input=y_protein_frontier_pred,
            target=y_protein_frontier.view(-1, 1).float()
        ).clamp_max(10.)
        loss_frontier = F.binary_cross_entropy_with_logits(
            input = y_frontier_pred,
            target = y_frontier.view(-1, 1).float()
        ).clamp_max(10.)
        loss_pos = -torch.log(
            self.pos_predictor.get_mdn_probability(abs_pos_mu, pos_sigma, pos_pi, pos_generate) + 1e-16
        ).mean().clamp_max(10.)
        # loss_notpos = self.pos_predictor.get_mdn_probability(abs_pos_mu, pos_sigma, pos_pi, pos_notgenerate).mean()
        loss_cls = self.smooth_cross_entropy(y_real_pred, y_real.argmax(-1)).clamp_max(10.)    # Classes
        loss_edge = F.cross_entropy(edge_pred, edge_label).clamp_max(10.)
        # real and fake loss
        energy_real = -1 *  torch.logsumexp(y_real_pred, dim=-1)  # (N_real)
        energy_fake = -1 * torch.logsumexp(y_fake_pred, dim=-1)   # (N_fake)
        energy_real = torch.clamp_max(energy_real, 40)
        energy_fake = torch.clamp_min(energy_fake, -40)
        loss_real = self.bceloss_with_logits(-energy_real, torch.ones_like(energy_real)).clamp_max(10.)
        loss_fake = self.bceloss_with_logits(-energy_fake, torch.zeros_like(energy_fake)).clamp_max(10.)

        loss = (torch.nan_to_num(loss_frontier)
                    + torch.nan_to_num(loss_pos)
                    + torch.nan_to_num(loss_cls)
                    + torch.nan_to_num(loss_edge)
                    + torch.nan_to_num(loss_real)
                    + torch.nan_to_num(loss_fake)
                    + torch.nan_to_num(loss_surf)
        )
        return loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, torch.nan_to_num(loss_surf)  # loss_notpos
        
    def query_batch(self, pos_query_list, batch, limit=10000):
        pos_query, batch_query = concat_tensors_to_batch(pos_query_list)
        num_query = pos_query.size(0)
        assert len(torch.unique(batch_query)) == 1, NotImplementedError('Modify get_batch_edge to support multiple batches')
        y_cls_all, y_ind_all = [], []
        for pos_query_partial, batch_query_partial in zip(split_tensor_to_segments(pos_query, limit), split_tensor_to_segments(batch_query, limit)):
            PM = batch_intersection_mask(batch.protein_element_batch, batch_query_partial)
            LM = batch_intersection_mask(batch.ligand_context_element_batch, batch_query_partial)
            ligand_context_bond_index, ligand_context_bond_type = get_batch_edge(
                batch.ligand_context_bond_index,
                batch.ligand_context_bond_type,
            )

            y_cls_partial, y_ind_partial, _ = self(
                # Query
                pos_query = pos_query_partial,
                batch_query = batch_query_partial,
                edge_index_query = [],
                # Protein
                protein_pos = batch.protein_pos[PM],
                protein_atom_feature = batch.protein_atom_feature.float()[PM],
                batch_protein = batch.protein_element_batch[PM],
                # Ligand
                ligand_pos = batch.ligand_context_pos[LM],
                ligand_atom_feature = batch.ligand_context_feature_full.float()[LM], 
                batch_ligand = batch.ligand_context_element_batch[LM],
                ligand_context_bond_index = ligand_context_bond_index,
                ligand_context_bond_type = ligand_context_bond_type,
            )
            y_cls_all.append(y_cls_partial)
            y_ind_all.append(y_ind_partial)
        
        y_cls_all = torch.cat(y_cls_all, dim=0)
        y_ind_all = torch.cat(y_ind_all, dim=0)

        lengths = [x.size(0) for x in pos_query_list]
        y_cls_list = split_tensor_by_lengths(y_cls_all, lengths)
        y_ind_list = split_tensor_by_lengths(y_ind_all, lengths)

        return y_cls_list, y_ind_list

    def query_position(self, pos_query, h_compose, compose_pos,
        idx_ligand, ligand_bond_index, ligand_bond_type):
        device = pos_query.device
        #NOTE: Only one parent batch at a time (i.e. batch size = 1)
        edge_index_query = torch.stack(torch.meshgrid(
                torch.arange(len(pos_query), dtype=torch.int64, device=device),
                torch.arange(len(idx_ligand), dtype=torch.int64, device=device),
                indexing=None
            ), dim=0).reshape(2, -1)
        query_compose_knn_edge_index = knn(x=compose_pos, y=pos_query, k=self.config.field.knn, num_workers=16)
        index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat = self.get_tri_edges(
            edge_index_query = edge_index_query,
            pos_query = pos_query,
            idx_ligand = idx_ligand,
            ligand_bond_index = ligand_bond_index,
            ligand_bond_type = ligand_bond_type
        )
        y_real_pred, edge_pred = self.field(
            pos_query = pos_query,
            edge_index_query = edge_index_query,
            pos_compose = compose_pos,
            node_attr_compose = h_compose,
            edge_index_q_cps_knn = query_compose_knn_edge_index,

            index_real_cps_edge_for_atten = index_real_cps_edge_for_atten,
            tri_edge_index = tri_edge_index,
            tri_edge_feat = tri_edge_feat
        )
        edge_pred = edge_pred.reshape(len(pos_query), len(idx_ligand), self.num_bond_types+1)
        return y_real_pred, edge_pred
        
    def get_tri_edges(self, edge_index_query, pos_query, idx_ligand, ligand_bond_index, ligand_bond_type):
        row, col = edge_index_query
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
        for node in torch.arange(pos_query.size(0)):
            num_edges = (row == node).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, ).to('cuda') + acc_num_edges
            index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing=None)
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_real_cps_edge_i_list.append(index_edge_i)
            index_real_cps_edge_j_list.append(index_edge_j)
            acc_num_edges += num_edges
        index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0)  # add len(real_compose_edge_index) in the dataloader for batch
        index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0)

        node_a_cps_tri_edge = col[index_real_cps_edge_i]  # the node of tirangle edge for the edge attention (in the compose)
        node_b_cps_tri_edge = col[index_real_cps_edge_j]
        n_context = len(idx_ligand)
        adj_mat = (torch.zeros([n_context, n_context], dtype=torch.long) - torch.eye(n_context, dtype=torch.long)).to('cuda')
        adj_mat[ligand_bond_index[0], ligand_bond_index[1]] = ligand_bond_type
        tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
        tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]]).to('cuda')).long()

        index_real_cps_edge_for_atten = torch.stack([
            index_real_cps_edge_i, index_real_cps_edge_j  # plus len(real_compose_edge_index_0) for dataloader batch
        ], dim=0)
        tri_edge_index = torch.stack([
            node_a_cps_tri_edge, node_b_cps_tri_edge  # plus len(compose_pos) for dataloader batch
        ], dim=0)
        return index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat
