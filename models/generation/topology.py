import torch
from torch.nn import Module, Sequential, LayerNorm
from torch_scatter import scatter_add, scatter_softmax, scatter_sum

from math import pi as PI

from ..model_utils import GaussianSmearing, EdgeExpansion
from ..invariant import GVLinear, GVPerceptronVN, MessageModule


# [NEW] RDKit相关导入
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import HybridizationType
# from rdkit.Chem import rdDecomposition # 已废弃
# [MODIFIED] RDKit相关导入
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import HybridizationType
# from rdkit.Chem import rdDecomposition  # 删除这行
from rdkit.Chem.Draw import IPythonConsole

# [NEW] 如果需要分解相关功能，可以使用以下替代方法
from rdkit.Chem import BRICS  # 用于分子分解
from rdkit.Chem import Recap  # 另一个分子分解工具
from rdkit.Chem.Draw import IPythonConsole

# [NEW] 用于并行处理的工具
from multiprocessing import Pool
from functools import partial

# [NEW] 用于类型提示的导入
from typing import Dict, List, Tuple, Optional, Union
import warnings

# [NEW] 忽略RDKit的警告
warnings.filterwarnings('ignore', category=UserWarning, module='rdkit')

# 引入了键长约束
# class SpatialClassifierVN(Module):
#     """空间向量分类器类，用于处理分子的3D结构和化学特征"""
    
#     def __init__(self, num_classes, num_bond_types, in_sca, in_vec, num_filters, edge_channels, num_heads, k=32, cutoff=10.0):
#         """
#         初始化函数
#         Args:
#             num_classes: 原子类型的数量
#             num_bond_types: 化学键类型的数量
#             in_sca: 输入标量特征的维度
#             in_vec: 输入向量特征的维度
#             num_filters: 神经网络中的过滤器数量，是一个包含两个元素的列表 [标量过滤器数, 向量过滤器数]
#             edge_channels: 边特征的通道数
#             num_heads: 注意力机制中的头数
#             k: KNN搜索中的邻居数量，默认32
#             cutoff: 距离截断值，默认10.0埃
#         """
#         super().__init__()
#         # 存储化学键类型的数量
#         self.num_bond_types = num_bond_types
        
#         # 初始化消息传递模块，用于在原子之间传递信息
#         self.message_module = MessageModule(
#             in_sca,              # 输入标量特征维度
#             in_vec,              # 输入向量特征维度
#             edge_channels,       # 边特征通道数
#             edge_channels,       # 边特征通道数
#             num_filters[0],      # 标量过滤器数量
#             num_filters[1],      # 向量过滤器数量
#             cutoff              # 距离截断值
#         )

#         # 边特征处理网络：处理原子对之间的边特征
#         self.nn_edge_ij = Sequential(
#             # 第一层：向量感知机，处理边的几何信息
#             GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
#             # 第二层：线性变换，进一步处理特征
#             GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
#         )
        
#         # 分类器网络：用于预测原子类型
#         self.classifier = Sequential(
#             # 第一层：向量感知机，处理节点特征
#             GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
#             # 第二层：线性变换，输出类别预测
#             GVLinear(num_filters[0], num_filters[1], num_classes, 1)
#         )

#         # 边特征转换网络：组合和转换边的特征
#         self.edge_feat = Sequential(
#             # 第一层：向量感知机，处理连接的节点和边特征
#             # 输入维度 = 2个节点的标量特征 + 原始标量特征
#             GVPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
#             # 第二层：线性变换
#             GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
#         )
        
#         # 边注意力层：关注重要的边特征
#         self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        
#         # 边类型预测层：预测化学键类型
#         self.edge_pred = GVLinear(num_filters[0], num_filters[1], num_bond_types + 1, 1)
        
#         # 距离编码器：将距离转换为高斯基函数表示
#         self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
#         # 3埃距离编码器：专门用于较短距离的编码
#         self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
#         # 向量编码器：处理3D向量特征
#         self.vector_expansion = EdgeExpansion(edge_channels)
        
#         # 存储KNN参数和截断距离
#         self.k = k
#         self.cutoff = cutoff

#     def forward(self, pos_query, edge_index_query, pos_compose, node_attr_compose, edge_index_q_cps_knn,
#                 index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[]):
#         """
#         前向传播函数
#         Args:
#             pos_query: 形状(N_query, 3)的张量，查询原子的3D坐标
#             edge_index_query: 形状(2, E)的张量，边的连接关系
#             pos_compose: 形状(N_compose, 3)的张量，环境原子的3D坐标
#             node_attr_compose: 环境原子的特征，包含标量和向量特征
#             edge_index_q_cps_knn: KNN搜索得到的边索引
#             index_real_cps_edge_for_atten: 用于注意力机制的真实边索引
#             tri_edge_index: 三角形边的索引
#             tri_edge_feat: 三角形边的特征
#         """

#         # 计算边的向量差和欧氏距离
#         vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
#         dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # 计算边长度
        
#         # 将距离和向量转换为特征表示
#         edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
#         # 通过消息传递模块传递信息
#         h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)

#         # 聚合每个节点的消息
#         y = [
#             # 聚合标量特征
#             scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)),
#             # 聚合向量特征
#             scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))
#         ]

#         # 预测原子类型
#         y_cls, _ = self.classifier(y)

#         # 如果存在边，则预测边类型
#         if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
#             # 获取边的起始节点索引和特征
#             idx_node_i = edge_index_query[0]
#             node_mol_i = [
#                 y[0][idx_node_i],  # 起始节点的标量特征
#                 y[1][idx_node_i]   # 起始节点的向量特征
#             ]
            
#             # 获取边的终止节点索引和特征
#             idx_node_j = edge_index_query[1]
#             node_mol_j = [
#                 node_attr_compose[0][idx_node_j],  # 终止节点的标量特征
#                 node_attr_compose[1][idx_node_j]   # 终止节点的向量特征
#             ]
            
#             # 计算边的几何特征
#             vec_ij = pos_query[idx_node_i] - pos_compose[idx_node_j]  # 边的向量
#             dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)    # 边的长度

#             # 添加统一的键长约束：1.0-2.0埃
#             bond_mask = (dist_ij >= 1.0) & (dist_ij <= 2.0)
            
#             if bond_mask.any():
#                 # 只为有效键长的边计算特征
#                 edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij)
#                 edge_feat = self.nn_edge_ij(edge_ij)
                
#                 edge_attr = (
#                     torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),
#                     torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1)
#                 )
                
#                 edge_attr = self.edge_feat(edge_attr)
#                 edge_attr = self.edge_atten(
#                     edge_attr,
#                     edge_index_query,
#                     pos_compose,
#                     index_real_cps_edge_for_atten,
#                     tri_edge_index,
#                     tri_edge_feat
#                 )
                
#                 # 预测边类型
#                 edge_pred, _ = self.edge_pred(edge_attr)
                
#                 # 对超出键长范围的边，将预测设为无键（索引0）
#                 no_bond_prediction = torch.zeros((dist_ij.size(0), self.num_bond_types + 1), 
#                                               device=dist_ij.device)
#                 no_bond_prediction[:, 0] = 1.0  # 将第一列（无键）设为1
                
#                 # 使用mask合并预测结果
#                 edge_pred = torch.where(bond_mask, edge_pred, no_bond_prediction)
#             else:
#                 # 如果所有边都超出键长范围，全部预测为无键
#                 edge_pred = torch.zeros((dist_ij.size(0), self.num_bond_types + 1), 
#                                       device=dist_ij.device)
#                 edge_pred[:, 0] = 1.0  # 将第一列（无键）设为1

#         else:
#             edge_pred = torch.empty([0, self.num_bond_types+1], device=pos_query.device)

#         return y_cls, edge_pred

# 再引入价键规则和不稳定化学结构

class SpatialClassifierVN(Module):
    """空间向量分类器类，用于处理分子的3D结构和化学特征"""

    def __init__(self, num_classes, num_bond_types, in_sca, in_vec, num_filters, edge_channels, num_heads, k=32, cutoff=10.0):
        super().__init__()
        self.num_bond_types = num_bond_types
        self.k = k
        self.cutoff = cutoff

        # 初始化 RDKit 分子对象
        self.molecule = Chem.RWMol()  # 分子对象作为类的属性
        self.atom_positions = []  # 用于存储原子坐标

        # 初始化模块（保持原样）
        self.message_module = MessageModule(
            in_sca, in_vec, edge_channels, edge_channels,
            num_filters[0], num_filters[1], cutoff
        )
        self.nn_edge_ij = Sequential(
            GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.classifier = Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_classes, 1)
        )
        self.edge_feat = Sequential(
            GVPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        self.edge_pred = GVLinear(num_filters[0], num_filters[1], num_bond_types + 1, 1)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)

    def forward(self, pos_query, edge_index_query, pos_compose, node_attr_compose, edge_index_q_cps_knn,
                index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[]):
        """
        前向传播函数
        """
        # 原始边特征和距离计算（保持原样）
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)
        y = [
            scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)),
            scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))
        ]
        y_cls, _ = self.classifier(y)

        # 更新分子对象：添加原子和它们的化学类型
        self._update_molecule(y_cls, pos_query)

        # 如果存在边，进行边的类型预测
        if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
            idx_node_i = edge_index_query[0]
            node_mol_i = [y[0][idx_node_i], y[1][idx_node_i]]
            idx_node_j = edge_index_query[1]
            node_mol_j = [node_attr_compose[0][idx_node_j], node_attr_compose[1][idx_node_j]]
            vec_ij = pos_query[idx_node_i] - pos_compose[idx_node_j]
            dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)

            bond_mask = (dist_ij >= 1.0) & (dist_ij <= 2.0)

            if bond_mask.any():
                edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij)
                edge_feat = self.nn_edge_ij(edge_ij)
                edge_attr = (
                    torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),
                    torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1)
                )
                edge_attr = self.edge_feat(edge_attr)
                edge_attr = self.edge_atten(
                    edge_attr,
                    edge_index_query,
                    pos_compose,
                    index_real_cps_edge_for_atten,
                    tri_edge_index,
                    tri_edge_feat
                )
                edge_pred, _ = self.edge_pred(edge_attr)

                # 检查价键规则并重新采样
                edge_pred = self._check_and_resample(edge_pred, edge_index_query, dist_ij, bond_mask)
                
            else:
                edge_pred = torch.zeros((dist_ij.size(0), self.num_bond_types + 1), device=dist_ij.device)
                edge_pred[:, 0] = 1.0

        else:
            edge_pred = torch.empty([0, self.num_bond_types+1], device=pos_query.device)

        return y_cls, edge_pred


    def check_alert_structures(self, mol, alert_smarts_list):
        """
        检查分子是否包含指定的不稳定化学结构。

        Args:
            mol (rdkit.Chem.Mol): RDKit 分子对象。
            alert_smarts_list (list): 包含 SMARTS 表达式的列表，用于定义不合法的化学结构。

        Returns:
            bool: 如果分子包含任何不合法的结构，则返回 True，否则返回 False。
            list: 匹配的 SMARTS 规则（如果存在）。
        """
        if mol is None:
            raise ValueError("Invalid molecule object: `mol` is None.")

        matched_alerts = []
        for smarts in alert_smarts_list:
            try:
                # 将 SMARTS 转换为 RDKit 的模式对象
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    print(f"Warning: Invalid SMARTS pattern: {smarts}")
                    continue
                
                # 检查分子是否包含匹配的子结构
                if mol.HasSubstructMatch(pattern):
                    matched_alerts.append(smarts)
            except Exception as e:
                print(f"Error processing SMARTS pattern '{smarts}': {e}")

        # 如果有任何匹配的 SMARTS，返回 True 和匹配列表
        return len(matched_alerts) > 0, matched_alerts
    def _update_molecule(self, y_cls, pos_query):
        """
        根据预测的化学类型和坐标更新分子对象。
        """
        # 将预测的化学类型转化为原子类型
        atom_types = y_cls.argmax(dim=-1).cpu().numpy()  # 假设 y_cls 是化学类型的 logits
        atomic_numbers = [self._get_atomic_number(atom_type) for atom_type in atom_types]

        # 清空分子对象，重新添加原子
        self.molecule = Chem.RWMol()  # 清空分子
        self.atom_positions = []  # 清空坐标列表

        for i, (atomic_num, pos) in enumerate(zip(atomic_numbers, pos_query.cpu().numpy())):
            atom = Chem.Atom(atomic_num)
            self.molecule.AddAtom(atom)  # 添加原子到分子中
            self.atom_positions.append(pos)  # 保存原子坐标

    def check_valency(self):
        """
        检查类中分子对象的价键规则。
        """
        try:
            Chem.SanitizeMol(self.molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True
        except ValueError:
            return False

    # def _check_and_resample(self, edge_pred, edge_index_query, dist_ij, bond_mask):
        """
        检查分子价键规则，如果不满足则移除无效的键。
        """
        # 遍历所有边的预测
        for edge_idx, pred in enumerate(edge_pred):
            if pred.argmax() > 0:  # 如果预测为有键
                i, j = edge_index_query[:, edge_idx].tolist()

                # 检查索引是否越界
                if i >= self.molecule.GetNumAtoms() or j >= self.molecule.GetNumAtoms():
                    # print(f"Skipping invalid bond: i={i}, j={j}, num_atoms={self.molecule.GetNumAtoms()}")
                    continue

                # 如果是自健（自己指向自己）或者重复键，则跳过
                if i == j or self.molecule.GetBondBetweenAtoms(i, j) is not None:
                    continue
                bond_type = self._get_bond_type(pred.argmax())

                
                # 添加键
                self.molecule.AddBond(i, j, bond_type)

                # 检查价键规则
                if not self.check_valency():
                    # 如果价键规则不满足，移除刚刚添加的键
                    # print(f"Valency check failed. Removing bond: i={i}, j={j}, bond_type={bond_type}")
                    self.molecule.RemoveBond(i, j)

        # 如果价键检查失败，则默认返回无键预测
        if not self.check_valency():
            print("Final valency check failed. Returning default predictions.")
            no_bond_prediction = torch.zeros((dist_ij.size(0), self.num_bond_types + 1), device=dist_ij.device)
            no_bond_prediction[:, 0] = 1.0
            return no_bond_prediction

        return edge_pred
    def _check_and_resample(self, edge_pred, edge_index_query, dist_ij, bond_mask):
        """
        检查分子价键规则和不稳定化学键，如果不满足则移除无效的键。
        """
        # 定义不合法的化学结构（SMARTS 规则）
        alert_smarts_list = [
            '[O]-[O]',                    # 过氧键
            '[N]-[O,Br,Cl,I,F,P]',        # 氮与氧或卤素键
            '[S,P]-[Br,Cl,I,F]',          # 硫或磷与卤素键
            '[P]-[O]-[P]',                # 磷氧磷键
            '[Br,Cl,I,F]-[Br,Cl,I,F]',    # 卤素之间的直接键
        ]

        for edge_idx, pred in enumerate(edge_pred):
            if pred.argmax() > 0:  # 如果预测为有键
                i, j = edge_index_query[:, edge_idx].tolist()

                # 检查索引是否越界
                if i >= self.molecule.GetNumAtoms() or j >= self.molecule.GetNumAtoms():
                    # print(f"Skipping invalid bond: i={i}, j={j}, num_atoms={self.molecule.GetNumAtoms()}")
                    continue

                # 如果是自健（自己指向自己）或者重复键，则跳过
                if i == j or self.molecule.GetBondBetweenAtoms(i, j) is not None:
                    continue

                # 获取预测的键类型
                bond_type = self._get_bond_type(pred.argmax())

                # 添加键
                self.molecule.AddBond(i, j, bond_type)

                # 检查价键规则
                if not self.check_valency():
                    # 如果价键规则不满足，移除刚刚添加的键
                    # print(f"Valency check failed. Removing bond: i={i}, j={j}, bond_type={bond_type}")
                    self.molecule.RemoveBond(i, j)
                    continue

                # 检查不稳定化学键
                has_alert, matched_alerts = self.check_alert_structures(self.molecule, alert_smarts_list)
                if has_alert:
                    # 如果检测到不稳定化学键，移除刚刚添加的键
                    print(f"Unstable bond detected between atoms {i} and {j}: {matched_alerts}. Removing bond.")
                    self.molecule.RemoveBond(i, j)

        # 如果价键检查失败，则默认返回无键预测
        if not self.check_valency():
            print("Final valency check failed. Returning default predictions.")
            no_bond_prediction = torch.zeros((dist_ij.size(0), self.num_bond_types + 1), device=dist_ij.device)
            no_bond_prediction[:, 0] = 1.0
            return no_bond_prediction

        return edge_pred
    def _get_atomic_number(self, atom_type):
        """
        根据化学类型索引返回元素的原子序数。
        """
        atomic_number_map = {
            0: 1,  # 氢
            1: 6,  # 碳
            2: 7,  # 氮
            3: 8,  # 氧
            # 根据需要扩展其他元素
        }
        return atomic_number_map.get(atom_type, 1)  # 默认返回氢的原子序数

    def _get_bond_type(self, bond_idx):
        """
        根据边类型索引获取 RDKit 中的化学键类型。
        """
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]
        return bond_types[bond_idx - 1]  # 跳过无键的索引

# 错误版本
# class SpatialClassifierVN(Module):

#     def __init__(self, num_classes, num_bond_types, in_sca, in_vec, num_filters, edge_channels, num_heads, k=32, cutoff=10.0):
#         super().__init__()
#         self.num_bond_types = num_bond_types
#         self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)

#         self.nn_edge_ij = Sequential(
#             GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
#             GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
#         )
        
#         self.classifier = Sequential(
#             GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
#             GVLinear(num_filters[0], num_filters[1], num_classes, 1)
#         )

#         self.edge_feat = Sequential(
#             GVPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
#             GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
#         )
#         self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
#         self.edge_pred = GVLinear(num_filters[0], num_filters[1], num_bond_types + 1, 1)
        
#         self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
#         self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
#         self.vector_expansion = EdgeExpansion(edge_channels)
#         self.k = k
#         self.cutoff = cutoff

#         # [NEW] 化学约束相关参数
#         self.bond_length_range = (0.8, 2.0)  # 键长范围约束
#         self.resample_edge_failed = False
#         self.alert_patterns = [  # 不合理结构模式
#             '[O]-[O]',
#             '[N]-[O,Br,Cl,I,F,P]',
#             '[S,P]-[Br,Cl,I,F]',
#             '[P]-[O]-[P]',
#             '[Br,Cl,I,F]-[Br,Cl,I,F]'
#         ]
#         self.max_resample_attempts = 50  # 最大重采样次数

        

#     # [NEW] 添加价键规则的实现函数
#     def check_valency(mol):
#         """检查分子是否满足价键规则
        
#         Args:
#             mol: RDKit mol对象
            
#         Returns:
#             bool: 满足价键规则返回True，否则返回False
#         """
#         try:
#             Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
#             return True
#         except ValueError:
#             return False

#     # [NEW] 添加不合理化学结构检查方法
#     def check_structure_alerts(self, mol):
#         """检查分子是否包含不合理的结构模式"""
#         if mol is None:
#             return True
        
#         try:
#             smiles = Chem.MolToSmiles(mol)
#             for pattern in self.alert_patterns:
#                 substructure = Chem.MolFromSmarts(pattern)
#                 if mol.HasSubstructMatch(substructure):
#                     return True
#             return False
#         except:
#             return True

#     # [NEW]
#     def check_bond_length(self, pos1, pos2):
#         """检查键长是否在合理范围内"""
#         dist = torch.norm(pos1 - pos2, p=2)
#         return self.bond_length_range[0] <= dist <= self.bond_length_range[1]

#     # [NEW]
#     def process_new_bonds(self, rw_mol, atom_idx, new_edge_idx, new_bond_type_to_add, new_pos):
#         """处理新生成的化学键，包含所有化学约束检查"""
#         resample_edge = 0
        
#         while resample_edge < self.max_resample_attempts:
#             # 检查键长约束
#             all_bonds_valid = True
#             for ix in range(new_edge_idx.size(1)):
#                 i, j = new_edge_idx[:, ix]
#                 if not self.check_bond_length(new_pos[i], new_pos[j]):
#                     all_bonds_valid = False
#                     break
            
#             if not all_bonds_valid:
#                 resample_edge += 1
#                 continue

#             # 添加新的化学键
#             for ix in range(new_edge_idx.size(1)):
#                 i, j = new_edge_idx[:, ix].tolist()
#                 bond_type = new_bond_type_to_add[ix].item()
#                 rw_mol.AddBond(atom_idx, j, Chem.BondType.values[bond_type])

#             # 检查价键规则
#             if not self.check_valency(rw_mol):
#                 for ix in range(new_edge_idx.size(1)):
#                     i, j = new_edge_idx[:, ix].tolist()
#                     rw_mol.RemoveBond(atom_idx, j)
#                 resample_edge += 1
#                 continue

#             # 检查不合理结构模式
#             if self.check_structure_alerts(rw_mol):
#                 for ix in range(new_edge_idx.size(1)):
#                     i, j = new_edge_idx[:, ix].tolist()
#                     rw_mol.RemoveBond(atom_idx, j)
#                 resample_edge += 1
#                 continue

#             return True

#         self.resample_edge_failed = True
#         return False

#     def forward(self, pos_query, edge_index_query, pos_compose, node_attr_compose, edge_index_q_cps_knn,
#                          index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[]):
#         vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
#         dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)
#         edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
#         h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)

#         y = [scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)),
#              scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))]

#         y_cls, _ = self.classifier(y)

#         if len(edge_index_q_cps_knn) != 0:
#             idx_knn_i = edge_index_q_cps_knn[0]
#             idx_knn_j = edge_index_q_cps_knn[1]
#             vec_knn = pos_query[idx_knn_i] - pos_compose[idx_knn_j]
#             dist_knn = torch.norm(vec_knn, p=2, dim=-1).view(-1, 1)

#             edge_knn = self.distance_expansion_kNN(dist_knn), self.vector_expansion(vec_knn)
#             edge_feat_knn = self.nn_edge_knn(edge_knn)
#             node_query = self.prepare_node(edge_feat_knn,
#                                          edge_index_q_cps_knn,
#                                          pos_query.size(0))

#         # bond classification
#         if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
#             idx_node_i = edge_index_query[0]
#             idx_node_j = edge_index_query[1]
#             vec_ij = pos_query[idx_node_i] - pos_compose[idx_node_j]
#             dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)

#             # [NEW] 使用bond_length_range进行约束
#             valid_bonds = (dist_ij >= self.bond_length_range[0]) & (dist_ij <= self.bond_length_range[1])

#             edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij)
#             edge_feat = self.nn_edge_ij(edge_ij)

#             node_mol_i = node_query
#             node_mol_j = (node_attr_compose[0][idx_node_j], node_attr_compose[1][idx_node_j])

#             edge_attr = (torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),
#                         torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1))
            
#             edge_attr = self.edge_feat(edge_attr)
#             edge_attr = self.edge_atten(edge_attr, edge_index_query, pos_compose,
#                                       index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat)
            
#             edge_pred, _ = self.edge_pred(edge_attr)
#             # [NEW] 应用键长约束
#             edge_pred[~valid_bonds.squeeze()] = torch.tensor([1.0] + [0.0] * self.num_bond_types,
#                                                            device=edge_pred.device)
#         else:
#             edge_pred = torch.empty([0, self.num_bond_types+1], device=pos_query.device)

#         return y_cls, edge_pred

#     def prepare_node(self, edge_feat_knn, edge_index_knn, num_nodes):
#         """Node feature preparation"""
#         idx_i = edge_index_knn[0]
#         node_sca = scatter_mean(edge_feat_knn[0], idx_i, dim=0, dim_size=num_nodes)
#         node_vec = scatter_mean(edge_feat_knn[1], idx_i, dim=0, dim_size=num_nodes)
#         return node_sca, node_vec

class AttentionEdges(Module):

    def __init__(self, hidden_channels, key_channels, num_heads=1, num_bond_types=3):
        super().__init__()
        
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)
        """使用assert语句确保hidden_channels和key_channels的每个元素都能被num_heads整除。
        这可能是因为在后续的计算中，hidden_channels和key_channels的每个元素都需要被num_heads平均分配。"""

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        # 将传入的参数hidden_channels、key_channels和num_heads保存为类的属性。

        # linear transformation for attention 
        self.q_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.k_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.v_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])
        """创建三个GVLinear实例，分别命名为self.q_lin、self.k_lin和self.v_lin。
        GVLinear可能是一个用于处理标量和向量输入的模块，通过特殊的线性变换将标量和向量输入转换为标量和向量输出。
        这三个实例可能分别用于计算注意力机制中的查询（Q）、键（K）和值（V）。"""

        self.atten_bias_lin = AttentionBias(self.num_heads, hidden_channels, num_bond_types=num_bond_types) 
        # 创建一个AttentionBias实例，命名为self.atten_bias_lin。AttentionBias是一个用于计算注意力偏置的模块。
        
        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])
        # 创建两个LayerNorm实例，分别命名为self.layernorm_sca和self.layernorm_vec。LayerNorm是一个用于规范化的层归一化模块。
        

    def forward(self, edge_attr, edge_index, pos_compose, 
                          index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat,):
        """方法接收了多个参数，包括edge_attr、edge_index、pos_compose、index_real_cps_edge_for_atten、tri_edge_index和tri_edge_feat。
        其中，edge_attr可能是边的特征，edge_index可能是边的索引，pos_compose可能是节点的位置，
        index_real_cps_edge_for_atten可能是用于注意力机制的边的索引，tri_edge_index可能是三角形边的索引，tri_edge_feat可能是三角形边的特征。"""
        """
        Args:
            x:  edge features: scalar features (N, feat), vector features(N, feat, 3)
            edge_attr:  (E, H)
            edge_index: (2, E). the row can be seen as batch_edge
        """
        scalar, vector = edge_attr
        N = scalar.size(0)
        row, col = edge_index   # (N,) 

        # Project to multiple key, query and value spaces
        h_queries = self.q_lin(edge_attr)
        h_queries = (h_queries[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_queries[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_keys = self.k_lin(edge_attr)
        h_keys = (h_keys[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_keys[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_values = self.v_lin(edge_attr)
        h_values = (h_values[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_values[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)

        # assert (index_edge_i_list == index_real_cps_edge_for_atten[0]).all()
        # assert (index_edge_j_list == index_real_cps_edge_for_atten[1]).all()
        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        # # get nodes of triangle edges

        atten_bias = self.atten_bias_lin(
            tri_edge_index,
            tri_edge_feat,
            pos_compose,
        )


        # query * key
        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),  # (N', heads)
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1)  # (N', heads)
        ]

        alpha = [
            atten_bias[0] + qk_ij[0],
            atten_bias[1] + qk_ij[1]
        ]
        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),  # (N', heads)
            scatter_softmax(alpha[1], index_edge_i_list, dim=0)  # (N', heads)
        ] 

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output =[
            scatter_sum((alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1), index_edge_i_list, dim=0, dim_size=N),   # (N, H, 3)
            scatter_sum((alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3), index_edge_i_list, dim=0, dim_size=N)   # (N, H, 3)
        ]

        # output 
        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        output = [self.layernorm_sca(output[0]), self.layernorm_vec(output[1])]

        return output



class AttentionBias(Module):

    def __init__(self, num_heads, hidden_channels, cutoff=10., num_bond_types=3): #TODO: change the cutoff
        super().__init__()
        num_edge_types = num_bond_types + 1
        self.num_bond_types = num_bond_types
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels[0] - num_edge_types-1)  # minus 1 for self edges (e.g. edge 0-0)
        self.vector_expansion = EdgeExpansion(hidden_channels[1])  # Linear(in_features=1, out_features=hidden_channels[1], bias=False)
        self.gvlinear = GVLinear(hidden_channels[0], hidden_channels[1], num_heads, num_heads)
    """首先，通过super().__init__()调用父类的初始化方法，
    然后，将传入的参数num_heads、hidden_channels、cutoff和num_bond_types保存为类的属性。
    其中，num_heads可能是注意力机制中的头的数量，hidden_channels可能是隐藏层的通道数量，cutoff可能是高斯模糊的截止距离，num_bond_types可能是边的类型数量。
    接着，创建一个GaussianSmearing实例，命名为self.distance_expansion。
    GaussianSmearing可能是一个用于处理距离特征的模块，通过高斯模糊将距离特征转换为更适合神经网络处理的形式。
    然后，创建一个EdgeExpansion实例，命名为self.vector_expansion。
    EdgeExpansion可能是一个用于处理边向量的模块，通过线性变换将边向量扩展到更高的维度。
    最后，创建一个GVLinear实例，命名为self.gvlinear。
    GVLinear可能是一个用于处理标量和向量输入的模块，通过特殊的线性变换将标量和向量输入转换为标量和向量输出。"""

    def forward(self,  tri_edge_index, tri_edge_feat, pos_compose):
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1) # type: ignore
        
        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([
            dist_feat,
            tri_edge_feat,
        ], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gvlinear([sca_feat, vec_feat])
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec
    """首先，从tri_edge_index中获取两个节点的索引node_a和node_b，然后从pos_compose中获取这两个节点的位置pos_a和pos_b。
    接着，计算两个节点位置的差值vector，并计算其2范数dist，得到两个节点的距离。
    然后，将距离dist通过self.distance_expansion进行处理，得到dist_feat。
    这可能是一个用于处理距离特征的模块，通过高斯模糊将距离特征转换为更适合神经网络处理的形式。
    接着，将dist_feat和tri_edge_feat进行拼接，得到sca_feat。这是一个包含了距离特征和边特征的向量。
    然后，将位置差值vector通过self.vector_expansion进行处理，得到vec_feat。
    这可能是一个用于处理边向量的模块，通过线性变换将边向量扩展到更高的维度。
    接着，将sca_feat和vec_feat通过self.gvlinear进行处理，得到output_sca和output_vec。
    这可能是一个用于处理标量和向量输入的模块，通过特殊的线性变换将标量和向量输入转换为标量和向量输出。
    最后，计算output_vec的平方和，得到新的output_vec。"""
