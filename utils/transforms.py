import copy
# from multiprocessing import context
import os
import sys
sys.path.append('.')
import random
import time
import uuid
from itertools import compress
# from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.nn import knn, radius
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from rdkit import Chem
from rdkit.Chem import rdMMPA
from scipy.spatial import distance_matrix
# import multiprocessing as multi
# from torch_geometric.data import DataLoader
import os.path as osp
from torch_geometric.utils import to_undirected
try:
    from .data import ProteinLigandData
    from .datasets import *
    from .misc import *
    from .train import inf_iterator
    from .protein_ligand import ATOM_FAMILIES
    from .chem import remove_dummys_mol, check_linkers, Murcko_decompose
    from .surface import geodesic_matrix, dst2knnedge, read_ply_geom
except:
    from utils.data import ProteinLigandData
    from utils.datasets import *
    from utils.misc import *
    from utils.train import inf_iterator
    from utils.protein_ligand import ATOM_FAMILIES
    from utils.chem import remove_dummys_mol, check_linkers, Murcko_decompose
    from utils.surface import geodesic_matrix, dst2knnedge, read_ply_geom
import argparse
import logging

class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_feature = data.protein_feature
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        """提取了 data 对象中的 protein_feature 和 ligand_element 属性。然后，通过检查 ligand_element 是否等于 1 来确定哪些配体元素是氢原子。"""
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            """创建一个 not_H_ligand 的布尔数组，该数组的每个元素表示对应的配体元素是否不是氢原子。
            然后，使用这个 not_H_ligand 数组来更新 data 对象中的 ligand_atom_feature、ligand_element 和 ligand_pos 属性，从而删除所有的氢原子。"""
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand)) # type: ignore
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i:[index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh in enumerate(new_nbh_list)}
            """接下来，代码将创建一个 index_atom_H 的数组，该数组包含所有氢原子的索引。然后，代码将创建一个 index_changer 的数组，该数组的每个元素表示对应的配体元素在删除所有氢原子后的新索引。
            然后，代码将更新 data 对象中的 ligand_nbh_list 属性，该属性可能表示每个配体元素的邻居列表。代码将删除所有包含氢原子的邻居，并将剩余邻居的索引更新为新的索引。"""
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]
            """更新 data 对象中的 ligand_bond_index 和 ligand_bond_type 属性，这两个属性可能表示配体元素之间的键的索引和类型。
            代码将删除所有包含氢原子的键，并将剩余键的索引更新为新的索引。"""

        return data


class FeaturizeProteinAtom(object):

    def __init__(self):  
        super().__init__()  
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se 
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20
    """
    在 __init__ 方法中，首先调用了父类的初始化方法 super().__init__()。
    然后，定义了一个名为 self.atomic_numbers 的属性，这是一个长整型的张量，
    包含了蛋白质原子的原子序数（即元素在周期表中的序号），这里包括了碳（C，6）、氮（N，7）、氧（O，8）、硫（S，16）和硒（Se，34）。
    此外，还定义了一个名为 self.max_num_aa 的属性，其值为20，可能表示蛋白质中氨基酸的最大数量。
    """    
        

    @property
    def feature_dim(self): 
        return 5
        #return self.atomic_numbers.size(0) + self.max_num_aa + 1 + 1
    """
    feature_dim 是一个属性方法，返回特征的维度，这里直接返回了5。
    在注释掉的代码中，特征的维度是由原子序数的数量、最大氨基酸数量以及其他两个1（可能表示其他两种特征）的和来计算的。
    """

    def __call__(self, data:ProteinLigandData):
        
        feature = data.protein_feature
        is_mol_atom = torch.zeros(feature.shape[0], dtype=torch.long).unsqueeze(-1)
        # x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        x = torch.cat([feature, is_mol_atom], dim=-1)
        data.protein_surf_feature = x
        # data.compose_index = torch.arange(len(element), dtype=torch.long)
        return data
    """__call__ 方法定义了类的调用行为。
    当一个 FeaturizeProteinAtom 对象被当作函数调用时，这个方法就会被执行。
    这个方法接受一个 ProteinLigandData 对象作为输入，
    然后将蛋白质的特征和一个全零的长整型张量（表示是否为分子原子）拼接在一起，
    然后将结果赋值给 data.protein_surf_feature。
    最后，返回处理后的 data 对象。"""


class FeaturizeLigandAtom(object):
    
    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6,7,8,9,15,16,17])  # C N O F P S Cl
        assert len(self.atomic_numbers) == 7, NotImplementedError('fix the staticmethod: chagne_bond')

    # @property
    # def num_properties(self):
        # return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + (1 + 1 + 1) + 3

    def __call__(self, data:ProteinLigandData):
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # chem_feature = data.ligand_atom_feature
        is_mol_atom = torch.ones([len(element), 1], dtype=torch.long)
        n_neigh = data.ligand_num_neighbors.view(-1, 1)
        n_valence = data.ligand_atom_valence.view(-1, 1)
        ligand_atom_num_bonds = data.ligand_atom_num_bonds
        # x = torch.cat([element, chem_feature, ], dim=-1)
        x = torch.cat([element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1)
        data.ligand_atom_feature_full = x
        return data

    @staticmethod
    def change_features_of_neigh(ligand_feature_full, new_num_neigh, new_num_valence, ligand_atom_num_bonds):
        idx_n_neigh = 7 + 1
        idx_n_valence = idx_n_neigh + 1
        idx_n_bonds = idx_n_valence + 1
        ligand_feature_full[:, idx_n_neigh] = new_num_neigh.long()
        ligand_feature_full[:, idx_n_valence] = new_num_valence.long()
        ligand_feature_full[:, idx_n_bonds:idx_n_bonds+3] = ligand_atom_num_bonds.long()
        return ligand_feature_full



class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data:ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1 , num_classes=3)    # (1,2,3) to (0,1,2)-onehot
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()
    """在 LigandCountNeighbors 类中，定义了一个静态方法 count_neighbors，该方法接受四个参数：
    edge_index、symmetry、valence 和 num_nodes。edge_index 可能表示配体元素之间的键的索引，
    symmetry 表示这些键是否是对称的，valence 表示每个键的价数，num_nodes 表示配体元素的数量。
    这个方法首先检查 symmetry 是否为 True，然后计算每个配体元素的邻居数量，并返回一个包含这些数量的张量。"""

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = self.count_neighbors(
            data.ligand_bond_index, 
            symmetry=True, 
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_num_bonds = torch.stack([
            self.count_neighbors(
                data.ligand_bond_index, 
                symmetry=True, 
                valence=(data.ligand_bond_type == i).long(),
                num_nodes=data.ligand_element.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        """在 LigandCountNeighbors 类的 __call__ 方法中，首先计算了每个配体元素的邻居数量，
        并将这些数量存储在 data.ligand_num_neighbors 属性中。
        然后，计算了每个配体元素的价数，并将这些价数存储在 data.ligand_atom_valence 属性中。
        最后，对于价数为 1、2 和 3 的配体元素，分别计算了它们的邻居数量，并将这些数量存储在 data.ligand_atom_num_bonds 属性中。"""
        return data


class Geodesic_builder(object):

    def  __init__(self, knn=16):
        super().__init__()
        self.knn = knn  # knn of compose atoms knn 参数被设置为默认值 16，这个参数表示构建 k-近邻图时每个节点的近邻数量。
    
    def __call__(self, data:ProteinLigandData):
        """接受一个 ProteinLigandData 对象作为输入，然后根据这个对象的蛋白质位置和面信息，构建出 Delaunay 边缘索引。
        然后，它使用这个索引和节点数量来构建一个无向图。
        接下来，它将预先计算的地理 k-近邻边缘索引和 Delaunay 边缘索引传递给 gds_edge_process 方法，以生成地理边缘标量特征。
        最后，这个特征被赋值给输入数据对象的 gds_edge_sca 属性。"""
        # construct the Delanay edge 
        surf_pos = data.protein_pos
        num_nodes = surf_pos.shape[0]
        surf_face = data.face
        edge_index = torch.cat([surf_face[:2], surf_face[1:], surf_face[::2]], dim=1)
        dlny_edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # conrtruct the geodesic distance matrix corresponding to the Delanay edge 
        # gds_mat = geodesic_matrix(surf_pos, dlny_edge_index)

        # construct the knn_edge_index according to geodesic distance
        # gds_knn_edge_index and gds_knn_edge_dist have been pre-computed
        # gds_knn_edge_index, gds_knn_edge_dist = dst2knnedge(gds_mat, num_knn=self.knn)
        
        # assign the scalar feature to the geodesic knn edge 
        gds_knn_edge_index = data.gds_knn_edge_index
        gds_edge_sca = self.gds_edge_process(dlny_edge_index, gds_knn_edge_index, num_nodes=num_nodes)

        data.gds_edge_sca = gds_edge_sca 
        # data.gds_knn_edge_index = gds_knn_edge_index
        # data.gds_dist = gds_knn_edge_dist

        # specify the number of nodes in the graph
        # data.num_nodes = data.compose_feature.shape[0]

        return data

    @staticmethod
    def gds_edge_process(tri_edge_index,gds_knn_edge_index,num_nodes):
        """gds_edge_process 是一个静态方法，它接受 Delaunay 边缘索引、地理 k-近邻边缘索引和节点数量作为输入。
        这个方法首先计算出每个边缘的唯一标识符，然后找出地理 k-近邻边缘索引中与 Delaunay 边缘索引匹配的边缘。
        然后，它创建一个全零的张量，然后将匹配边缘的位置设置为 1。最后，它使用 one-hot 编码对这个张量进行编码，然后返回结果。"""
        id_tri_edge = tri_edge_index[0] * num_nodes + tri_edge_index[1]
        id_gds_knn_edge = gds_knn_edge_index[0] * num_nodes + gds_knn_edge_index[1]
        idx_edge = [torch.nonzero(id_gds_knn_edge == id_) for id_ in id_tri_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        compose_gds_edge_type = torch.zeros(len(gds_knn_edge_index[0]), dtype=torch.long) 
        compose_gds_edge_type[idx_edge[idx_edge>=0]] = torch.ones_like(idx_edge[idx_edge>=0])
        gds_edge_sca = F.one_hot(compose_gds_edge_type)

        return gds_edge_sca


class AtomComposer(object):

    def  __init__(self, protein_dim, ligand_dim, knn):
        super().__init__() # super().__init__() 是对父类的初始化函数的调用。由于 AtomComposer 类没有明确的父类，所以这个调用实际上是对 Python 内置的 object 类的初始化函数的调用。
        self.protein_dim = protein_dim  
        self.ligand_dim = ligand_dim
        self.knn = knn  # knn of compose atoms
        """self.protein_dim、self.ligand_dim 和 self.knn 是 AtomComposer 类的实例变量。
        self.protein_dim 和 self.ligand_dim 分别表示蛋白质和配体的维度，self.knn 表示用于组合原子的 k-近邻算法的 k 值。
        这些变量在类的其他方法中可能会被使用，例如在计算原子之间的距离或者构建 k-近邻图时。"""
    
    def __call__(self, data:ProteinLigandData):
        # fetch ligand context and protein from data
        ligand_context_pos = data.ligand_context_pos
        ligand_context_feature_full = data.ligand_context_feature_full
        protein_pos = data.protein_pos
        protein_surf_feature = data.protein_surf_feature
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)
        """首先，从 data 中获取配体环境的位置（ligand_context_pos）、配体环境的全特征（ligand_context_feature_full）、
        蛋白质的位置（protein_pos）和蛋白质表面的特征（protein_surf_feature）。然后计算配体环境和蛋白质的长度。"""

        # compose ligand context and protein. save idx of them in compose
        data.compose_pos = torch.cat([ligand_context_pos, protein_pos], dim=0) # 将配体环境和蛋白质的位置进行拼接，并保存在 data.compose_pos 中。然后计算拼接后的长度 len_compose。
        len_compose = len_ligand_ctx + len_protein
        protein_surf_feature_full_expand = torch.cat([
            protein_surf_feature, torch.zeros([len_protein,self.ligand_dim- self.protein_dim], dtype=torch.long)
        ], dim=1) #将蛋白质表面的特征和一个全零的 tensor 进行拼接，得到 protein_surf_feature_full_expand。这个全零的 tensor 的形状是 [len_protein,self.ligand_dim- self.protein_dim]，数据类型是 torch.long。
        # ligand_context_feature_full_expand = torch.cat([
        #     ligand_context_feature_full, torch.zeros([len_ligand_ctx, self.protein_dim - self.ligand_dim], dtype=torch.long)
        # ], dim=1)
        # data.compose_feature = torch.cat([ligand_context_feature_full_expand, protein_surf_feature], dim=0)
        data.compose_feature = torch.cat([ligand_context_feature_full, protein_surf_feature_full_expand],dim=0)
        data.idx_ligand_ctx_in_compose = torch.arange(len_ligand_ctx, dtype=torch.long)  # can be delete
        data.idx_protein_in_compose = torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx  # can be delete
        #计算配体环境在拼接后数据中的索引 data.idx_ligand_ctx_in_compose 和蛋白质在拼接后数据中的索引 data.idx_protein_in_compose。
        # gds change

        # build knn graph and bond type
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        """接下来，调用 self.get_knn_graph 方法，构建 k-近邻图并确定键类型。
        这个方法接受 data、self.knn、len_ligand_ctx、len_compose 和 num_workers 作为参数，并返回处理后的 data。"""
        return data
    

    @staticmethod
    def get_knn_graph(data:ProteinLigandData, knn, len_ligand_ctx, len_compose, num_workers=1, ):
        data.compose_knn_edge_index = knn_graph(data.compose_pos, knn, flow='target_to_source', num_workers=num_workers)
        """使用 knn_graph 函数计算 k-近邻图，并将结果保存在 data.compose_knn_edge_index 中。
        knn_graph 函数接受 data.compose_pos、knn、flow='target_to_source' 和 num_workers 作为参数。"""

        id_compose_edge = data.compose_knn_edge_index[0, :len_ligand_ctx*knn] * len_compose + data.compose_knn_edge_index[1, :len_ligand_ctx*knn]
        id_ligand_ctx_edge = data.ligand_context_bond_index[0] * len_compose + data.ligand_context_bond_index[1]
        """计算 id_compose_edge 和 id_ligand_ctx_edge。这两个值是通过将边的两个端点的索引进行特定的组合得到的，可以用来唯一标识每一条边。"""
        
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        """找出 id_compose_edge 中与 id_ligand_ctx_edge 相等的元素的索引，并保存在 idx_edge 中。这个过程是通过列表推导式和 torch.nonzero 函数实现的。"""
        
        data.compose_knn_edge_type = torch.zeros(len(data.compose_knn_edge_index[0]), dtype=torch.long)  # for encoder edge embedding
        data.compose_knn_edge_type[idx_edge[idx_edge>=0]] = data.ligand_context_bond_type[idx_edge>=0]
        """创建一个全零的 tensor data.compose_knn_edge_type，用于存储边的类型。然后，将 data.ligand_context_bond_type 中对应 idx_edge 的值赋给 data.compose_knn_edge_type。"""
        
        data.compose_knn_edge_feature = torch.cat([
            torch.ones([len(data.compose_knn_edge_index[0]), 1], dtype=torch.long),
            torch.zeros([len(data.compose_knn_edge_index[0]), 3], dtype=torch.long),
        ], dim=-1) 
        data.compose_knn_edge_feature[idx_edge[idx_edge>=0]] = F.one_hot(data.ligand_context_bond_type[idx_edge>=0], num_classes=4)    # 0 (1,2,3)-onehot
        """创建一个 tensor data.compose_knn_edge_feature，它由一个全一的 tensor 和一个全零的 tensor 拼接而成。然后，将 data.ligand_context_bond_type 中对应 idx_edge 的值进行 one-hot 编码，并赋给 data.compose_knn_edge_feature。"""
        
        return data


class FocalBuilder(object):
    def __init__(self, close_threshold=0.8, max_bond_length=2.4):
        self.close_threshold = close_threshold
        self.max_bond_length = max_bond_length
        super().__init__()

    def __call__(self, data:ProteinLigandData):
        # ligand_context_pos = data.ligand_context_pos
        # ligand_pos = data.ligand_pos
        ligand_masked_pos = data.ligand_masked_pos # 包含配体被掩蔽部分的位置信息的数据结构 
        protein_pos = data.protein_pos # 包含蛋白质位置信息的数据结构
        context_idx = data.context_idx # 包含上下文索引信息的数据结构
        masked_idx = data.masked_idx # 包含被掩蔽部分索引信息的数据结构
        old_bond_index = data.ligand_bond_index # 包含配体键索引信息的数据结构
        # old_bond_types = data.ligand_bond_type  # type: 0, 1, 2
        has_unmask_atoms = context_idx.nelement() > 0 # 判断是否有未被掩蔽的原子
        
        if has_unmask_atoms:
            # # get bridge bond index (mask-context bond)
            ind_edge_index_candidate = [
                (context_node in context_idx) and (mask_node in masked_idx)
                for mask_node, context_node in zip(*old_bond_index)
            ]  # the mask-context order is right
            bridge_bond_index = old_bond_index[:, ind_edge_index_candidate]
            # candidate_bond_types = old_bond_types[idx_edge_index_candidate]
            idx_generated_in_whole_ligand = bridge_bond_index[0]
            idx_focal_in_whole_ligand = bridge_bond_index[1]
            """检查context_idx和masked_idx中的元素来找出桥接键的索引，这些索引可能表示配体和蛋白质之间的连接。
            然后，它将这些索引用于old_bond_index来获取桥接键的索引。"""
            
            index_changer_masked = torch.zeros(masked_idx.max()+1, dtype=torch.int64)
            index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
            idx_generated_in_ligand_masked = index_changer_masked[idx_generated_in_whole_ligand]
            pos_generate = ligand_masked_pos[idx_generated_in_ligand_masked]
            """创建了两个索引更改器index_changer_masked和index_changer_context。
            这两个索引更改器都是使用torch.zeros创建的全零张量，大小分别为masked_idx.max()+1和context_idx.max()+1。
            然后，它将这两个张量在masked_idx和context_idx的位置上分别设置为torch.arange(len(masked_idx))和torch.arange(len(context_idx))的值。
            这可能是为了创建一个新的索引映射，将原始的索引映射到新的位置。"""

            data.idx_generated_in_ligand_masked = idx_generated_in_ligand_masked
            data.pos_generate = pos_generate

            index_changer_context = torch.zeros(context_idx.max()+1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))
            idx_focal_in_ligand_context = index_changer_context[idx_focal_in_whole_ligand]
            """使用这两个索引更改器来更新idx_generated_in_whole_ligand和idx_focal_in_whole_ligand的值，
            得到新的idx_generated_in_ligand_masked和idx_focal_in_ligand_context。这两个新的索引可能是在新的配体和蛋白质数据中的位置。"""
            
            idx_focal_in_compose = idx_focal_in_ligand_context  # if ligand_context was not before protein in the compose, this was not correct
            data.idx_focal_in_compose = idx_focal_in_compose
            
            data.idx_protein_all_mask = torch.empty(0, dtype=torch.long)  # no use if has context
            data.y_protein_frontier = torch.empty(0, dtype=torch.bool)  # no use if has context
            """将这些新的索引和位置信息存储到data对象中，以便后续的处理和使用。
            同时，它还创建了两个空的张量data.idx_protein_all_mask和data.y_protein_frontier，可能是为了在后续的处理中使用。"""
            
        else:  # # the initial atom. surface atoms between ligand and protein
            assign_index = radius(x=ligand_masked_pos, y=protein_pos, r=4., num_workers=16)
            """使用radius函数来找出配体和蛋白质之间距离在4以内的所有点的索引。radius函数的返回值assign_index是一个张量，其中包含了满足条件的点的索引。"""
            if assign_index.size(1) == 0:
                dist = torch.norm(data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1) # type: ignore
                assign_index = torch.nonzero(dist <= torch.min(dist)+1e-5)[0:1].transpose(0, 1)
            """检查assign_index的大小。如果assign_index为空，那么它会计算data.protein_pos和data.ligand_masked_pos之间的欧氏距离，
            并找出距离最小的点的索引，然后将这个索引赋值给assign_index。"""
            idx_focal_in_protein = assign_index[0]
            data.idx_focal_in_compose = idx_focal_in_protein  # no ligand context, so all composes are protein atoms
            """它从assign_index中获取idx_focal_in_protein的值，并将这个值存储到data.idx_focal_in_compose中。这可能是因为在这个情况下，所有的组合都是蛋白质原子，所以焦点索引在蛋白质中。"""
            
            data.pos_generate = ligand_masked_pos[assign_index[1]]
            data.idx_generated_in_ligand_masked = torch.unique(assign_index[1])  # for real of the contractive transform
            data.idx_protein_all_mask = data.idx_protein_in_compose  # for input of initial frontier prediction
            """使用assign_index来获取生成位置pos_generate和在配体被掩蔽部分中生成的索引idx_generated_in_ligand_masked。
            这两个值分别被存储到data.pos_generate和data.idx_generated_in_ligand_masked中。"""
            
            y_protein_frontier = torch.zeros_like(data.idx_protein_all_mask, dtype=torch.bool)  # for label of initial frontier prediction
            y_protein_frontier[torch.unique(idx_focal_in_protein)] = True
            data.y_protein_frontier = y_protein_frontier
            """它创建了一个全零的布尔张量y_protein_frontier，并将idx_focal_in_protein中的唯一值对应的位置设置为True。
            这个张量可能是用来表示蛋白质的前沿预测的标签。然后，它将这个张量存储到data.y_protein_frontier中。"""
            
        return data


class LigandRandomMask(object):
    '''
    '''
    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked

    def __call__(self, data:ProteinLigandData):
        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)

        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked
    
            
        idx = np.arange(num_atoms)
        np.random.shuffle(idx)
        
        # if data.keep_frag is not None:
        #     data.context_keep_frag = []
        #     for kf in data.keep_frag:
        #         data.context_keep_frag.append(np.where(idx==kf)[0][0])

        idx = torch.LongTensor(idx)
        masked_idx = idx[:num_masked]
        context_idx = idx[num_masked:]

        data.context_idx = context_idx  # for change bond index
        data.masked_idx = masked_idx

        # masked ligand atom element/feature/pos.
        data.ligand_masked_element = data.ligand_element[masked_idx]
        # data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction. these features are chem properties
        data.ligand_masked_pos = data.ligand_pos[masked_idx]

        # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        # new bond with ligand context atoms
        if data.ligand_bond_index.size(1) != 0:
            data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph( # type: ignore
                context_idx,
                data.ligand_bond_index,
                edge_attr = data.ligand_bond_type,
                relabel_nodes = True,
            )
        else:
            data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
            data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)
        # change context atom features that relate to bonds
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes = context_idx.size(0),
        )
        data.ligand_context_valence = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            valence=data.ligand_context_bond_type,
            num_nodes=context_idx.size(0)
        )
        data.ligand_context_num_bonds = torch.stack([
            LigandCountNeighbors.count_neighbors(
                data.ligand_context_bond_index, 
                symmetry=True, 
                valence=(data.ligand_context_bond_type == i).long(),
                num_nodes=context_idx.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        # re-calculate ligand_context_featrure_full
        data.ligand_context_feature_full = FeaturizeLigandAtom.change_features_of_neigh(
            data.ligand_context_feature_full,
            data.ligand_context_num_neighbors,
            data.ligand_context_valence,
            data.ligand_context_num_bonds
        )

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'random'

        return data


class LigandBFSMask(object):
    
    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, inverse=False):
        super().__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_num_masked = min_num_masked
        self.min_num_unmasked = min_num_unmasked
        self.inverse = inverse

    @staticmethod
    def get_bfs_perm(nbh_list):
        num_nodes = len(nbh_list)
        num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])

        bfs_queue = [random.randint(0, num_nodes-1)]
        bfs_perm = []
        num_remains = [num_neighbors.clone()]
        bfs_next_list = {}
        visited = {bfs_queue[0]}   

        num_nbh_remain = num_neighbors.clone()
        
        while len(bfs_queue) > 0:
            current = bfs_queue.pop(0)
            for nbh in nbh_list[current]:
                num_nbh_remain[nbh] -= 1
            bfs_perm.append(current)
            num_remains.append(num_nbh_remain.clone())
            next_candid = []
            for nxt in nbh_list[current]:
                if nxt in visited: continue
                next_candid.append(nxt)
                visited.add(nxt)
                
            random.shuffle(next_candid)
            bfs_queue += next_candid
            bfs_next_list[current] = copy.copy(bfs_queue)

        return torch.LongTensor(bfs_perm), bfs_next_list, num_remains

    def __call__(self, data):
        bfs_perm, bfs_next_list, num_remaining_nbs = self.get_bfs_perm(data.ligand_nbh_list)

        ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
        num_atoms = data.ligand_element.size(0)
        num_masked = int(num_atoms * ratio)
        if num_masked < self.min_num_masked:
            num_masked = self.min_num_masked
        if (num_atoms - num_masked) < self.min_num_unmasked:
            num_masked = num_atoms - self.min_num_unmasked

        if self.inverse:
            masked_idx = bfs_perm[:num_masked]
            context_idx = bfs_perm[num_masked:]
        else:
            masked_idx = bfs_perm[-num_masked:]
            context_idx = bfs_perm[:-num_masked]

        data.context_idx = context_idx  # for change bond index
        data.masked_idx = masked_idx

        # masked ligand atom element/feature/pos.
        data.ligand_masked_element = data.ligand_element[masked_idx]
        # data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction. these features are chem properties
        data.ligand_masked_pos = data.ligand_pos[masked_idx]
        
        # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
        data.ligand_context_element = data.ligand_element[context_idx]
        data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
        data.ligand_context_pos = data.ligand_pos[context_idx]

        # new bond with ligand context atoms
        if data.ligand_bond_index.size(1) != 0:
            data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
                context_idx,
                data.ligand_bond_index,
                edge_attr = data.ligand_bond_type,
                relabel_nodes = True,
            )
        else:
            data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
            data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)
        # re-calculate atom features that relate to bond
        data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            num_nodes = context_idx.size(0),
        )
        data.ligand_context_valence = LigandCountNeighbors.count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            valence=data.ligand_context_bond_type,
            num_nodes=context_idx.size(0)
        )
        data.ligand_context_num_bonds = torch.stack([
            LigandCountNeighbors.count_neighbors(
                data.ligand_context_bond_index, 
                symmetry=True, 
                valence=data.ligand_context_bond_type == i,
                num_nodes=context_idx.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        # re-calculate ligand_context_featrure_full
        data.ligand_context_feature_full = FeaturizeLigandAtom.change_features_of_neigh(
            data.ligand_context_feature_full,
            data.ligand_context_num_neighbors,
            data.ligand_context_valence,
            data.ligand_context_num_bonds
        )

        data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]

        data._mask = 'invbfs' if self.inverse else 'bfs'

        return data


class LigandMaskAll(LigandRandomMask):

    def __init__(self):
        super().__init__(min_ratio=1.0)


class LigandMaskZero(LigandRandomMask):
    
    def __init__(self):
        super().__init__(max_ratio=0.0, min_num_masked=0)




class LigandMixedMask(object):

    def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25, p_invbfs=0.25):
        super().__init__()

        self.t = [
            LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
            LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
        ]
        self.p = [p_random, p_bfs, p_invbfs]

    def __call__(self, data):
        f = random.choices(self.t, k=1, weights=self.p)[0]
        return f(data)


def get_mask(cfg):
    if cfg.type == 'bfs':
        return LigandBFSMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
        """如果 cfg.type 是 'bfs'，函数将返回一个 LigandBFSMask 对象。这个对象的构造函数接受一些参数，如 min_ratio、max_ratio、min_num_masked 和 min_num_unmasked，这些参数都从 cfg 中获取。"""
    elif cfg.type == 'random':
        return LigandRandomMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
        )
        """如果 cfg.type 是 'random'，函数将返回一个 LigandRandomMask 对象。这个对象的构造函数接受的参数与 LigandBFSMask 相同。"""
    elif cfg.type == 'mixed':
        return LigandMixedMask(
            min_ratio=cfg.min_ratio, 
            max_ratio=cfg.max_ratio, 
            min_num_masked=cfg.min_num_masked,
            min_num_unmasked=cfg.min_num_unmasked,
            p_random = cfg.p_random,
            p_bfs = cfg.p_bfs,
            p_invbfs = cfg.p_invbfs,
        )
        """如果 cfg.type 是 'mixed'，函数将返回一个 LigandMixedMask 对象。这个对象的构造函数接受的参数比前两者多一些，
        除了 min_ratio、max_ratio、min_num_masked 和 min_num_unmasked，还有 p_random、p_bfs 和 p_invbfs，这些参数都从 cfg 中获取。"""
    elif cfg.type == 'all':
        return LigandMaskAll()
        """如果 cfg.type 是 'all'，函数将返回一个 LigandMaskAll 对象。这个对象的构造函数不接受任何参数。"""
    else:
        raise NotImplementedError('Unknown mask: %s' % cfg.type)
        """如果 cfg.type 是其他值，函数将抛出一个 NotImplementedError 异常。"""

        
class ContrastiveSample(object):
    def __init__(self, num_real=50, num_fake=50, pos_real_std=0.05, pos_fake_std=2.0, knn=32, elements=None):
    # def __init__(self, knn=32, elements=None):
        super().__init__()
        self.num_real = num_real
        self.num_fake = num_fake
        self.pos_real_std = pos_real_std
        self.pos_fake_std = pos_fake_std
        self.knn = knn
        """设置了一些实例变量，包括 num_real、num_fake、pos_real_std、pos_fake_std 和 knn。
        这些变量分别表示生成的真实样本数量、生成的假样本数量、真实样本位置的标准差、假样本位置的标准差和最近邻数量。"""
        if elements is None:
            elements = [6,7,8,9,15,16,17] # C N O F P S Cl 表示7种原子序数
        self.elements = torch.LongTensor(elements) # 将 elements 转换为一个长整型张量，并将其赋值给 self.elements

    @property
    def num_elements(self):
        return self.elements.size(0)

    def __call__(self, data:ProteinLigandData):
        # Positive samples
        pos_real_mode = data.ligand_masked_pos
        element_real = data.ligand_masked_element
        """从 data 中提取了 ligand_masked_pos 和 ligand_masked_element 属性，并将它们赋值给 pos_real_mode 和 element_real 变量。"""
        
        # ind_real = data.ligand_masked_feature
        cls_real = data.ligand_masked_element.view(-1, 1) == self.elements.view(1, -1)
        assert (cls_real.sum(-1) > 0).all(), 'Unexpected elements.'
        """创建了一个名为 cls_real 的张量，该张量表示 ligand_masked_element 中的元素是否在 self.elements 中。
        这是通过使用 view 方法将 ligand_masked_element 和 self.elements 转换为相同的形状，然后使用 == 运算符进行比较来实现的。"""
        
        p = np.zeros(len(pos_real_mode), dtype=np.float32)
        p[data.idx_generated_in_ligand_masked] = 1.
        """创建了一个全零的 numpy 数组 p，长度为 pos_real_mode 的长度。然后，它将 data.idx_generated_in_ligand_masked 中对应的位置的值设置为 1。
        这样，p 就成了一个概率分布，其中 data.idx_generated_in_ligand_masked 中的位置的概率为 1，其他位置的概率为 0。"""
        
        real_sample_idx = np.random.choice(np.arange(pos_real_mode.size(0)), size=self.num_real, p=p/p.sum())
        """使用 numpy 的 random.choice 函数从 pos_real_mode.size(0)（即 pos_real_mode 的长度）个数中，
        按照 p 的概率分布，随机选择 self.num_real 个样本。这些样本的索引被存储在 real_sample_idx 中。"""

        data.pos_real = pos_real_mode[real_sample_idx]
        """从 pos_real_mode 中选择了一些样本，并将它们赋值给 data.pos_real。这些样本的索引存储在 real_sample_idx 中。"""
        data.pos_real += torch.randn_like(data.pos_real) * self.pos_real_std
        """将 data.pos_real 中的样本的位置进行了扰动。这个扰动是通过使用 torch.randn_like 函数生成一个与 data.pos_real 相同形状的随机张量，
        然后这个随机张量被乘以 self.pos_real_std 并添加到 data.pos_real 上。这样可以增加数据的多样性，使模型更健壮。"""
        data.element_real = element_real[real_sample_idx]
        """从 element_real 中选择了一些样本，并将它们赋值给 data.element_real。这些样本的索引存储在 real_sample_idx 中。"""
        data.cls_real = cls_real[real_sample_idx]
        """从 cls_real 中选择了一些样本，并将它们赋值给 data.cls_real。这些样本的索引同样存储在 real_sample_idx 中。"""
        # data.ind_real = ind_real[real_sample_idx]
        # data.num_neighbors_real = data.ligand_masked_num_neighbors[real_sample_idx]

        mask_ctx_edge_index_0 = data.mask_ctx_edge_index_0
        mask_ctx_edge_index_1 = data.mask_ctx_edge_index_1
        mask_ctx_edge_type = data.mask_ctx_edge_type
        """首先从 data 对象中获取了 mask_ctx_edge_index_0、mask_ctx_edge_index_1 和 mask_ctx_edge_type 属性，并将它们赋值给相应的变量。"""
        
        real_ctx_edge_idx_0_list, real_ctx_edge_idx_1_list, real_ctx_edge_type_list = [], [], [] # 创建了三个空列表，用于存储处理后的边的信息。
        for new_idx, real_node in enumerate(real_sample_idx):
            """使用 enumerate 函数遍历 real_sample_idx。对于每一个元素，enumerate 函数返回一个元组，
            其中第一个元素是元素的索引（在这里被命名为 new_idx），第二个元素是元素的值（在这里被命名为 real_node）。"""
            
            idx_edge = (mask_ctx_edge_index_0 == real_node)
            # real_ctx_edge_idx_0 = mask_ctx_edge_index_0[idx_edge]  # get edges related to this node
            real_ctx_edge_idx_1 = mask_ctx_edge_index_1[idx_edge]  # get edges related to this node
            real_ctx_edge_type = mask_ctx_edge_type[idx_edge]
            """找出 mask_ctx_edge_index_0 中等于 real_node 的元素，这些元素对应的边与当前的节点相关。
            它使用这些元素的索引从 mask_ctx_edge_index_1 和 mask_ctx_edge_type 中获取相关的边的信息，
            并将它们赋值给 real_ctx_edge_idx_1 和 real_ctx_edge_type。"""
            
            real_ctx_edge_idx_0 = new_idx * torch.ones(idx_edge.sum(), dtype=torch.long)  # change to new node index
            """创建了一个新的张量 real_ctx_edge_idx_0, 这个张量的所有元素都是 new_idx，长度是 idx_edge.sum()（即 idx_edge 中 True 的数量）。
            这个张量表示与当前节点相关的边的新索引。"""
            
            real_ctx_edge_idx_0_list.append(real_ctx_edge_idx_0)
            real_ctx_edge_idx_1_list.append(real_ctx_edge_idx_1)
            real_ctx_edge_type_list.append(real_ctx_edge_type)
            """将 real_ctx_edge_idx_0、real_ctx_edge_idx_1 和 real_ctx_edge_type 添加到相应的列表中。"""

        data.real_ctx_edge_index_0 = torch.cat(real_ctx_edge_idx_0_list, dim=-1)
        data.real_ctx_edge_index_1 = torch.cat(real_ctx_edge_idx_1_list, dim=-1)
        data.real_ctx_edge_type = torch.cat(real_ctx_edge_type_list, dim=-1)
        """使用 PyTorch 的 cat 函数将 real_ctx_edge_idx_0_list、real_ctx_edge_idx_1_list 和 real_ctx_edge_type_list 
        这三个列表中的张量沿着最后一个维度（由 -1 指定）拼接起来，
        然后将结果赋值给 data.real_ctx_edge_index_0、data.real_ctx_edge_index_1 和 data.real_ctx_edge_type。"""
        
        data.real_compose_edge_index_0 = data.real_ctx_edge_index_0
        data.real_compose_edge_index_1 = data.idx_ligand_ctx_in_compose[data.real_ctx_edge_index_1]  # actually are the same
        data.real_compose_edge_type = data.real_ctx_edge_type
        """将 real_ctx_edge_index_0、real_ctx_edge_index_1 和 real_ctx_edge_type 分别赋值给 data.real_compose_edge_index_0、
        data.real_compose_edge_index_1 和 data.real_compose_edge_type。"""

        # the triangle edge of the mask-compose edge
        row, col = data.real_compose_edge_index_0, data.real_compose_edge_index_1
        """从 data 对象中获取了 real_compose_edge_index_0 和 real_compose_edge_index_1 属性，并将它们赋值给 row 和 col。"""
        acc_num_edges = 0
        index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
        """创建了一个变量 acc_num_edges，用于累计边的数量，
        以及两个空列表 index_real_cps_edge_i_list 和 index_real_cps_edge_j_list，用于存储处理后的边的索引。"""
        for node in torch.arange(data.pos_real.size(0)):
            num_edges = (row == node).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, ) + acc_num_edges # type: ignore
            index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing=None)
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_real_cps_edge_i_list.append(index_edge_i)
            index_real_cps_edge_j_list.append(index_edge_j)
            """使用 torch.arange 函数遍历 data.pos_real.size(0)。
            对于每一个节点，它找出 row 中等于当前节点的元素，这些元素对应的边与当前的节点相关。
            它使用这些元素的数量创建了一个新的张量 index_edge_i，然后使用 torch.meshgrid 函数生成了两个二维张量，表示所有可能的边的组合。
            然后，它使用 flatten 方法将这两个二维张量转换为一维张量，并将它们添加到相应的列表中。"""
            acc_num_edges += num_edges
            """更新了 acc_num_edges，将当前节点的边的数量加到了累计的边的数量上。"""
        index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0)  # add len(real_compose_edge_index) in the dataloader for batch
        index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0)
        """使用 PyTorch 的 cat 函数将 index_real_cps_edge_i_list 和 index_real_cps_edge_j_list 
        这两个列表中的张量沿着第一个维度（由 0 指定）拼接起来，然后将结果赋值给 index_real_cps_edge_i 和 index_real_cps_edge_j。"""

        node_a_cps_tri_edge = col[index_real_cps_edge_i]  # the node of tirangle edge for the edge attention (in the compose)
        node_b_cps_tri_edge = col[index_real_cps_edge_j]
        """使用 index_real_cps_edge_i 和 index_real_cps_edge_j 作为索引，从 col 中选择元素，然后将结果赋值给 node_a_cps_tri_edge 和 node_b_cps_tri_edge。"""
        
        n_context = len(data.ligand_context_pos)
        adj_mat = torch.zeros([n_context, n_context], dtype=torch.long) - torch.eye(n_context, dtype=torch.long)
        adj_mat[data.ligand_context_bond_index[0], data.ligand_context_bond_index[1]] = data.ligand_context_bond_type
        tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
        """首先，它获取了 data.ligand_context_pos 的长度，也就是上下文中的节点数量，并将其赋值给 n_context。
        然后，它创建了一个新的零矩阵 adj_mat，其形状为 [n_context, n_context]，表示一个空的邻接矩阵。
        然后，它从这个零矩阵中减去一个单位矩阵，这样在邻接矩阵中，对角线上的元素就是 -1，表示没有边连接到自身。
        接着，它使用 data.ligand_context_bond_index[0] 和 data.ligand_context_bond_index[1] 作为索引，从 adj_mat 中选择元素，
        然后将 data.ligand_context_bond_type 赋值给这些元素。这样，邻接矩阵中的元素就表示了相应的边的类型。
        最后，它使用 node_a_cps_tri_edge 和 node_b_cps_tri_edge 作为索引，从 adj_mat 中选择元素，
        然后将结果赋值给 tri_edge_type。这样，tri_edge_type 就表示了三角形边的类型。"""
        
        tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]])).long()
        """创建了一个新的张量 tri_edge_feat，这个张量的元素是 tri_edge_type.view([-1, 1]) 和 torch.tensor([[-1, 0, 1, 2, 3]]) 的元素是否相等的结果。"""

        data.index_real_cps_edge_for_atten = torch.stack([
            index_real_cps_edge_i, index_real_cps_edge_j  # plus len(real_compose_edge_index_0) for dataloader batch
        ], dim=0)
        data.tri_edge_index = torch.stack([
            node_a_cps_tri_edge, node_b_cps_tri_edge  # plus len(compose_pos) for dataloader batch
        ], dim=0)
        data.tri_edge_feat = tri_edge_feat
        """使用 PyTorch 的 stack 函数将 index_real_cps_edge_i 和 index_real_cps_edge_j，以及 node_a_cps_tri_edge 和 node_b_cps_tri_edge 
        这两对张量沿着第一个维度（由 0 指定）堆叠起来，然后将结果赋值给 data.index_real_cps_edge_for_atten 和 data.tri_edge_index。
        然后，它将 tri_edge_feat 赋值给 data.tri_edge_feat。"""

        # Negative samples
        if len(data.ligand_context_pos) != 0: # all mask
            pos_fake_mode = data.ligand_context_pos[data.ligand_frontier]
        else:
            pos_fake_mode = data.protein_pos[data.y_protein_frontier]
        """检查 data.ligand_context_pos 的长度是否不为零。如果不为零，说明存在配体上下文位置信息，
        那么它就从 data.ligand_context_pos 中选择 data.ligand_frontier 对应的元素，然后将结果赋值给 pos_fake_mode。
        如果长度为零，说明不存在配体上下文位置信息，那么它就从 data.protein_pos 中选择 data.y_protein_frontier 对应的元素，
        然后将结果赋值给 pos_fake_mode。"""
        fake_sample_idx = np.random.choice(np.arange(pos_fake_mode.size(0)), size=self.num_fake)
        """使用 np.random.choice 函数从 np.arange(pos_fake_mode.size(0)) 中随机选择 self.num_fake 个元素，然后将结果赋值给 fake_sample_idx。"""
        pos_fake = pos_fake_mode[fake_sample_idx]
        """它使用 fake_sample_idx 作为索引，从 pos_fake_mode 中选择元素，然后将结果赋值给 pos_fake。"""
        data.pos_fake = pos_fake + torch.randn_like(pos_fake) * self.pos_fake_std / 2.
        """在 pos_fake 的基础上添加了一个正态分布的随机噪声，噪声的标准差是 self.pos_fake_std / 2.，然后将结果赋值给 data.pos_fake。"""

        # knn of query nodes
        real_compose_knn_edge_index = knn(x=data.compose_pos, y=data.pos_real, k=self.knn, num_workers=16)
        '''调用 knn 函数，将 data.compose_pos 作为 x，data.pos_real 作为 y，self.knn 作为 k，并设置 num_workers 为 16。
        这个函数会找到 data.pos_real 中每个元素在 data.compose_pos 中的 k 个最近邻。然后，它将结果赋值给 real_compose_knn_edge_index。'''
        data.real_compose_knn_edge_index_0, data.real_compose_knn_edge_index_1 = real_compose_knn_edge_index
        """将 real_compose_knn_edge_index 的两个元素分别赋值给 data.real_compose_knn_edge_index_0 和 data.real_compose_knn_edge_index_1。
        这样，data.real_compose_knn_edge_index_0 和 data.real_compose_knn_edge_index_1 就分别表示了组合位置与真实位置之间的 k-近邻关系的两个端点。"""
        
        fake_compose_knn_edge_index = knn(x=data.compose_pos, y=data.pos_fake, k=self.knn, num_workers=16)
        data.fake_compose_knn_edge_index_0, data.fake_compose_knn_edge_index_1 =fake_compose_knn_edge_index

        return data


# def get_contrastive_sampler(cfg):
#     return ContrastiveSample(
#         num_real = cfg.num_real,
#         num_fake = cfg.num_fake,
#         pos_real_std = cfg.pos_real_std,
#         pos_fake_std = cfg.pos_fake_std,
#     )






class EdgeSample(object):

    def __init__(self, cfg, num_bond_types=3):
        super().__init__()
        # self.neg_pos_ratio = cfg.neg_pos_ratio
        self.k = cfg.k
        # self.r = cfg.r
        self.num_bond_types = num_bond_types
    """调用了父类的初始化方法 super().__init__()。然后，从配置对象 cfg 中获取了参数 k，并将其赋值给实例变量 self.k。
    此外，还将参数 num_bond_types（默认值为3）赋值给实例变量 self.num_bond_types。
    这里的 k 和 num_bond_types 是用于后续处理蛋白质-配体数据的关键参数。k 可能用于确定近邻节点的数量，num_bond_types 则可能用于表示键的类型数量。"""

    def __call__(self, data:ProteinLigandData):
        
        ligand_context_pos = data.ligand_context_pos
        ligand_masked_pos = data.ligand_masked_pos
        context_idx = data.context_idx
        masked_idx = data.masked_idx
        old_bond_index = data.ligand_bond_index
        old_bond_types = data.ligand_bond_type
        """从 data 对象中提取了一些属性，包括 ligand_context_pos、ligand_masked_pos、context_idx、masked_idx、old_bond_index 和 old_bond_types。"""
        
        # candidate edge: mask-contex edge
        idx_edge_index_candidate = [
            (context_node in context_idx) and (mask_node in masked_idx)
            for mask_node, context_node in zip(*old_bond_index)
        ]  # the mask-context order is right
        """代码创建了一个名为 idx_edge_index_candidate 的列表，这个列表通过遍历 old_bond_index 中的元素并检查每个元素是否同时在 context_idx 和 masked_idx 中来生成。这个列表用于标记哪些边是候选边。"""
        
        candidate_bond_index = old_bond_index[:, idx_edge_index_candidate]
        candidate_bond_types = old_bond_types[idx_edge_index_candidate]
        """使用 idx_edge_index_candidate 从 old_bond_index 和 old_bond_types 中提取出候选边的索引和类型，分别存储在 candidate_bond_index 和 candidate_bond_types 中。"""
        
        # index changer
        index_changer_masked = torch.zeros(masked_idx.max()+1, dtype=torch.int64)
        index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
        """创建了一个名为 index_changer_masked 的张量，它的大小是 masked_idx 中的最大值加一，类型是 torch.int64。
        这个张量的初始值都是零，然后在 masked_idx 对应的位置上，设置为从零到 len(masked_idx) 的序列。这个张量可以用来改变索引。"""

        has_unmask_atoms = context_idx.nelement() > 0 # 检查context_idx 的元素数量是否大于零，来判断是否有未被掩码的原子，结果存储在 has_unmask_atoms 中。
        
        if has_unmask_atoms:
            index_changer_context = torch.zeros(context_idx.max()+1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))

            # new edge index (positive)
            new_edge_index_0 = index_changer_masked[candidate_bond_index[0]]
            new_edge_index_1 = index_changer_context[candidate_bond_index[1]]
            new_edge_index = torch.stack([new_edge_index_0, new_edge_index_1])
            new_edge_type = candidate_bond_types
            """创建一个名为 index_changer_context 的张量，用于改变索引。然后，使用 index_changer_masked 和 index_changer_context 对 candidate_bond_index 进行索引操作，生成新的边索引 new_edge_index，并将 candidate_bond_types 赋值给 new_edge_type。"""

            neg_version = 0
            false_edge_index = torch.empty([2, 0], dtype=torch.long)  # Define false_edge_index before using it
            false_edge_types = torch.empty([0], dtype=torch.int64)  # Initialize false_edge_types
            
            if neg_version == 1:  # radiu + tri_edge, 如果 neg_version 为 1，那么会使用半径和三角形边的方式生成负样本。
                # negative edge index (types = 0)
                id_edge_pos = new_edge_index[0] * len(context_idx) + new_edge_index[1]
                # 1. radius all edges
                edge_index_radius = radius(ligand_context_pos, ligand_masked_pos, r=3.0, num_workers=16)  # r = 3.0
                id_edge_radius = edge_index_radius[0] * len(context_idx) + edge_index_radius[1]
                not_pos_in_radius = torch.tensor([id_ not in id_edge_pos for id_ in id_edge_radius])
                """计算所有可能的负样本边的索引 id_edge_pos。然后，使用 radius 函数计算所有在给定半径内的边的索引 edge_index_radius，并计算这些边的索引 id_edge_radius。接着，找出在 id_edge_radius 中但不在 id_edge_pos 中的边，这些边就是真正的负样本边。"""
                # 2. pick true neg edges and random choice
                if not_pos_in_radius.size(0) > 0:
                    edge_index_neg = edge_index_radius[:, not_pos_in_radius]
                    dist = torch.norm(ligand_masked_pos[edge_index_neg[0]] - ligand_context_pos[edge_index_neg[1]], p=2.0, dim=-1) # type: ignore
                    probs = torch.clip(0.8 * (dist ** 2) - 4.8 * dist + 7.3 + 0.4, min=0.5, max=0.95)
                    values = torch.rand(len(dist))
                    choice = values < probs
                    edge_index_neg = edge_index_neg[:, choice]
                else: 
                    edge_index_neg = torch.empty([2, 0], dtype=torch.long)
                """如果存在真正的负样本边，就计算这些边的距离 dist，并根据这个距离计算出一个概率 probs。接着，生成一个随机值 values，并根据这个随机值和概率 probs 选择出最终的负样本边 edge_index_neg。如果不存在真正的负样本边，那么就直接将 edge_index_neg 设置为空张量。"""
                # 3. edges form ring should be choicen
                bond_index_ctx = data.ligand_context_bond_index
                edge_index_ring_candidate = [[], []]
                for node_i, node_j in zip(*new_edge_index):
                    node_k_all = bond_index_ctx[1, bond_index_ctx[0] == node_j]
                    edge_index_ring_candidate[0].append( torch.ones_like(node_k_all) * node_i)
                    edge_index_ring_candidate[1].append(node_k_all)
                edge_index_ring_candidate[0] = torch.cat(edge_index_ring_candidate[0], dim=0) # type: ignore
                edge_index_ring_candidate[1] = torch.cat(edge_index_ring_candidate[1], dim=0) # type: ignore
                id_ring_candidate = edge_index_ring_candidate[0] * len(context_idx) + edge_index_ring_candidate[1]
                edge_index_ring_candidate = torch.stack(edge_index_ring_candidate, dim=0) # type: ignore
                not_pos_in_ring = torch.tensor([id_ not in id_edge_pos for id_ in id_ring_candidate])
                if not_pos_in_ring.size(0) > 0:
                    edge_index_ring = edge_index_ring_candidate[:, not_pos_in_ring]
                    dist = torch.norm(ligand_masked_pos[edge_index_ring[0]] - ligand_context_pos[edge_index_ring[1]], p=2.0, dim=-1) # type: ignore
                    edge_index_ring = edge_index_ring[:, dist < 4.0]
                else:
                    edge_index_ring = torch.empty([2, 0], dtype=torch.long)
                """计算出所有可能形成环的边的索引 edge_index_ring_candidate，并找出在 edge_index_ring_candidate 中但不在 id_edge_pos 中的边，这些边就是可能形成环的负样本边。然后，如果存在可能形成环的负样本边，就计算这些边的距离 dist，并根据这个距离选择出最终的可能形成环的负样本边 edge_index_ring。如果不存在可能形成环的负样本边，那么就直接将 edge_index_ring 设置为空张量。"""
                # 4.cat neg and ring
                false_edge_index = torch.cat([
                    edge_index_neg, edge_index_ring
                ], dim=-1)
                false_edge_types = torch.zeros(len(false_edge_index[0]), dtype=torch.int64)
                """将 edge_index_neg 和 edge_index_ring 合并成 false_edge_index，并生成对应的类型 false_edge_types。"""
                
            elif neg_version == 0:  # knn edge 使用 k-最近邻 (k-NN) 方法生成负样本边。
                edge_index_knn = knn(ligand_context_pos, ligand_masked_pos, k=self.k, num_workers=16)
                dist = torch.norm(ligand_masked_pos[edge_index_knn[0]] - ligand_context_pos[edge_index_knn[1]], p=2.0, dim=-1) # type: ignore
                idx_sort = torch.argsort(dist)  #  choose negative edges as short as possible
                """首先，使用 knn 函数计算 ligand_context_pos 和 ligand_masked_pos 之间的 k-最近邻边 edge_index_knn。然后，计算这些边的欧几里得距离 dist。接着，使用 argsort 函数对这些距离进行排序，得到的 idx_sort 是距离从小到大的边的索引。"""
                
                num_neg_edges = min(len(ligand_masked_pos) * (self.k // 2) + len(new_edge_index[0]), len(idx_sort))
                idx_sort = torch.unique(
                    torch.cat([
                        idx_sort[:num_neg_edges],
                        torch.linspace(0, len(idx_sort), len(ligand_masked_pos)+1, dtype=torch.long)[:-1]  # each mask pos at least has one negative edge
                    ], dim=0)
                )
                """然后，计算负样本边的数量 num_neg_edges，这个数量是 ligand_masked_pos 的长度乘以 k 的一半，再加上 new_edge_index[0] 的长度，但不能超过 idx_sort 的长度。
                接着，使用 unique 函数对 idx_sort[:num_neg_edges] 和 linspace(0, len(idx_sort), len(ligand_masked_pos)+1, dtype=torch.long)[:-1] 进行合并和去重，
                得到的 idx_sort 是每个 ligand_masked_pos 至少有一个负样本边的索引。"""
                
                edge_index_knn = edge_index_knn[:, idx_sort]
                id_edge_knn = edge_index_knn[0] * len(context_idx) + edge_index_knn[1]  # delete false negative edges
                id_edge_new = new_edge_index[0] * len(context_idx) + new_edge_index[1]
                idx_real_edge_index = torch.tensor([id_ in id_edge_new for id_ in id_edge_knn])
                """使用新的 idx_sort 对 edge_index_knn 进行筛选，得到新的 edge_index_knn。
                接着计算 edge_index_knn 和 new_edge_index 的边的唯一标识 id_edge_knn 和 id_edge_new。
                然后，找出 id_edge_knn 中存在于 id_edge_new 的边的索引 idx_real_edge_index。"""
                
                false_edge_index = edge_index_knn[:, ~idx_real_edge_index]
                false_edge_types = torch.zeros(len(false_edge_index[0]), dtype=torch.int64)
                """找出 edge_index_knn 中不是真实边的边 false_edge_index，并生成对应的类型 false_edge_types，这些类型都是 0，表示这些边都是负样本边。"""

            # cat 
            # print('Num of pos : neg edge:', len(new_edge_type), len(false_edge_types), len(new_edge_type) / len(false_edge_types))
            
            new_edge_index = torch.cat([new_edge_index, false_edge_index], dim=-1)
            new_edge_type = torch.cat([new_edge_type, false_edge_types], dim=0)
            """将 new_edge_index 和 false_edge_index 合并成 new_edge_index，并将 new_edge_type 和 false_edge_types 合并成 new_edge_type。"""

            data.mask_ctx_edge_index_0 = new_edge_index[0]
            data.mask_ctx_edge_index_1 = new_edge_index[1]
            """将 new_edge_index 的第一维和第二维分别赋值给 data.mask_ctx_edge_index_0 和 data.mask_ctx_edge_index_1。
            new_edge_index 是一个二维张量，包含了所有的边索引，第一维是边的起始节点的索引，第二维是边的结束节点的索引。"""
            data.mask_ctx_edge_type = new_edge_type
            """将 new_edge_type 赋值给 data.mask_ctx_edge_type。new_edge_type 是一个一维张量，包含了所有的边类型。"""
            data.mask_compose_edge_index_0 = data.mask_ctx_edge_index_0
            """将 data.mask_ctx_edge_index_0 赋值给 data.mask_compose_edge_index_0。表示在组合图中，边的起始节点的索引和在上下文图中是一样的。"""
            data.mask_compose_edge_index_1 = data.idx_ligand_ctx_in_compose[data.mask_ctx_edge_index_1]  # actually are the same
            """在组合图中，边的结束节点的索引是通过 data.idx_ligand_ctx_in_compose 映射得到的，这个映射表将上下文图中的节点索引映射到组合图中。"""
            data.mask_compose_edge_type = new_edge_type
            """将 new_edge_type 赋值给 data.mask_compose_edge_type，表示在组合图中，边的类型和在上下文图中是一样的。"""
            
        else:
            data.mask_ctx_edge_index_0 = torch.empty([0], dtype=torch.int64)
            data.mask_ctx_edge_index_1 = torch.empty([0], dtype=torch.int64)
            data.mask_ctx_edge_type = torch.empty([0], dtype=torch.int64)
            data.mask_compose_edge_index_0 = torch.empty([0], dtype=torch.int64)
            data.mask_compose_edge_index_1 = torch.empty([0], dtype=torch.int64)
            data.mask_compose_edge_type = torch.empty([0], dtype=torch.int64)
            """如果没有未被掩码的原子，那么就将上下文图中的边索引和类型都设置为空张量。"""

        return data
