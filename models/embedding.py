import torch
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F


class AtomEmbedding(Module):
    def __init__(self, in_scalar, in_vector,
                 out_scalar, out_vector, vector_normalizer=20.):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)
        """首先调用super().__init__()来初始化父类，这里的父类是torch.nn.Module，它是所有PyTorch神经网络模块的基类。
        然后，它断言输入向量的维度必须为1，这可能是因为在这个类中，输入向量的维度被固定为1。
        接着，它将输入标量的维度、向量标准化器存储到类的属性中。
        向量标准化器的默认值为20，这个值可能是经过实验得出的最优值。然后，它使用Linear类来创建标量嵌入和向量嵌入。
        Linear类是一个线性变换层，它的构造函数接受输入特征的大小和输出特征的大小作为参数。"""

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        """在forward函数中，首先将输入向量除以向量标准化器，这是一个常见的特征标准化步骤。
        然后，它断言输入向量的形状必须为(3,)，这可能是因为在这个类中，输入向量的形状被固定为(3,)。
        接着，它使用标量嵌入来处理标量输入，使用向量嵌入来处理向量输入。最后，它返回标量嵌入和向量嵌入。"""
        return sca_emb, vec_emb
        
        
