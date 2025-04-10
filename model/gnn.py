import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_node_types, num_edge_types, heads=4, dropout=0.2):
        """
        Args:
            input_dim (int): 输入节点特征维度 (x.shape[1])
            hidden_dim (int): GNN 隐藏层维度
            output_dim (int): 输出维度（活动类别数量）
            num_node_types (int): 节点类型数量
            num_edge_types (int): 边类型数量
            heads (int): 多头注意力的头数
            dropout (float): Dropout 概率
        """
        super(GNN, self).__init__()
        
        # 节点嵌入层
        self.node_embedding = nn.Embedding(num_node_types, input_dim)
        
        # 边嵌入层
        self.edge_embedding = nn.Embedding(num_edge_types, input_dim)
        
        # 图注意力层
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout)
        
        # BatchNorm 和 Dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * heads)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_type, node_type):
        """
        Args:
            x (Tensor): 节点特征矩阵 [num_nodes, input_dim]
            edge_index (Tensor): 边索引 [2, num_edges]
            edge_type (Tensor): 边类型 [num_edges]
            node_type (Tensor): 节点类型 [num_nodes]
        Returns:
            logits (Tensor): 每个节点的预测分布 [num_nodes, output_dim]
        """
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # 节点类型嵌入
        node_type_emb = self.node_embedding(node_type)  # [num_nodes, input_dim]
        x = x + node_type_emb  # 将节点类型嵌入加到节点特征中

        # 边类型嵌入
        edge_type_emb = self.edge_embedding(edge_type)  # [num_edges, input_dim]

        # 图注意力层 1
        x = self.conv1(x, edge_index, edge_attr=edge_type_emb)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 图注意力层 2
        x = self.conv2(x, edge_index, edge_attr=edge_type_emb)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 输出层
        logits = self.fc(x)  # [num_nodes, output_dim]
        return logits