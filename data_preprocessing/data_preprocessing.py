import pandas as pd
import networkx as nx
from collections import defaultdict
import torch

def build_knowledge_graph(event_log_path):
    """
    从事件日志构建知识图谱（KG）。
    
    输入:
        event_log_path: CSV文件路径，包含列 case_id, activity, resource, timestamp。
    
    输出:
        dfg: 直接跟随图（DiGraph），边权重表示活动间的转移频率。
        context_graph: 上下文图（MultiDiGraph），包含活动-资源、活动-时间等关系。
    """
    # 1. 加载和清洗数据
    df = pd.read_csv(event_log_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])


    # 2. 初始化图
    dfg = nx.DiGraph()          # 直接跟随图（有向，边权重=频率）
    context_graph = nx.MultiDiGraph()  # 上下文图（支持多种边类型）

    # 3. 构建直接跟随图（DFG）
    case_activities = defaultdict(list)
    for case_id, group in df.groupby('case_id'):
        activities = group['activity'].tolist()
        case_activities[case_id] = activities
        for i in range(len(activities) - 1):
            src, tgt = activities[i], activities[i + 1]
            if dfg.has_edge(src, tgt):
                dfg[src][tgt]['weight'] += 1
            else:
                dfg.add_edge(src, tgt, weight=1, edge_type='dfg')

    # 4. 构建上下文图
    for _, row in df.iterrows():
        activity = row['activity']
        resource = row['resource']
        hour = row['timestamp'].hour
        time_slot = 'morning' if 6 <= hour < 12 else 'afternoon'

        # 添加活动-资源边
        context_graph.add_edge(activity, resource, edge_type='act_res')
        # 添加活动-时间边
        context_graph.add_edge(activity, time_slot, edge_type='act_time')

    return dfg, context_graph

def retrieve_subgraph(dfg, context_graph, activities):
    """
    从KG中检索与历史活动序列相关的子图。
    
    输入:
        dfg: 直接跟随图。
        context_graph: 上下文图。
        activities: 历史活动列表（如 ["submit_order", "verify_payment"]）。
    
    输出:
        subgraph: 合并DFG和上下文相关节点的子图。
    """
    subgraph = nx.DiGraph()
    
    # 1. 添加DFG中的活动转移边
    for i in range(len(activities) - 1):
        src, tgt = activities[i], activities[i + 1]
        if dfg.has_edge(src, tgt):
            subgraph.add_edge(src, tgt, **dfg[src][tgt])
    
    # 2. 添加上下文节点（最后活动的资源和时间）
    last_act = activities[-1]
    for _, neighbor, attr in context_graph.edges(last_act, data=True):
        subgraph.add_edge(last_act, neighbor, **attr)
    
    return subgraph

# 使用示例
dfg, context_graph = build_knowledge_graph("../data/helpdesk_processed.csv")
subgraph = retrieve_subgraph(dfg, context_graph, ["assign seriousness", "take in charge ticket"])

node_types = {
    'activity': 0,
    'resource': 1,
    'time': 2
}
edge_types = {
    'dfg': 0,       # 直接跟随边
    'act_res': 1,    # 活动-资源边
    'act_time': 2    # 活动-时间边
}


def build_node_features_and_types(dfg, context_graph):
    # 收集所有节点
    all_nodes = set(dfg.nodes())
    all_nodes.update(context_graph.nodes())
    
    # 初始化节点特征和类型
    node_features = []
    node_type_labels = []
    node_id_to_idx = {}  # 节点名称到索引的映射
    
    # 为每个节点分配特征和类型
    for idx, node in enumerate(all_nodes):
        node_id_to_idx[node] = idx
        
        # 判断节点类型并分配特征
        if node in dfg.nodes():
            # 活动节点：特征为 one-hot 或预训练嵌入（此处简化）
            node_features.append([1.0, 0.0, 0.0])  # 示例：3维特征
            node_type_labels.append(node_types['activity'])
        elif node in context_graph.nodes():
            if isinstance(node, str) and node in ['morning', 'afternoon']:
                # 时间节点
                node_features.append([0.0, 0.0, 1.0])
                node_type_labels.append(node_types['time'])
            else:
                # 资源节点
                node_features.append([0.0, 1.0, 0.0])
                node_type_labels.append(node_types['resource'])
    
    return (
        torch.FloatTensor(node_features),  # x
        torch.LongTensor(node_type_labels),  # node_type
        node_id_to_idx
    )

x, node_type, node_id_to_idx = build_node_features_and_types(dfg, context_graph)



def build_edge_index_and_type(dfg, context_graph, node_id_to_idx):
    edge_index = []
    edge_type_labels = []
    
    # 处理直接跟随图（DFG）的边
    for src, tgt, attr in dfg.edges(data=True):
        edge_index.append([node_id_to_idx[src], node_id_to_idx[tgt]])
        edge_type_labels.append(edge_types['dfg'])
    
    # 处理上下文图的边
    for src, tgt, attr in context_graph.edges(data=True):
        edge_index.append([node_id_to_idx[src], node_id_to_idx[tgt]])
        edge_type_labels.append(edge_types[attr['edge_type']])
    
    # 转换为 PyTorch 张量
    edge_index = torch.LongTensor(edge_index).t().contiguous()  # [2, num_edges]
    edge_type = torch.LongTensor(edge_type_labels)  # [num_edges]
    
    return edge_index, edge_type

edge_index, edge_type = build_edge_index_and_type(dfg, context_graph, node_id_to_idx)

print("Node Features (x):", x.shape)           # [num_nodes, in_channels]
print("Edge Index:", edge_index.shape)         # [2, num_edges]
print("Edge Type:", edge_type.shape)           # [num_edges]
print("Node Type:", node_type.shape)           # [num_nodes]