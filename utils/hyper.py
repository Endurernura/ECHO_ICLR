import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, Sequential
from torch_geometric.data import Data
import numpy as np
from scipy.sparse import csr_matrix, block_diag
from sklearn.metrics.pairwise import cosine_distances

# 超图构建工具函数
def build_sub_hypergraph(X_m, n_neighbors=2):
    """为单模态特征构建子超图"""
    n_models = len(X_m)
    # 调整特征形状以计算距离矩阵 [n_models, hidden_dim]
    X_np = torch.stack(X_m).detach().cpu().numpy()
    dist_matrix = cosine_distances(X_np)
    
    edges = []
    weights = []
    for i in range(n_models):
        neighbors = np.argsort(dist_matrix[i])[1:1+n_neighbors]
        edge = np.unique([i] + neighbors.tolist())
        edges.append(edge)
        weights.append(1.0 / (np.mean(dist_matrix[i, neighbors]) + 1e-8))
    
    # 构建关联矩阵H^m
    n_edges = len(edges)
    row, col = [], []
    for j in range(n_edges):
        for i in edges[j]:
            row.append(i)
            col.append(j)
    data = np.ones_like(row)
    H_m = csr_matrix((data, (row, col)), shape=(n_models, n_edges))
    
    # 超边权重矩阵W^m
    W_m = csr_matrix(np.diag(weights))
    return {'H': H_m, 'W': W_m}

def compute_adjacency_matrix(hypergraph):
    """计算归一化邻接矩阵A"""
    H = hypergraph['H']
    W = hypergraph['W']
    
    # 节点度矩阵D_v和超边度矩阵D_e
    D_v = np.array(H @ W.sum(axis=1)).flatten()
    D_v_sqrt_inv = np.diag(1 / np.sqrt(D_v + 1e-8))
    D_e = np.array(H.sum(axis=0)).flatten()
    D_e_inv = np.diag(1 / (D_e + 1e-8))
    
    # 计算归一化邻接矩阵A [n_models, n_models]
    A_np = D_v_sqrt_inv @ H.T @ D_e_inv @ W @ H @ D_v_sqrt_inv
    return torch.tensor(A_np, dtype=torch.float32)

def create_gnn_model(conv_type, input_dim, hidden_dim):
    """创建标准化的GNN"""
    return Sequential('x, edge_index, batch', [
        (conv_type(input_dim, hidden_dim), 'x, edge_index -> x'),
        torch.nn.ReLU(inplace=True),
        (conv_type(hidden_dim, hidden_dim), 'x, edge_index -> x'),
        (global_mean_pool, 'x, batch -> x'),
        (lambda x: x.squeeze(0), 'x -> x')  # 移除批次维度
    ])

class HyperFusion(torch.nn.Module):
    """使用方法：
        fusion = HyperFusion(input_dim, hidden_dim)
        preds = fusion(data_1, data_2, data_3)
        目前版本使用线性分类器，暂未引入fc分类器
    """

    def __init__(self, input_dim, hidden_dim, n_neighbors=2):
        super(HyperFusion, self).__init__()
        self.n_neighbors = n_neighbors
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def get_modal_features(self, data_1, data_2, data_3):
        data_list = [data_1, data_2, data_3]
        modal_features = [data.x for data in data_list]
        return modal_features

    def forward(self, data_1, data_2, data_3):
        """前向传播：特征提取 -> 超图融合 -> 预测"""
        modal_features = self.get_modal_features(data_1, data_2, data_3)
        hypergraph = build_sub_hypergraph(modal_features, self.n_neighbors)
        
        # 3. 计算归一化邻接矩阵
        A = compute_adjacency_matrix(hypergraph)
        
        # 消息传递
        X_m = torch.stack(modal_features)  # 形状: [n_models, hidden_dim]
        A = A.to(X_m.device)
        
        X_fused = torch.matmul(A, X_m)  # 结果形状: [n_models, hidden_dim]
        
        # 5. 聚合融合特征并输出二分类结果
        X_agg = X_fused.mean(dim=0)  # 平均所有模型的融合特征 [hidden_dim]
        logits = self.classifier(X_agg.unsqueeze(0))  # 添加批次维度
        preds = F.sigmoid(logits).squeeze()
        
        return preds

# 测试代码
if __name__ == "__main__":
    input_dim = 16
    hidden_dim = 32
    
    # 初始化HyperFusion模型（现在不需要外部传入GNN模型）
    hyperfusion = HyperFusion(input_dim, hidden_dim)
    
    # 模拟图数据
    data = Data(
        x=torch.randn(10, input_dim),  # 10个节点，每个节点16维特征
        edge_index=torch.randint(0, 10, (2, 20)),  # 20条边
        batch=torch.zeros(10, dtype=torch.long)  # 单图
    )

    gcn = create_gnn_model(GCNConv, input_dim, hidden_dim)
    gat = create_gnn_model(GATConv, input_dim, hidden_dim)
    sage = create_gnn_model(SAGEConv, input_dim, hidden_dim)
    data_1 = gcn(data.x, data.edge_index, data.batch)
    data_2 = gat(data.x, data.edge_index, data.batch)
    data_3 = sage(data.x, data.edge_index, data.batch)

    # 直接使用forward方法进行预测
    preds = hyperfusion(data_1, data_2, data_3)
    print("二分类预测结果：", preds.item())  # 输出0-1之间的概率值
