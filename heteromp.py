import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear
from torch_geometric.data import HeteroData

class UniversalGlobalAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads, dropout=0.2):
        super(UniversalGlobalAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_projs = nn.ModuleDict()
        self.k_projs = nn.ModuleDict()
        self.v_projs = nn.ModuleDict()
        self.out_projs = nn.ModuleDict()
        
        self.dropout = nn.Dropout(dropout)
    
    def _ensure_projections(self, edge_types, device):
        for edge_type in edge_types:
            src, _, dst = edge_type
            key = f"{src}_{dst}"
            
            if key not in self.q_projs:
                self.q_projs[key] = nn.Linear(self.hidden_channels, self.hidden_channels).to(device)
                self.k_projs[key] = nn.Linear(self.hidden_channels, self.hidden_channels).to(device)
                self.v_projs[key] = nn.Linear(self.hidden_channels, self.hidden_channels).to(device)
                self.out_projs[key] = nn.Linear(self.hidden_channels, self.hidden_channels).to(device)
            else:
                self.q_projs[key] = self.q_projs[key].to(device)
                self.k_projs[key] = self.k_projs[key].to(device)
                self.v_projs[key] = self.v_projs[key].to(device)
                self.out_projs[key] = self.out_projs[key].to(device)

    def forward(self, x_dict, edge_index_dict):
        device = next(iter(x_dict.values())).device
        edge_types = list(edge_index_dict.keys())
        self._ensure_projections(edge_types, device)
        
        result_dict = {node_type: [] for node_type in x_dict}
        
        for edge_type in edge_types:
            src, rel, dst = edge_type
            
            src_x = x_dict[src]  # [num_src_nodes, hidden_channels]
            dst_x = x_dict[dst]  # [num_dst_nodes, hidden_channels]
            
            edge_index = edge_index_dict[edge_type]  # [2, num_edges]
            src_idx, dst_idx = edge_index[0], edge_index[1]
            
            src_idx = src_idx.to(device)
            dst_idx = dst_idx.to(device)
            
            q = self.q_projs[f"{src}_{dst}"](src_x[src_idx])  # [num_edges, hidden_channels]
            k = self.k_projs[f"{src}_{dst}"](dst_x[dst_idx])  # [num_edges, hidden_channels]
            v = self.v_projs[f"{src}_{dst}"](dst_x[dst_idx])  # [num_edges, hidden_channels]
            
            q = q.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
            k = k.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
            v = v.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
            
            attn_scores = (q * k).sum(dim=-1) * self.scaling  # [num_edges, num_heads]
            
            attn_weights = torch.zeros_like(attn_scores, device=device)
            for i in range(src_x.size(0)):
                mask = (src_idx == i)
                if mask.sum() > 0:
                    attn_weights[mask] = F.softmax(attn_scores[mask], dim=0)
            
            attn_weights = attn_weights.unsqueeze(-1)  # [num_edges, num_heads, 1]
            weighted_values = v * attn_weights  # [num_edges, num_heads, head_dim]
            
            output = torch.zeros(src_x.size(0), self.num_heads, self.head_dim, device=device)
            for i in range(src_x.size(0)):
                mask = (src_idx == i)
                if mask.sum() > 0:
                    output[i] = weighted_values[mask].sum(dim=0)  # [num_heads, head_dim]
            
            output = output.reshape(src_x.size(0), -1)  # [num_src_nodes, hidden_channels]
            output = self.out_projs[f"{src}_{dst}"](output)  # [num_src_nodes, hidden_channels]
            output = self.dropout(output)
            
            result_dict[src].append(output)
        
        for node_type in result_dict:
            if result_dict[node_type]:
                result_dict[node_type] = sum(result_dict[node_type])
            else:
                result_dict[node_type] = x_dict[node_type]
        
        return result_dict


class UniversalHeteroGAT(nn.Module):
    def __init__(self, node_configs, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(UniversalHeteroGAT, self).__init__()        
        self.heads = heads
        self.hidden_channels = hidden_channels
        self.node_configs = node_configs
        
        self.node_gats = nn.ModuleDict()
        self.feature_projections = nn.ModuleDict()
        
        for node_type, config in node_configs.items():
            self.feature_projections[node_type] = nn.Linear(
                config['in_channels'], 
                config['in_channels']
            )
            
            self.node_gats[node_type] = GATv2Conv(
                in_channels=config['in_channels'],
                out_channels=hidden_channels,
                heads=heads,
                concat=True,
                dropout=dropout
            )
        
        self.global_atten = UniversalGlobalAttention(
            hidden_channels=hidden_channels * heads,
            num_heads=heads,
            dropout=dropout
        )
        self.output_linear = Linear(hidden_channels * heads, out_channels)
        
    def _adapt_features(self, x, expected_dim, node_type):
        current_dim = x.size(-1)
        device = x.device
        
        if current_dim != expected_dim:
            if node_type not in self.feature_projections or \
               self.feature_projections[node_type].in_features != current_dim:
                
                self.feature_projections[node_type] = nn.Linear(
                    current_dim, expected_dim
                ).to(device)
                
                self.node_gats[node_type] = GATv2Conv(
                    in_channels=expected_dim,
                    out_channels=self.hidden_channels,
                    heads=self.heads,
                    concat=True,
                    dropout=0.2
                ).to(device)
            else:
                self.feature_projections[node_type] = self.feature_projections[node_type].to(device)
                self.node_gats[node_type] = self.node_gats[node_type].to(device)
            
            x = self.feature_projections[node_type](x)
        
        return x
        
    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        
        device = next(iter(x_dict.values())).device
        for edge_type in edge_index_dict:
            edge_index_dict[edge_type] = edge_index_dict[edge_type].to(device)
        x_dict_updated = {}
        for node_type, config in self.node_configs.items():
            if node_type in x_dict:
                x_dict[node_type] = x_dict[node_type].to(device)
                adapted_x = self._adapt_features(
                    x_dict[node_type], 
                    config['in_channels'], 
                    node_type
                )
                
                if config['self_edge_type'] in edge_index_dict:
                    node_x = self.node_gats[node_type](
                        adapted_x, 
                        edge_index_dict[config['self_edge_type']]
                    )
                    x_dict_updated[node_type] = F.elu(node_x)
                else:
                    dummy_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                    node_x = self.node_gats[node_type](adapted_x, dummy_edge_index)
                    x_dict_updated[node_type] = F.elu(node_x)
        
        global_x_dict = self.global_atten(x_dict_updated, edge_index_dict)
        
        for node_type in global_x_dict:
            global_x_dict[node_type] = F.elu(global_x_dict[node_type])
        
        atom_output = self.output_linear(global_x_dict['atom'])
        
        return atom_output, global_x_dict


def create_fg_hetero_gat(atom_in_channels=9, fg_in_channels=167, hidden_channels=64, out_channels=1, heads=4):
    node_configs = {
        'atom': {
            'in_channels': atom_in_channels,
            'self_edge_type': ('atom', 'bonds', 'atom')
        },
        'fg': {
            'in_channels': fg_in_channels,
            'self_edge_type': ('fg', 'bonded', 'fg')
        }
    }
    return UniversalHeteroGAT(node_configs, hidden_channels, out_channels, heads)


def create_bond_hetero_gat(atom_in_channels=9, bond_in_channels=4, hidden_channels=64, out_channels=1, heads=4):
    node_configs = {
        'atom': {
            'in_channels': atom_in_channels,
            'self_edge_type': ('atom', 'bonds', 'atom')
        },
        'bond': {
            'in_channels': bond_in_channels,
            'self_edge_type': ('bond', 'angle', 'bond')
        }
    }
    return UniversalHeteroGAT(node_configs, hidden_channels, out_channels, heads)

def create_and_process_fg_graph(mol, atom_data):
    from utils.fg_extractor import create_hetero_graph
    hetero_data: HeteroData = create_hetero_graph(mol, atom_data)
    
    model = create_fg_hetero_gat(
        atom_in_channels=atom_data.x.shape[1],
        fg_in_channels=167,  # MACCS
        hidden_channels=64,
        out_channels=1,
        heads=4
    )
    
    output, node_representations = model(hetero_data)
    return output, node_representations, hetero_data


def create_and_process_bond_graph(mol, atom_data):
    from utils.angle_extractor import create_bond_graph
    hetero_data: HeteroData = create_bond_graph(mol, atom_data)
    
    model = create_bond_hetero_gat(
        atom_in_channels=atom_data.x.shape[1],
        bond_in_channels=4,
        hidden_channels=64,
        out_channels=1,
        heads=4
    )
    
    output, node_representations = model(hetero_data)
    return output, node_representations, hetero_data
    