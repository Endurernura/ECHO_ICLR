import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
from torch import cuda, device, compile
from utils.angle_extractor import create_bond_graph as bondgenerator
from utils.fg_extractor import create_hetero_graph as fggenerator
from utils.atom_extractor import atom, load_smiles
from utils.others import Variablefc, hypergraphSyn
from heteromp import create_bond_hetero_gat, create_fg_hetero_gat

device = device('cuda' if cuda.is_available() else 'cpu')

class ECHO(nn.Module):
    def __init__(self, atom_in_channels, fg_in_channels, bond_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        # fg_channels = 167, hidden_channels = 128, bond_channels = 4
        self.atomgenerator = atom()
        self.dropout = nn.Dropout(dropout)
        self.heads = heads

        self.fpgenerator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=False
        )
        
        self.fgag_atten = create_fg_hetero_gat(
            atom_in_channels=atom_in_channels,
            fg_in_channels=fg_in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads
        )
        self.bgag_atten = create_bond_hetero_gat(
            atom_in_channels=atom_in_channels,
            bond_in_channels=bond_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=heads
        )
        
        self.readout_dim = hidden_channels * heads
        self.graph_projection = nn.Linear(self.readout_dim, 2048)
        self.fusion_gate = nn.Linear(2048 + 2048, 1)
        
        self.final_fc = Variablefc(
            input_size=2048,
            hidden_sizes=[1024, 512],  # 2048 -> 1024 -> 512 -> out_channels
            output_size=out_channels,
            activation=nn.ReLU()
        )
        
    def hypergraph_readout(self, fgag_representations, bgag_representations):
        """
        使用hypergraphSyn实现两种异质图的融合读出
        """
        readout_vectors = []
        
        # 1. 从官能团图提取表示
        if 'atom' in fgag_representations and fgag_representations['atom'].numel() > 0:
            # 原子节点的全局平均池化
            atom_fgag = torch.mean(fgag_representations['atom'], dim=0)
            readout_vectors.append(atom_fgag)
        
        if 'fg' in fgag_representations and fgag_representations['fg'].numel() > 0:
            # 官能团节点的全局平均池化
            fg_repr = torch.mean(fgag_representations['fg'], dim=0)
            readout_vectors.append(fg_repr)
        
        # 2. 从键角图提取表示
        if 'atom' in bgag_representations and bgag_representations['atom'].numel() > 0:
            # 原子节点的全局平均池化（来自键角图）
            atom_bgag = torch.mean(bgag_representations['atom'], dim=0)
            readout_vectors.append(atom_bgag)
        
        if 'bond' in bgag_representations and bgag_representations['bond'].numel() > 0:
            # 键节点的全局平均池化
            bond_repr = torch.mean(bgag_representations['bond'], dim=0)
            readout_vectors.append(bond_repr)
        
        # 3. 使用hypergraphSyn融合所有表示
        if len(readout_vectors) > 1:
            fused_representation = hypergraphSyn(readout_vectors)
            # 确保输出维度正确
            if fused_representation.dim() > 1:
                fused_representation = fused_representation.squeeze()
            return fused_representation
        elif len(readout_vectors) == 1:
            return readout_vectors[0]
        else:
            # 修复设备一致性问题
            return torch.zeros(self.readout_dim, device=device)

    def morgan_residual_connection(self, fused_representation, morgan_fp):
        """
        将图表示投影到Morgan指纹维度，然后使用门控机制融合
        """
        # 将图表示投影到Morgan指纹的维度（2048）
        graph_projected = self.graph_projection(fused_representation)
        
        # 计算门控权重
        gate_input = torch.cat([graph_projected, morgan_fp], dim=0)
        gate_weight = torch.sigmoid(self.fusion_gate(gate_input))
        
        # 门控融合：保持Morgan指纹不变，只调整融合比例
        enhanced_representation = gate_weight * graph_projected + (1 - gate_weight) * morgan_fp
        
        return enhanced_representation
        
    def forward(self, smiles_batch):
        """支持批处理的forward方法"""
        if isinstance(smiles_batch, str):
            # 单个样本处理（保持向后兼容）
            return self._forward_single(smiles_batch)
        
        # 批处理
        batch_outputs = []
        
        # 预处理所有分子（可以并行化）
        mols = []
        morgan_fps = []
        atom_graphs = []
        hetero_fgags = []
        hetero_bgags = []
        
        for smiles in smiles_batch:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            mols.append(mol)
            
            # 预计算所有特征
            morgan_fp_bits = self.fpgenerator.GetFingerprint(mol)
            morgan_fp = torch.tensor(list(morgan_fp_bits), dtype=torch.float32, device=device)
            morgan_fps.append(morgan_fp)
            
            atom_graph = self.atomgenerator(mol, device=device)
            atom_graphs.append(atom_graph)
            
            hetero_fgag = fggenerator(mol, atom_graph)
            hetero_fgags.append(hetero_fgag)
            
            hetero_bgag = bondgenerator(mol, atom_graph)
            hetero_bgags.append(hetero_bgag)
        
        # 批量处理图神经网络部分
        for i in range(len(smiles_batch)):
            # fgag-conv
            fgag_output, fgag_representations = self.fgag_atten(hetero_fgags[i])
            # bgag-conv
            bgag_output, bgag_representations = self.bgag_atten(hetero_bgags[i])
            
            # 使用hypergraph读出融合两种图的表示
            fused_representation = self.hypergraph_readout(
                fgag_representations, 
                bgag_representations
            )
            
            # 应用dropout
            fused_representation = self.dropout(fused_representation)
            
            # 引入Morgan指纹的残差连接
            enhanced_representation = self.morgan_residual_connection(
                fused_representation, 
                morgan_fps[i]
            )
            
            # 最终预测
            output = self.final_fc(enhanced_representation)
            batch_outputs.append(output)
        
        return torch.stack(batch_outputs)
    
    def _forward_single(self, smiles: str):
        """单样本处理方法"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # 生成Morgan指纹
        morgan_fp_bits = self.fpgenerator.GetFingerprint(mol)
        morgan_fp = torch.tensor(list(morgan_fp_bits), dtype=torch.float32, device=device)
        
        atom_graph = self.atomgenerator(mol, device=device)
        hetero_fgag = fggenerator(mol, atom_graph)
        hetero_bgag = bondgenerator(mol, atom_graph)

        # 第一层注意力 + 残差连接
        fgag_output_1, fgag_representations_1 = self.fgag_atten(hetero_fgag)
        bgag_output_1, bgag_representations_1 = self.bgag_atten(hetero_bgag)
        
        # 保存第一层的输入作为残差连接的基础
        fgag_residual = {k: v.clone() for k, v in hetero_fgag.x_dict.items()}
        bgag_residual = {k: v.clone() for k, v in hetero_bgag.x_dict.items()}
        
        # 更新异质图的节点特征为第一层的输出
        hetero_fgag_updated = hetero_fgag.clone()
        hetero_bgag_updated = hetero_bgag.clone()
        
        # 将第一层的输出作为第二层的输入（需要维度匹配处理）
        for node_type in fgag_representations_1:
            if node_type in hetero_fgag_updated.x_dict:
                # 如果维度不匹配，需要投影到原始维度
                if fgag_representations_1[node_type].shape[-1] != hetero_fgag_updated.x_dict[node_type].shape[-1]:
                    # 使用线性层投影回原始维度（需要在__init__中定义这些投影层）
                    hetero_fgag_updated.x_dict[node_type] = self.fgag_dim_proj[node_type](fgag_representations_1[node_type])
                else:
                    hetero_fgag_updated.x_dict[node_type] = fgag_representations_1[node_type]
        
        for node_type in bgag_representations_1:
            if node_type in hetero_bgag_updated.x_dict:
                if bgag_representations_1[node_type].shape[-1] != hetero_bgag_updated.x_dict[node_type].shape[-1]:
                    hetero_bgag_updated.x_dict[node_type] = self.bgag_dim_proj[node_type](bgag_representations_1[node_type])
                else:
                    hetero_bgag_updated.x_dict[node_type] = bgag_representations_1[node_type]
        
        # 第二层注意力
        fgag_output_2, fgag_representations_2 = self.fgag_atten(hetero_fgag_updated)
        bgag_output_2, bgag_representations_2 = self.bgag_atten(hetero_bgag_updated)
        
        # 残差连接：第二层输出 + 第一层输入
        fgag_representations_final = {}
        bgag_representations_final = {}
        
        for node_type in fgag_representations_2:
            if node_type in fgag_residual:
                # 残差连接需要维度匹配
                if fgag_representations_2[node_type].shape[-1] == fgag_residual[node_type].shape[-1]:
                    fgag_representations_final[node_type] = fgag_representations_2[node_type] + fgag_residual[node_type]
                else:
                    # 如果维度不匹配，只使用第二层输出
                    fgag_representations_final[node_type] = fgag_representations_2[node_type]
            else:
                fgag_representations_final[node_type] = fgag_representations_2[node_type]
        
        for node_type in bgag_representations_2:
            if node_type in bgag_residual:
                if bgag_representations_2[node_type].shape[-1] == bgag_residual[node_type].shape[-1]:
                    bgag_representations_final[node_type] = bgag_representations_2[node_type] + bgag_residual[node_type]
                else:
                    bgag_representations_final[node_type] = bgag_representations_2[node_type]
            else:
                bgag_representations_final[node_type] = bgag_representations_2[node_type]

        # 使用hypergraph读出融合两种图的表示
        fused_representation = self.hypergraph_readout(
            fgag_representations_final, 
            bgag_representations_final
        )
        
        # 应用dropout
        fused_representation = self.dropout(fused_representation)
        
        # 引入Morgan指纹的残差连接
        enhanced_representation = self.morgan_residual_connection(
            fused_representation, 
            morgan_fp
        )
        
        # 最终预测
        output = self.final_fc(enhanced_representation)
        
        return output
        
        # 添加维度投影层用于层间连接
        self.fgag_dim_proj = nn.ModuleDict({
            'atom': nn.Linear(hidden_channels * heads, atom_in_channels),
            'fg': nn.Linear(hidden_channels * heads, fg_in_channels)
        })
        
        self.bgag_dim_proj = nn.ModuleDict({
            'atom': nn.Linear(hidden_channels * heads, atom_in_channels),
            'bond': nn.Linear(hidden_channels * heads, bond_channels)
        })

        

        

        






        




