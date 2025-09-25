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
        readout_vectors = []
        
        if 'atom' in fgag_representations and fgag_representations['atom'].numel() > 0:
            atom_fgag = torch.mean(fgag_representations['atom'], dim=0)
            readout_vectors.append(atom_fgag)
        
        if 'fg' in fgag_representations and fgag_representations['fg'].numel() > 0:
            fg_repr = torch.mean(fgag_representations['fg'], dim=0)
            readout_vectors.append(fg_repr)
        
        if 'atom' in bgag_representations and bgag_representations['atom'].numel() > 0:
            atom_bgag = torch.mean(bgag_representations['atom'], dim=0)
            readout_vectors.append(atom_bgag)
        
        if 'bond' in bgag_representations and bgag_representations['bond'].numel() > 0:
            bond_repr = torch.mean(bgag_representations['bond'], dim=0)
            readout_vectors.append(bond_repr)
        
        if len(readout_vectors) > 1:
            fused_representation = hypergraphSyn(readout_vectors)
            if fused_representation.dim() > 1:
                fused_representation = fused_representation.squeeze()
            return fused_representation
        elif len(readout_vectors) == 1:
            return readout_vectors[0]
        else:
            return torch.zeros(self.readout_dim, device=device)

    def morgan_residual_connection(self, fused_representation, morgan_fp):
        graph_projected = self.graph_projection(fused_representation)
        gate_input = torch.cat([graph_projected, morgan_fp], dim=0)
        gate_weight = torch.sigmoid(self.fusion_gate(gate_input))
        
        enhanced_representation = gate_weight * graph_projected + (1 - gate_weight) * morgan_fp
        
        return enhanced_representation
        
    def forward(self, smiles_batch):
        """支持批处理的forward方法"""
        if isinstance(smiles_batch, str):
            return self._forward_single(smiles_batch)
        
        batch_outputs = []
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
            
            morgan_fp_bits = self.fpgenerator.GetFingerprint(mol)
            morgan_fp = torch.tensor(list(morgan_fp_bits), dtype=torch.float32, device=device)
            morgan_fps.append(morgan_fp)
            
            atom_graph = self.atomgenerator(mol, device=device)
            atom_graphs.append(atom_graph)
            
            hetero_fgag = fggenerator(mol, atom_graph)
            hetero_fgags.append(hetero_fgag)
            
            hetero_bgag = bondgenerator(mol, atom_graph)
            hetero_bgags.append(hetero_bgag)
        
        for i in range(len(smiles_batch)):
            # fgag-conv
            fgag_output, fgag_representations = self.fgag_atten(hetero_fgags[i])
            # bgag-conv
            bgag_output, bgag_representations = self.bgag_atten(hetero_bgags[i])
            
            fused_representation = self.hypergraph_readout(
                fgag_representations, 
                bgag_representations
            )
            
            fused_representation = self.dropout(fused_representation)
            enhanced_representation = self.morgan_residual_connection(
                fused_representation, 
                morgan_fps[i]
            )
            output = self.final_fc(enhanced_representation)
            batch_outputs.append(output)
        
        return torch.stack(batch_outputs)
    
    def _forward_single(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        morgan_fp_bits = self.fpgenerator.GetFingerprint(mol)
        morgan_fp = torch.tensor(list(morgan_fp_bits), dtype=torch.float32, device=device)
        
        atom_graph = self.atomgenerator(mol, device=device)
        hetero_fgag = fggenerator(mol, atom_graph)
        hetero_bgag = bondgenerator(mol, atom_graph)
        
        fgag_output_1, fgag_representations_1 = self.fgag_atten(hetero_fgag)
        bgag_output_1, bgag_representations_1 = self.bgag_atten(hetero_bgag)
        
        fgag_residual = {k: v.clone() for k, v in hetero_fgag.x_dict.items()}
        bgag_residual = {k: v.clone() for k, v in hetero_bgag.x_dict.items()}
        
        hetero_fgag_updated = hetero_fgag.clone()
        hetero_bgag_updated = hetero_bgag.clone()
        
        for node_type in fgag_representations_1:
            if node_type in hetero_fgag_updated.x_dict:
                if fgag_representations_1[node_type].shape[-1] != hetero_fgag_updated.x_dict[node_type].shape[-1]:
                    hetero_fgag_updated.x_dict[node_type] = self.fgag_dim_proj[node_type](fgag_representations_1[node_type])
                else:
                    hetero_fgag_updated.x_dict[node_type] = fgag_representations_1[node_type]
        
        for node_type in bgag_representations_1:
            if node_type in hetero_bgag_updated.x_dict:
                if bgag_representations_1[node_type].shape[-1] != hetero_bgag_updated.x_dict[node_type].shape[-1]:
                    hetero_bgag_updated.x_dict[node_type] = self.bgag_dim_proj[node_type](bgag_representations_1[node_type])
                else:
                    hetero_bgag_updated.x_dict[node_type] = bgag_representations_1[node_type]
        
        fgag_output_2, fgag_representations_2 = self.fgag_atten(hetero_fgag_updated)
        bgag_output_2, bgag_representations_2 = self.bgag_atten(hetero_bgag_updated)
        
        fgag_representations_final = {}
        bgag_representations_final = {}
        
        for node_type in fgag_representations_2:
            if node_type in fgag_residual:
                if fgag_representations_2[node_type].shape[-1] == fgag_residual[node_type].shape[-1]:
                    fgag_representations_final[node_type] = fgag_representations_2[node_type] + fgag_residual[node_type]
                else:
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

        fused_representation = self.hypergraph_readout(
            fgag_representations_final, 
            bgag_representations_final
        )
        fused_representation = self.dropout(fused_representation)
        enhanced_representation = self.morgan_residual_connection(
            fused_representation, 
            morgan_fp
        )
        output = self.final_fc(enhanced_representation)
        
        return output


        

        

        






        




