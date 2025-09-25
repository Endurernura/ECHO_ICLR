import torch
import os
import pandas as pd
from torch_geometric.data import Data
from rdkit import Chem
from utils.access import atom_list

def load_smiles(n):
    file_path = os.path.join('data', 'hiv_cleaned.csv')
    df = pd.read_csv(file_path)

    smiles = df.iloc[n, 0]
    label = df.iloc[n, 1]

    return smiles, label

class atom(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def one_of_k_encoding(self, x, allowable_set):
        """
        对输入x进行one-of-k编码，如果x不在允许集合中则抛出异常。
        """
        if x not in allowable_set:
            raise Exception("输入 {0} 不在允许集合 {1} 中:".format(x, allowable_set))
        return torch.tensor(list(map(lambda s: x == s, allowable_set)), dtype=torch.float)

    def one_of_k_encoding_unk(self, x, allowable_set):
        """
        对输入x进行one-of-k编码，如果x不在允许集合中则将其视为allowable_set的最后一个元素（通常是'Unknown'）。
        """
        if x not in allowable_set:
            x = allowable_set[-1]
        return torch.tensor(list(map(lambda s: x == s, allowable_set)), dtype=torch.float)

    def atom_features(self, atom):
        """
        根据RDKit原子对象生成原子特征向量。
        增加了电负性（electronegativity）和杂化方式（hybridization）的编码。
        """
        # 元素符号编码
        symbol_encoding = self.one_of_k_encoding_unk(atom.GetSymbol(),
                                        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                        'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                        'Pt', 'Hg', 'Pb', 'Unknown'])
        # 原子度编码
        degree_encoding = self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # 总氢原子数编码
        total_hs_encoding = self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # 隐式价态编码
        implicit_valence_encoding = self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # 是否芳香性编码
        is_aromatic = torch.tensor([atom.GetIsAromatic()], dtype=torch.float)

        # **新增杂化方式编码**
        # RDKit的HybridizationType枚举提供了常见的杂化类型
        hybridization_encoding = self.one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.HybridizationType.S,
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
            Chem.HybridizationType.UNSPECIFIED # 处理未知或未指定的杂化类型
        ])


        electronegativity_value = 0.0
        symbol = atom.GetSymbol()
        electronegativity_map = atom_list
        electronegativity_value = electronegativity_map.get(symbol, electronegativity_map['Unknown'])

        # 将电负性值进行分箱并进行one-hot编码（示例分箱）
        # 这里将电负性值四舍五入到最近的0.5，然后进行one-hot编码。
        electronegativity_encoding = self.one_of_k_encoding_unk(round(electronegativity_value * 2) / 2, 
                                                    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 'Unknown_EN']) # 示例分箱
        
        # 将所有特征拼接起来
        return torch.cat((symbol_encoding, degree_encoding, total_hs_encoding, 
                        implicit_valence_encoding, is_aromatic, 
                        hybridization_encoding, electronegativity_encoding))
    #特征包括：原子是什么，它的度（相连的原子数量），周围有几个氢，化合价，是否有芳香性，电负性，杂化方式。


    def bond_features(self, bond):
        """
        Builds a feature vector for a bond.
        """
        bond_type = bond.GetBondType()
        return torch.tensor([
            bond_type == Chem.BondType.SINGLE,
            bond_type == Chem.BondType.DOUBLE,
            bond_type == Chem.BondType.TRIPLE,
            bond_type == Chem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ], dtype=torch.float)


    def forward(self, mol:Chem.Mol, device=None):
        # 如果没有指定设备，使用默认设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        num_nodes = mol.GetNumAtoms()
        node_features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            node_features.append(feature)
        node_features = torch.stack(node_features).to(device)
        edge_index = []
        edge_features = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            #添加双向边
            edge_index.extend([[start, end], [end, start]])
            # 添加双向边的特征
            feats = self.bond_features(bond)
            edge_features.extend([feats, feats])

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_features = torch.stack(edge_features).to(device)
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, mol=mol)
        return graph

# class functional_2(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def get_maccs_fingerprint(self, mol):
#         """
#         计算分子的MACCS指纹
#         """
#         from rdkit.Chem import MACCSkeys
#         from rdkit import Chem
#         try:
#             if mol is not None:
#                 # 正确初始化环信息的方法
#                 Chem.GetSSSR(mol)
#                 # 或者使用
#                 # Chem.FastFindRings(mol)
#                 return MACCSkeys.GenMACCSKeys(mol)
#             return [0] * 166  # 返回全零指纹
#         except Exception as e:
#             print(f"计算MACCS指纹时出错: {e}, 官能团: {getattr(mol, 'name', 'unknown')}")
#             return [0] * 166  # MACCS指纹长度为166
    
#     def forward(self, atom_graph, mol):
#         """
#         构建官能团-原子混合图，其中官能团节点与其包含的原子节点之间有边
        
#         参数:
#             atom_graph: 原子图 (PyG Data对象)
#             mol: RDKit分子对象
            
#         返回:
#             混合图 (PyG Data对象)
#         """
#         from rdkit import Chem
#         from rdkit.Chem import AllChem
#         from access import FUNCTIONAL_GROUPS
#         import numpy as np
#         from torch_geometric.data import Data, Batch
        
#         if mol is None or atom_graph is None:
#             return None
            
#         # 获取原子图的节点特征和边信息
#         atom_features = atom_graph.x
#         atom_edge_index = atom_graph.edge_index
#         atom_edge_attr = atom_graph.edge_attr
#         num_atoms = atom_features.size(0)
        
#         # 初始化官能团节点列表和映射
#         fg_nodes = []
#         fg_to_atoms = {}
#         fg_idx = 0
        
#         # 遍历所有官能团类型
#         for fg_name, smarts in FUNCTIONAL_GROUPS.items():
#             try:
#                 # 查找分子中的官能团
#                 pattern = Chem.MolFromSmarts(smarts)
#                 if pattern is None:
#                     print(f"无法解析SMARTS模式: {smarts} 对应官能团: {fg_name}")
#                     continue
                    
#                 matches = mol.GetSubstructMatches(pattern)
                
#                 # 为每个匹配的官能团创建节点
#                 for match in matches:
#                     # 记录官能团中的原子
#                     fg_to_atoms[fg_idx] = list(match)
                    
#                     # 创建官能团子结构 - 使用 MolFragmentToSmiles 替代 PathToSubmol
#                     try:
#                         # 获取匹配原子之间的键
#                         bonds = []
#                         atom_set = set(match)
#                         for bond in mol.GetBonds():
#                             begin_idx = bond.GetBeginAtomIdx()
#                             end_idx = bond.GetEndAtomIdx()
#                             if begin_idx in atom_set and end_idx in atom_set:
#                                 bonds.append(bond.GetIdx())
                        
#                         # 创建子结构
#                         fg_mol = Chem.RWMol()
#                         atom_map = {}
                        
#                         # 添加原子
#                         for atom_idx in match:
#                             atom = mol.GetAtomWithIdx(atom_idx)
#                             new_atom = Chem.Atom(atom.GetAtomicNum())
#                             new_atom.SetFormalCharge(atom.GetFormalCharge())
#                             new_atom.SetIsAromatic(atom.GetIsAromatic())
#                             new_atom.SetChiralTag(atom.GetChiralTag())
#                             new_atom.SetHybridization(atom.GetHybridization())
#                             new_idx = fg_mol.AddAtom(new_atom)
#                             atom_map[atom_idx] = new_idx
                        
#                         # 添加键
#                         for bond_idx in bonds:
#                             bond = mol.GetBondWithIdx(bond_idx)
#                             begin_idx = bond.GetBeginAtomIdx()
#                             end_idx = bond.GetEndAtomIdx()
#                             fg_mol.AddBond(atom_map[begin_idx], atom_map[end_idx], bond.GetBondType())
                        
#                         # 转换为普通分子对象
#                         fg_mol = fg_mol.GetMol()
                        
#                         # 计算MACCS指纹作为节点特征
#                         maccs_fp = self.get_maccs_fingerprint(fg_mol)
#                         fg_nodes.append({
#                             'idx': fg_idx,
#                             'type': fg_name,
#                             'features': maccs_fp,
#                             'atoms': list(match)
#                         })
#                         fg_idx += 1
#                     except Exception as e:
#                         print(f"处理官能团时出错: {e}, 官能团: {fg_name}")
#                         continue
                        
#             except Exception as e:
#                 print(f"处理官能团 {fg_name} 时出错: {e}")
#                 continue
        
#         # 如果没有找到官能团，返回原始原子图
#         if len(fg_nodes) == 0:
#             print("未找到官能团，返回原始原子图")
#             return atom_graph
            
#         # 构建官能团节点特征矩阵
#         fg_features = []
#         for node in fg_nodes:
#             # 将MACCS指纹转换为PyTorch张量
#             fp_tensor = torch.tensor(list(node['features']), dtype=torch.float)
#             fg_features.append(fp_tensor)
        
#         fg_features = torch.stack(fg_features)
#         num_fgs = fg_features.size(0)
        
#         # 创建节点类型标记：0表示原子，1表示官能团
#         node_types = torch.cat([
#             torch.zeros(num_atoms, dtype=torch.long),  # 原子节点
#             torch.ones(num_fgs, dtype=torch.long)    # 官能团节点
#         ])
        
#         # 合并原子和官能团特征
#         # 注意：由于特征维度可能不同，我们需要进行维度调整
#         atom_feat_dim = atom_features.size(1)
#         fg_feat_dim = fg_features.size(1)
        
#         # 方法1：填充较小的特征向量以匹配较大的维度
#         if atom_feat_dim > fg_feat_dim:
#             # 填充官能团特征
#             padding = torch.zeros(num_fgs, atom_feat_dim - fg_feat_dim, dtype=torch.float)
#             fg_features_padded = torch.cat([fg_features, padding], dim=1)
#             all_features = torch.cat([atom_features, fg_features_padded], dim=0)
#         elif fg_feat_dim > atom_feat_dim:
#             # 填充原子特征
#             padding = torch.zeros(num_atoms, fg_feat_dim - atom_feat_dim, dtype=torch.float)
#             atom_features_padded = torch.cat([atom_features, padding], dim=1)
#             all_features = torch.cat([atom_features_padded, fg_features], dim=0)
#         else:
#             # 维度相同，直接拼接
#             all_features = torch.cat([atom_features, fg_features], dim=0)
        
#         # 构建官能团到原子的边
#         fg_to_atom_edges = []
#         edge_types = []
        
#         # 原子-原子边的类型为0
#         atom_edge_types = torch.zeros(atom_edge_index.size(1), dtype=torch.long)
        
#         # 为每个官能团创建与其包含的原子之间的边
#         for fg_id, atom_list in fg_to_atoms.items():
#             for atom_id in atom_list:
#                 # 官能团 -> 原子 (类型1)
#                 fg_to_atom_edges.append([fg_id + num_atoms, atom_id])
#                 edge_types.append(1)
                
#                 # 原子 -> 官能团 (类型2)
#                 fg_to_atom_edges.append([atom_id, fg_id + num_atoms])
#                 edge_types.append(2)
        
#         # 合并所有边
#         fg_to_atom_edges = torch.tensor(fg_to_atom_edges, dtype=torch.long).t() if fg_to_atom_edges else torch.zeros((2, 0), dtype=torch.long)
#         all_edge_index = torch.cat([atom_edge_index, fg_to_atom_edges], dim=1)
        
#         # 合并边类型
#         all_edge_types = torch.cat([atom_edge_types, torch.tensor(edge_types, dtype=torch.long)])
        
#         # 处理边特征
#         if atom_edge_attr is not None:
#             # 为官能团-原子边创建边特征
#             # 这里我们使用简单的全1特征，也可以根据需要设计更复杂的特征
#             fg_atom_edge_dim = atom_edge_attr.size(1)
#             fg_atom_edge_attr = torch.ones(len(edge_types), fg_atom_edge_dim, dtype=torch.float)
            
#             # 合并所有边特征
#             all_edge_attr = torch.cat([atom_edge_attr, fg_atom_edge_attr], dim=0)
#         else:
#             all_edge_attr = None
        
#         # 创建混合图
#         hybrid_graph = Data(
#             x=all_features,
#             edge_index=all_edge_index,
#             edge_attr=all_edge_attr,
#             edge_type=all_edge_types,
#             node_type=node_types,
#             num_atoms=num_atoms,
#             num_fgs=num_fgs,
#             fg_to_atoms=fg_to_atoms,
#             mol=mol
#         )
        
#         return hybrid_graph

if __name__ == '__main__':
    from tqdm import tqdm
    error_count = 0
    for i in tqdm(range(1000)):
        try:
            smiles, label = load_smiles(i)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无法解析SMILES字符串，索引: {i}, SMILES: {smiles}")
                error_count += 1
                continue
                
            atom_model = atom()
            graph = atom_model(mol)
            # functional = functional_2()
            # fg = functional(graph, mol)
        except Exception as e:
            print(f"处理索引 {i} 时出错: {e}")
            error_count += 1
    
    print(f"处理完成。总共有 {error_count} 个错误。")
