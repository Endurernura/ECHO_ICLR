import torch
from torch_geometric.data import Data, HeteroData
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem  # 添加AllChem导入
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.access import FUNCTIONAL_GROUPS

class FunctionalGroupExtractor:
    def __init__(self):
        self.fg_smarts = FUNCTIONAL_GROUPS
        
    def extract_functional_groups(self, mol):
        """提取分子中的官能团"""
        fg_matches = {}
        atom_to_fg = {}
        
        # 为每个原子初始化空列表
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_to_fg[atom_idx] = []
        
        # 查找每种官能团
        for fg_name, smarts in self.fg_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    fg_matches[fg_name] = matches
                    # 记录每个原子属于哪些官能团
                    for match in matches:
                        for atom_idx in match:
                            atom_to_fg[atom_idx].append(fg_name)
        
        return fg_matches, atom_to_fg
    
    def generate_maccs_fingerprints(self, mol, fg_matches):
        """为每个官能团生成MACCS指纹"""
        # 获取整个分子的MACCS指纹
        mol_maccs = MACCSkeys.GenMACCSKeys(mol)
        mol_maccs_array = np.array(mol_maccs)
        
        # 为每个官能团创建子分子并生成指纹
        fg_fingerprints = {}
        for fg_name, matches in fg_matches.items():
            if matches:
                # 使用第一个匹配作为代表
                match = matches[0]
                # 创建只包含官能团原子的子分子
                fg_mol = Chem.RWMol()
                atom_mapping = {}
                
                # 添加原子并保留属性
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    new_atom = Chem.Atom(atom.GetSymbol())
                    # 复制重要属性
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_atom.SetHybridization(atom.GetHybridization())
                    # 只为环中的原子设置芳香性
                    if atom.IsInRing():
                        new_atom.SetIsAromatic(atom.GetIsAromatic())
                    else:
                        new_atom.SetIsAromatic(False)
                    new_idx = fg_mol.AddAtom(new_atom)
                    atom_mapping[atom_idx] = new_idx
                
                # 添加键 - 使用集合跟踪已添加的键
                added_bonds = set()
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for bond in atom.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        if begin_idx in atom_mapping and end_idx in atom_mapping:
                            # 创建一个唯一的键标识符，确保不重复添加
                            bond_id = tuple(sorted([atom_mapping[begin_idx], atom_mapping[end_idx]]))
                            if bond_id not in added_bonds:
                                added_bonds.add(bond_id)
                                fg_mol.AddBond(atom_mapping[begin_idx], 
                                            atom_mapping[end_idx], 
                                            bond.GetBondType())
                
                try:
                    # 转换为不可变分子
                    fg_mol = fg_mol.GetMol()
                    
                    # 计算隐式氢和标准化分子，但不重新计算芳香性
                    Chem.SanitizeMol(fg_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^
                                    Chem.SanitizeFlags.SANITIZE_KEKULIZE^
                                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                    
                    # 计算2D坐标 - 这会隐式初始化环信息
                    AllChem.Compute2DCoords(fg_mol)
                    
                    # 生成MACCS指纹
                    fg_maccs = MACCSkeys.GenMACCSKeys(fg_mol)
                    fg_fingerprints[fg_name] = np.array(fg_maccs)
                except Exception as e:
                    print(f"生成{fg_name}的指纹时出错: {str(e)}")
                    fg_fingerprints[fg_name] = mol_maccs_array
        
        return fg_fingerprints


def create_hetero_graph(mol, atom_data: Data):
    """创建包含原子节点和官能团节点的异构图"""
    # 获取atom_data的设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 提取官能团
    extractor = FunctionalGroupExtractor()
    fg_matches, atom_to_fg = extractor.extract_functional_groups(mol)
    fg_fingerprints = extractor.generate_maccs_fingerprints(mol, fg_matches)
    
    # 创建异构图
    hetero_data = HeteroData()
    
    # 添加原子节点和边（确保在正确设备上）
    hetero_data['atom'].x = atom_data.x.to(device)
    hetero_data['atom', 'bonds', 'atom'].edge_index = atom_data.edge_index.to(device)
    
    # 添加官能团节点
    fg_names = list(fg_fingerprints.keys())
    fg_features = [torch.tensor(fg_fingerprints[name], dtype=torch.float, device=device) for name in fg_names]
    
    if fg_features:  # 确保有官能团
        hetero_data['fg'].x = torch.stack(fg_features)
        
        # 创建官能团到原子的连接
        fg_to_atom_src = []
        fg_to_atom_dst = []
        
        # 创建一个字典，记录每个官能团包含的原子
        fg_to_atoms = {fg_idx: set() for fg_idx in range(len(fg_names))}
        
        for atom_idx, fg_list in atom_to_fg.items():
            for fg_name in fg_list:
                if fg_name in fg_names:  # 确保官能团在我们的列表中
                    fg_idx = fg_names.index(fg_name)
                    fg_to_atom_src.append(fg_idx)
                    fg_to_atom_dst.append(atom_idx)
                    fg_to_atoms[fg_idx].add(atom_idx)  # 记录该官能团包含的原子
        
        if fg_to_atom_src:  # 确保有连接
            hetero_data['fg', 'contains', 'atom'].edge_index = torch.tensor(
                [fg_to_atom_src, fg_to_atom_dst], dtype=torch.long, device=device
            )
            
            # 添加官能团之间的连边（当两个官能团通过化学键相连时）
            fg_to_fg_src = []
            fg_to_fg_dst = []
            
            # 获取分子中的所有键
            bonds = []
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bonds.append((begin_idx, end_idx))
            
            # 检查每对官能团是否通过化学键相连
            for i in range(len(fg_names)):
                for j in range(i+1, len(fg_names)):  # 避免重复检查和自环
                    # 获取两个官能团的原子集合
                    atoms_i = fg_to_atoms[i]
                    atoms_j = fg_to_atoms[j]
                    
                    # 检查是否有连接两个官能团的化学键
                    connected = False
                    for begin_idx, end_idx in bonds:
                        # 如果一个键的一端在官能团i中，另一端在官能团j中，则这两个官能团通过化学键相连
                        if (begin_idx in atoms_i and end_idx in atoms_j) or \
                           (begin_idx in atoms_j and end_idx in atoms_i):
                            connected = True
                            break
                    
                    if connected:
                        # 添加双向边
                        fg_to_fg_src.extend([i, j])
                        fg_to_fg_dst.extend([j, i])
            
            if fg_to_fg_src:  # 确保有官能团之间的连接
                hetero_data['fg', 'bonded', 'fg'].edge_index = torch.tensor(
                    [fg_to_fg_src, fg_to_fg_dst], dtype=torch.long, device=device
                )
    
    return hetero_data