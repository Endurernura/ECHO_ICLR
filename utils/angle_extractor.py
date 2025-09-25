import torch
from torch_geometric.data import Data, HeteroData
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import math
import threading
import time
import concurrent.futures

class BondExtractor:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="BondOpt"
        )
        
    def calculate_angle(self, pos1, pos2, pos3):
        v1 = pos1 - pos2
        v2 = pos3 - pos2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def _optimize_with_timeout(self, mol_with_hs, timeout=1.0):
        try:
            future = self.executor.submit(AllChem.MMFFOptimizeMolecule, mol_with_hs)
            result = future.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutError:
            future.cancel()
            return False
        except Exception:
            try:
                future = self.executor.submit(AllChem.UFFOptimizeMolecule, mol_with_hs)
                result = future.result(timeout=timeout)
                return True
            except (concurrent.futures.TimeoutError, Exception):
                future.cancel()
                return False

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def _calculate_simple_bond_angles(self, mol):
        bond_angles = {}
        bonds = []
        
        for bond in mol.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            bonds.append(tuple(sorted((start_atom, end_atom))))
        
        atom_to_bonds = {i: [] for i in range(mol.GetNumAtoms())}
        for bond_idx, bond in enumerate(mol.GetBonds()):
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            atom_to_bonds[start_atom].append(bond_idx)
            atom_to_bonds[end_atom].append(bond_idx)
        
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            connected_bonds_indices = atom_to_bonds[i]
            
            if len(connected_bonds_indices) < 2:
                continue
            
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP3:
                ideal_angle = 109.47
            elif hybridization == Chem.HybridizationType.SP2:
                ideal_angle = 120.0
            elif hybridization == Chem.HybridizationType.SP:
                ideal_angle = 180.0
            else:
                num_connections = len(connected_bonds_indices)
                if num_connections == 2:
                    ideal_angle = 180.0
                elif num_connections == 3:
                    ideal_angle = 120.0
                else:
                    ideal_angle = 109.47
            
            for j in range(len(connected_bonds_indices)):
                for k in range(j + 1, len(connected_bonds_indices)):
                    bond1_idx = connected_bonds_indices[j]
                    bond2_idx = connected_bonds_indices[k]
                    
                    bond_tuple = tuple(sorted((bond1_idx, bond2_idx)))

                    if bond_tuple not in bond_angles:
                        variation = np.random.normal(0, 5)
                        angle = max(0, min(180, ideal_angle + variation))

                        bond_angles[bond_tuple] = angle
        
        return bond_angles, bonds

    def get_bond_angles(self, mol):
        if mol.GetNumAtoms() > 100:
            return self._calculate_simple_bond_angles(mol)
            
        try:
            mol_with_hs = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDG()) != 0:
                return self._calculate_simple_bond_angles(mol)
            optimization_success = self._optimize_with_timeout(mol_with_hs, timeout=1.0)
            
            if not optimization_success:
                return self._calculate_simple_bond_angles(mol)
            
            conf = mol_with_hs.GetConformer()
            num_atoms = mol.GetNumAtoms()
            bond_angles = {}
            
            atom_to_bonds = {i: [] for i in range(num_atoms)}
            bonds = []
            for bond_idx, bond in enumerate(mol.GetBonds()):
                start_atom = bond.GetBeginAtomIdx()
                end_atom = bond.GetEndAtomIdx()
                bonds.append(tuple(sorted((start_atom, end_atom))))
                atom_to_bonds[start_atom].append(bond_idx)
                atom_to_bonds[end_atom].append(bond_idx)

            for i in range(num_atoms):
                connected_bonds_indices = atom_to_bonds[i]
                
                for j in range(len(connected_bonds_indices)):
                    for k in range(j + 1, len(connected_bonds_indices)):
                        bond1_idx = connected_bonds_indices[j]
                        bond2_idx = connected_bonds_indices[k]
                        
                        bond1 = mol.GetBondWithIdx(bond1_idx)
                        bond2 = mol.GetBondWithIdx(bond2_idx)
                        
                        atom1 = bond1.GetOtherAtomIdx(i)
                        atom3 = bond2.GetOtherAtomIdx(i)
                        
                        pos1 = np.array(conf.GetAtomPosition(atom1))
                        pos2 = np.array(conf.GetAtomPosition(i))
                        pos3 = np.array(conf.GetAtomPosition(atom3))
                        
                        angle = self.calculate_angle(pos1, pos2, pos3)
                        
                        bond_tuple = tuple(sorted((bond1_idx, bond2_idx)))
                        if bond_tuple not in bond_angles:
                            bond_angles[bond_tuple] = angle
                            
            return bond_angles, bonds
            
        except Exception as e:
            return self._calculate_simple_bond_angles(mol)


def create_bond_graph(mol, atom_data: Data):
    device = atom_data.x.device
    extractor = BondExtractor()
    bond_angles, bonds = extractor.get_bond_angles(mol)
    
    hetero_data = HeteroData()
    hetero_data['atom'].x = atom_data.x.to(device)
    hetero_data['atom', 'bonds', 'atom'].edge_index = atom_data.edge_index.to(device)
    
    if not bonds:
        hetero_data['bond'].x = torch.empty(0, 4, dtype=torch.float, device=device)
        return hetero_data
        
    bond_features = []
    bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for bond_idx in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(bond_idx)
        b_type = bond.GetBondType()
        features = [1.0 if b_type == t else 0.0 for t in bond_types]
        bond_features.append(features)
    
    hetero_data['bond'].x = torch.tensor(bond_features, dtype=torch.float, device=device)

    if bond_angles:
        angle_src, angle_dst, angle_attr = [], [], []
        for (bond1_idx, bond2_idx), angle in bond_angles.items():
            angle_src.extend([bond1_idx, bond2_idx])
            angle_dst.extend([bond2_idx, bond1_idx])
            angle_attr.extend([[angle], [angle]])

        hetero_data['bond', 'angle', 'bond'].edge_index = torch.tensor([angle_src, angle_dst], dtype=torch.long, device=device)
        hetero_data['bond', 'angle', 'bond'].edge_attr = torch.tensor(angle_attr, dtype=torch.float, device=device)

    bond_to_atom_src, bond_to_atom_dst = [], []
    for bond_idx, (atom1_idx, atom2_idx) in enumerate(bonds):
        bond_to_atom_src.extend([bond_idx, bond_idx])
        bond_to_atom_dst.extend([atom1_idx, atom2_idx])
        
    hetero_data['bond', 'contains', 'atom'].edge_index = torch.tensor([bond_to_atom_src, bond_to_atom_dst], dtype=torch.long, device=device)

    return hetero_data