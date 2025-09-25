import torch
from torch_geometric.data import Data, HeteroData
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.access import FUNCTIONAL_GROUPS

class FunctionalGroupExtractor:
    def __init__(self):
        self.fg_smarts = FUNCTIONAL_GROUPS
        
    def extract_functional_groups(self, mol):
        fg_matches = {}
        atom_to_fg = {}
        
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_to_fg[atom_idx] = []
        
        for fg_name, smarts in self.fg_smarts.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                matches = mol.GetSubstructMatches(pattern)
                if matches:
                    fg_matches[fg_name] = matches
                    for match in matches:
                        for atom_idx in match:
                            atom_to_fg[atom_idx].append(fg_name)
        
        return fg_matches, atom_to_fg
    
    def generate_maccs_fingerprints(self, mol, fg_matches):
        mol_maccs = MACCSkeys.GenMACCSKeys(mol)
        mol_maccs_array = np.array(mol_maccs)
        
        fg_fingerprints = {}
        for fg_name, matches in fg_matches.items():
            if matches:
                match = matches[0]
                fg_mol = Chem.RWMol()
                atom_mapping = {}
                
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    new_atom = Chem.Atom(atom.GetSymbol())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_atom.SetHybridization(atom.GetHybridization())
                    if atom.IsInRing():
                        new_atom.SetIsAromatic(atom.GetIsAromatic())
                    else:
                        new_atom.SetIsAromatic(False)
                    new_idx = fg_mol.AddAtom(new_atom)
                    atom_mapping[atom_idx] = new_idx
                
                added_bonds = set()
                for atom_idx in match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    for bond in atom.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
                        if begin_idx in atom_mapping and end_idx in atom_mapping:
                            bond_id = tuple(sorted([atom_mapping[begin_idx], atom_mapping[end_idx]]))
                            if bond_id not in added_bonds:
                                added_bonds.add(bond_id)
                                fg_mol.AddBond(atom_mapping[begin_idx], 
                                            atom_mapping[end_idx], 
                                            bond.GetBondType())
            
                fg_mol = fg_mol.GetMol()
                
                Chem.SanitizeMol(fg_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^
                                Chem.SanitizeFlags.SANITIZE_KEKULIZE^
                                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
                
                AllChem.Compute2DCoords(fg_mol)
                fg_maccs = MACCSkeys.GenMACCSKeys(fg_mol)
                fg_fingerprints[fg_name] = np.array(fg_maccs)
        
        return fg_fingerprints


def create_hetero_graph(mol, atom_data: Data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    extractor = FunctionalGroupExtractor()
    fg_matches, atom_to_fg = extractor.extract_functional_groups(mol)
    fg_fingerprints = extractor.generate_maccs_fingerprints(mol, fg_matches)
    
    hetero_data = HeteroData()
    hetero_data['atom'].x = atom_data.x.to(device)
    hetero_data['atom', 'bonds', 'atom'].edge_index = atom_data.edge_index.to(device)
    fg_names = list(fg_fingerprints.keys())
    fg_features = [torch.tensor(fg_fingerprints[name], dtype=torch.float, device=device) for name in fg_names]
    
    if fg_features:
        hetero_data['fg'].x = torch.stack(fg_features)
        
        fg_to_atom_src = []
        fg_to_atom_dst = []
        
        fg_to_atoms = {fg_idx: set() for fg_idx in range(len(fg_names))}
        
        for atom_idx, fg_list in atom_to_fg.items():
            for fg_name in fg_list:
                if fg_name in fg_names:
                    fg_idx = fg_names.index(fg_name)
                    fg_to_atom_src.append(fg_idx)
                    fg_to_atom_dst.append(atom_idx)
                    fg_to_atoms[fg_idx].add(atom_idx)
        
        if fg_to_atom_src:
            hetero_data['fg', 'contains', 'atom'].edge_index = torch.tensor(
                [fg_to_atom_src, fg_to_atom_dst], dtype=torch.long, device=device
            )
            
            fg_to_fg_src = []
            fg_to_fg_dst = []
            bonds = []
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bonds.append((begin_idx, end_idx))
            
            for i in range(len(fg_names)):
                for j in range(i+1, len(fg_names)):
                    atoms_i = fg_to_atoms[i]
                    atoms_j = fg_to_atoms[j]
                    
                    connected = False
                    for begin_idx, end_idx in bonds:
                        if (begin_idx in atoms_i and end_idx in atoms_j) or \
                           (begin_idx in atoms_j and end_idx in atoms_i):
                            connected = True
                            break
                    
                    if connected:
                        fg_to_fg_src.extend([i, j])
                        fg_to_fg_dst.extend([j, i])
            
            if fg_to_fg_src:
                hetero_data['fg', 'bonded', 'fg'].edge_index = torch.tensor(
                    [fg_to_fg_src, fg_to_fg_dst], dtype=torch.long, device=device
                )
    
    return hetero_data