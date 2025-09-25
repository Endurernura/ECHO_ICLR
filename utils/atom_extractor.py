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
        if x not in allowable_set:
            raise Exception("输入 {0} 不在允许集合 {1} 中:".format(x, allowable_set))
        return torch.tensor(list(map(lambda s: x == s, allowable_set)), dtype=torch.float)

    def one_of_k_encoding_unk(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return torch.tensor(list(map(lambda s: x == s, allowable_set)), dtype=torch.float)

    def atom_features(self, atom):
        symbol_encoding = self.one_of_k_encoding_unk(atom.GetSymbol(),
                                        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                        'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                        'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                        'Pt', 'Hg', 'Pb', 'Unknown'])
        degree_encoding = self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        total_hs_encoding = self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        implicit_valence_encoding = self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        is_aromatic = torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        hybridization_encoding = self.one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.HybridizationType.S,
            Chem.HybridizationType.SP,
            Chem.HybridizationType.SP2,
            Chem.HybridizationType.SP3,
            Chem.HybridizationType.SP3D,
            Chem.HybridizationType.SP3D2,
            Chem.HybridizationType.UNSPECIFIED
        ])


        electronegativity_value = 0.0
        symbol = atom.GetSymbol()
        electronegativity_map = atom_list
        electronegativity_value = electronegativity_map.get(symbol, electronegativity_map['Unknown'])

        electronegativity_encoding = self.one_of_k_encoding_unk(round(electronegativity_value * 2) / 2, 
                                                    [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 'Unknown_EN'])
        
        return torch.cat((symbol_encoding, degree_encoding, total_hs_encoding, 
                        implicit_valence_encoding, is_aromatic, 
                        hybridization_encoding, electronegativity_encoding))

    def bond_features(self, bond):
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
            edge_index.extend([[start, end], [end, start]])
            feats = self.bond_features(bond)
            edge_features.extend([feats, feats])

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()
        edge_features = torch.stack(edge_features).to(device)
        graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, mol=mol)
        return graph

if __name__ == '__main__':
    from tqdm import tqdm
    error_count = 0
    for i in tqdm(range(1000)):
        try:
            smiles, label = load_smiles(i)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"invalid SMILES error. index: {i}, SMILES: {smiles}")
                error_count += 1
                continue
                
            atom_model = atom()
            graph = atom_model(mol)
            # functional = functional_2()
            # fg = functional(graph, mol)
        except Exception as e:
            print(f"index {i} error: {e}")
            error_count += 1

