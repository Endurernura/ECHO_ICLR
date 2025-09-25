FUNCTIONAL_GROUPS = {
    # 'alcohol': '[OX2H]',
    'aldehyde': '[CX3H1](=O)',
    'ketone': '[CX3](=O)[#6]',
    'carboxylic_acid': '[CX3](=O)[OX2H1]',
    'ester': '[CX3](=O)[OX2][#6]',
    'amide': '[NX3][CX3](=[OX1])',
    'amine': '[NX3;H2,H1,H0]',
    'nitro': '[NX3](=O)=O',
    'nitrile': '[NX1]#[CX2]',
    'sulfide': '[#16X2][#6]',
    'sulfoxide': '[#16X3](=[OX1])[#6]',
    'sulfone': '[#16X4](=[OX1])(=[OX1])[#6]',
    'alkene': '[CX3]=[CX3]',
    'alkyne': '[CX2]#[CX2]',
    'methyl': '[CH3]',
    'methylene': '[CH2]',
    'halogen': '[F,Cl,Br,I]',
    'phenyl': 'c1ccccc1',
    'hydroxyl': '[OH]',
    'thiol': '[SH]',
    'phosphate': '[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]',
    'ether': '[OD2]([#6])[#6]',
    'thioether': '[#16X2]([#6])[#6]',
    'primary_amine': '[NH2][CX4]',
    'secondary_amine': '[NH1]([CX4])[CX4]',
    'tertiary_amine': '[NX3]([CX4])([CX4])[CX4]',
    'imine': '[NX2]=[CX3]',
    'guanidine': '[NX3][CX3](=[NX2])[NX3]',
    'amidine': '[NX3][CX3]=[NX2]',
    'azide': '[NX1]=[NX2]=[NX1]',
    'isocyanate': '[NX2]=[CX2]=[OX1]',
    'isothiocyanate': '[NX2]=[CX2]=[SX1]',
    'sulfonamide': '[SX4](=[OX1])(=[OX1])([NX3])[#6]',
    'sulfonate': '[SX4](=[OX1])(=[OX1])([OX2])[#6]',
    'phosphonate': '[PX4](=[OX1])([OX2][#6])([OX2][#6])',
    'carbamate': '[NX3][CX3](=[OX1])[OX2][#6]',
    'urea': '[NX3][CX3](=[OX1])[NX3]',
    'thiourea': '[NX3][CX3](=[SX1])[NX3]',
    'heterocyclic': '[$(c1ncccc1),$(c1ncncc1),$(c1cncnc1)]',
    'epoxide': '[CX4]1[OX2][CX4]1',
    'acyl_halide': '[CX3](=[OX1])[FX1,ClX1,BrX1,IX1]',
    'anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'peroxide': '[OX2][OX2]',
    'organometallic': '[#6][Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,In,Sn,Pb,Bi]',
}

atom_list = {
    'H': 2.20,  # 氢
    'C': 2.55,  # 碳
    'N': 3.04,  # 氮
    'O': 3.44,  # 氧
    'F': 3.98,  # 氟
    'Si': 1.90, # 硅
    'P': 2.19,  # 磷
    'S': 2.58,  # 硫
    'Cl': 3.16, # 氯
    'Br': 2.96, # 溴
    'I': 2.66,  # 碘
    'Li': 0.98, # 锂
    'Na': 0.93, # 钠
    'K': 0.82,  # 钾
    'Mg': 1.31, # 镁
    'Ca': 1.00, # 钙
    'Al': 1.61, # 铝
    'B': 2.04,  # 硼
    'As': 2.18, # 砷
    'Sb': 2.05, # 锑
    'Sn': 1.96, # 锡
    'Pb': 2.33, # 铅
    'Ge': 2.01, # 锗
    'Fe': 1.83, # 铁
    'Co': 1.88, # 钴
    'Ni': 1.91, # 镍
    'Cu': 1.90, # 铜
    'Zn': 1.65, # 锌 (常见值, 来源可能略有不同)
    'Ag': 1.93, # 银
    'Cd': 1.69, # 镉
    'In': 1.78, # 铟
    'Mn': 1.55, # 锰
    'Cr': 1.66, # 铬
    'Pd': 2.20, # 钯
    'Pt': 2.28, # 铂
    'Au': 2.54, # 金
    'Hg': 2.00, # 汞
    'Se': 2.55, # 硒
    'Ti': 1.54, # 钛 (常见值, 来源可能略有不同)
    'Zr': 1.33, # 锆 (常见值, 来源可能略有不同)
    'V': 1.63,  # 钒 (常见值, 来源可能略有不同)
    'Tl': 1.62, # 铊 (常见值, 来源可能略有不同)
    'Yb': 1.1,  # 镱 (常见值, 来源可能略有不同)
    'Unknown': 0.0 # 用于处理未知的元素符号
}