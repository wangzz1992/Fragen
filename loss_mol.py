from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Allchem

suppl = Chem.SDMolSupplier('XX.sdf')

#利用物理化学性质进行过滤
def mol_property_loss(mol):
    MW = Descriptors.MolWt(mol)                  #新生成片段的分子质量
    NHA = Descriptors.NumHAcceptors(mol)         #新生成片段的氢键受体数量
    NHB = Descriptors.NumHDonors(mol)            #新生成片段的氢键供体数量
    RoC = Descriptors.RingCount(mol)             #新生成片段的环数量
    ring = mol.GetRingInfo()
    bond_index = ring.BondRings() #返回一个元组，如((2, 3, 4, 13, 0, 1), (6, 7, 8, 15, 3, 5), (10, 11, 12, 14, 0, 9))，该化合物包含两个环及组成环的化学键编号，我们发现3和0都出现在了两个不同的环中，说明该环至少为三个环并在一起，可以放弃

    ring_bond = {}                #判断重复出现的键，如果有多余等于两个键重复出现，则过滤
    for ring in bond_index:
        for index in ring:
            if index in ring_bond.keys():
                ring_bond[index] += 1
            else:
                ring_bond[index] = 1
    count_fuzed_bond = ring_bond.values()
    for value in count_fuzed_bond:
        if value > 1:
            pass
        else:
            count_fuzed_bond.remove(value)

    if len(count_fuzed_bond) < 2 and MW < 300 and NHA < 6 and NHB < 4 and RoC < 4:
        return True
    else:
        return False


 