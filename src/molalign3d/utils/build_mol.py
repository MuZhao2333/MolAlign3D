import json
import os
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
import numpy as np
from openbabel import openbabel as ob
from scipy.spatial.distance import pdist, squareform
import itertools
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdFMCS
class MolReconError(Exception):
    pass


UPGRADE_BOND_ORDER = {AllChem.BondType.SINGLE: AllChem.BondType.DOUBLE, AllChem.BondType.DOUBLE: AllChem.BondType.TRIPLE}


def calc_valence(rdatom):
    cnt = 0.0
    for bond in rdatom.GetBonds():
        cnt += bond.GetBondTypeAsDouble()
    return cnt


def fixup(atoms, mol, indicators):
    """Set atom properties to match channel.  Keep doing this to beat openbabel over the head with what we want to happen."""
    mol.SetAromaticPerceived(True)
    for i, atom in enumerate(atoms):
        if indicators is not None:
            if indicators[i]:
                atom.SetAromatic(True)
                atom.SetHyb(2)
            else:
                atom.SetAromatic(False)

        if (atom.GetAtomicNum() in (7, 8)) and atom.IsInRing():
            acnt = 0
            for nbr in ob.OBAtomAtomIter(atom):
                if nbr.IsAromatic():
                    acnt += 1
            if acnt > 1:
                atom.SetAromatic(True)


def postprocess_rd_mol_1(rdmol, remove_hs=True):
    if remove_hs:
        rdmol = AllChem.RemoveHs(rdmol)

    # Construct bond nbh list
    nbh_list = {}
    for bond in rdmol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in nbh_list:
            nbh_list[begin] = [end]
        else:
            nbh_list[begin].append(end)

        if end not in nbh_list:
            nbh_list[end] = [begin]
        else:
            nbh_list[end].append(begin)

    # Fix missing bond-order
    for atom in rdmol.GetAtoms():
        idx = atom.GetIdx()
        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            for j in nbh_list.get(idx, []):
                if j <= idx:
                    continue
                nb_atom = rdmol.GetAtomWithIdx(j)
                nb_radical = nb_atom.GetNumRadicalElectrons()
                if nb_radical > 0:
                    bond = rdmol.GetBondBetweenAtoms(idx, j)
                    bond.SetBondType(UPGRADE_BOND_ORDER[bond.GetBondType()])
                    nb_atom.SetNumRadicalElectrons(nb_radical - 1)
                    num_radical -= 1
            atom.SetNumRadicalElectrons(num_radical)

        num_radical = atom.GetNumRadicalElectrons()
        if num_radical > 0:
            atom.SetNumRadicalElectrons(0)
            num_hs = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(num_hs + num_radical)

    return rdmol


def postprocess_rd_mol_2(rdmol):
    rdmol_edit = AllChem.RWMol(rdmol)

    ring_info = rdmol.GetRingInfo()
    ring_info.AtomRings()
    rings = [set(r) for r in ring_info.AtomRings()]
    for i, ring_a in enumerate(rings):
        if len(ring_a) == 3:
            non_carbon = []
            atom_by_symb = {}
            for atom_idx in ring_a:
                symb = rdmol.GetAtomWithIdx(atom_idx).GetSymbol()
                if symb != 'C':
                    non_carbon.append(atom_idx)
                if symb not in atom_by_symb:
                    atom_by_symb[symb] = [atom_idx]
                else:
                    atom_by_symb[symb].append(atom_idx)
            if len(non_carbon) == 2:
                rdmol_edit.RemoveBond(*non_carbon)
            if 'O' in atom_by_symb and len(atom_by_symb['O']) == 2:
                rdmol_edit.RemoveBond(*atom_by_symb['O'])
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][0]).GetNumExplicitHs() + 1
                )
                rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).SetNumExplicitHs(
                    rdmol_edit.GetAtomWithIdx(atom_by_symb['O'][1]).GetNumExplicitHs() + 1
                )
    rdmol = rdmol_edit.GetMol()

    for atom in rdmol.GetAtoms():
        if atom.GetFormalCharge() > 0:
            atom.SetFormalCharge(0)

    return rdmol


class MoleculeBuilder:
    def __init__(self):
        pass

    def make_obmol(self, xyz, atomic_numbers):
        """创建 OBabel 分子对象"""
        mol = ob.OBMol()
        mol.BeginModify()
        atoms = []
        if isinstance(xyz, np.ndarray):
            xyz = xyz.tolist()
        if isinstance(atomic_numbers, np.ndarray):
            atomic_numbers = atomic_numbers.tolist()

        for xyz, t in zip(xyz, atomic_numbers):
            x, y, z = xyz
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
                raise ValueError(f"Invalid coordinates: {xyz}")
            atom = mol.NewAtom()
            atom.SetAtomicNum(t)
            atom.SetVector(x, y, z)
            atoms.append(atom)
        return mol, atoms

    def connect_the_dots(self, mol, atoms, indicators=None, covalent_factor=1.3):
        """Connect atoms into bonds using covalent radii heuristics and pruning."""
        pt = Chem.GetPeriodicTable()
        if len(atoms) == 0:
            return
        mol.BeginModify()
        coords = np.array([(a.GetX(), a.GetY(), a.GetZ()) for a in atoms])
        dists = squareform(pdist(coords))
        for i, j in itertools.combinations(range(len(atoms)), 2):
            a = atoms[i]
            b = atoms[j]
            a_r = ob.GetCovalentRad(a.GetAtomicNum()) * covalent_factor
            b_r = ob.GetCovalentRad(b.GetAtomicNum()) * covalent_factor
            if dists[i, j] < a_r + b_r:
                flag = 0
                if indicators and indicators[i] and indicators[j]:
                    flag = ob.OB_AROMATIC_BOND
                mol.AddBond(a.GetIdx(), b.GetIdx(), 1, flag)
        atom_maxb = {}
        for i, a in enumerate(atoms):
            maxb = min(ob.GetMaxBonds(a.GetAtomicNum()), pt.GetDefaultValence(a.GetAtomicNum()))
            if a.GetAtomicNum() == 16 and self.count_nbrs_of_elem(a, 8) >= 2:
                maxb = 6
            atom_maxb[a.GetIdx()] = maxb
        for bond in ob.OBMolBondIter(mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if atom_maxb[a1.GetIdx()] == 1 and atom_maxb[a2.GetIdx()] == 1:
                mol.DeleteBond(bond)

        def get_bond_info(biter):
            bonds = [b for b in biter]
            binfo = []
            for bond in bonds:
                bdist = bond.GetLength()
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                ideal = ob.GetCovalentRad(a1.GetAtomicNum()) + ob.GetCovalentRad(a2.GetAtomicNum())
                stretch = bdist / ideal
                binfo.append((stretch, bond))
            binfo.sort(reverse=True, key=lambda t: t[0])
            return binfo

        binfo = get_bond_info(ob.OBMolBondIter(mol))
        for stretch, bond in binfo:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if stretch > 1.2 or self.forms_small_angle(a1, a2) or self.forms_small_angle(a2, a1):
                if not self.reachable(a1, a2):
                    continue
                mol.DeleteBond(bond)

        hypers = [(atom_maxb[a.GetIdx()], a.GetExplicitValence() - atom_maxb[a.GetIdx()], a) for a in atoms]
        hypers = sorted(hypers, key=lambda aa: (aa[0], -aa[1]))
        for mb, diff, a in hypers:
            if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                continue
            binfo = get_bond_info(ob.OBAtomBondIter(a))
            for stretch, bond in binfo:
                if stretch < 0.9:
                    continue
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetExplicitValence() > atom_maxb[a1.GetIdx()] or a2.GetExplicitValence() > atom_maxb[a2.GetIdx()]:
                    if not self.reachable(a1, a2):
                        continue
                    mol.DeleteBond(bond)
                    if a.GetExplicitValence() <= atom_maxb[a.GetIdx()]:
                        break
        mol.EndModify()

    def convert_ob_mol_to_rd_mol(self, ob_mol, require_H=True):
        """将 OBMol 转换为 RDKit Mol（保留坐标），不进行深度 sanitize。"""
        if not require_H:
            ob_mol.DeleteHydrogens()
        n_atoms = ob_mol.NumAtoms()
        rd_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(n_atoms)
        for ob_atom in ob.OBMolAtomIter(ob_mol):
            rd_atom = Chem.Atom(ob_atom.GetAtomicNum())
            if ob_atom.IsAromatic() and ob_atom.IsInRing() and ob_atom.MemberOfRingSize() <= 6:
                rd_atom.SetIsAromatic(True)
            i = rd_mol.AddAtom(rd_atom)
            ob_coords = ob_atom.GetVector()
            x = ob_coords.GetX()
            y = ob_coords.GetY()
            z = ob_coords.GetZ()
            rd_conf.SetAtomPosition(i, Geometry.Point3D(x, y, z))
        rd_mol.AddConformer(rd_conf)
        for ob_bond in ob.OBMolBondIter(ob_mol):
            i = ob_bond.GetBeginAtomIdx() - 1
            j = ob_bond.GetEndAtomIdx() - 1
            bond_order = ob_bond.GetBondOrder()
            if bond_order == 1:
                rd_mol.AddBond(i, j, AllChem.BondType.SINGLE)
            elif bond_order == 2:
                rd_mol.AddBond(i, j, AllChem.BondType.DOUBLE)
            elif bond_order == 3:
                rd_mol.AddBond(i, j, AllChem.BondType.TRIPLE)
            else:
                raise Exception('unknown bond order {}'.format(bond_order))
            if ob_bond.IsAromatic():
                bond = rd_mol.GetBondBetweenAtoms(i, j)
                bond.SetIsAromatic(True)
        if not require_H:
            rd_mol = Chem.RemoveHs(rd_mol, sanitize=False)
        return rd_mol

    def build_mol(self, xyz, atomic_nums, aromatic=None, basic_mode=True, require_H=True, raise_on_error=False):
        """从坐标与原子类型重建分子，流程对齐自 reconstruct.py，可能抛出 MolReconError。"""
        if basic_mode:
            indicators = None
        else:
            indicators = aromatic

        if isinstance(xyz, np.ndarray):
            xyz = xyz.tolist()
        if isinstance(atomic_nums, np.ndarray):
            atomic_nums = atomic_nums.tolist()
        if isinstance(aromatic, np.ndarray):
            aromatic = aromatic.tolist()

        mol, atoms = self.make_obmol(xyz, atomic_nums)
        # multi-round fixup + bonding
        fixup(atoms, mol, indicators)

        self.connect_the_dots(mol, atoms, indicators, covalent_factor=1.3)
        fixup(atoms, mol, indicators)

        mol.AddPolarHydrogens()
        mol.PerceiveBondOrders()
        fixup(atoms, mol, indicators)

        for (i, a) in enumerate(atoms):
            ob.OBAtomAssignTypicalImplicitHydrogens(a)
        fixup(atoms, mol, indicators)

        mol.AddHydrogens()
        fixup(atoms, mol, indicators)

        # make rings aromatic if majority of carbons are aromatic
        for ring in ob.OBMolRingIter(mol):
            if 5 <= ring.Size() <= 6:
                carbon_cnt = 0
                aromatic_ccnt = 0
                for ai in ring._path:
                    a = mol.GetAtom(ai)
                    if a.GetAtomicNum() == 6:
                        carbon_cnt += 1
                        if a.IsAromatic():
                            aromatic_ccnt += 1
                if aromatic_ccnt >= carbon_cnt / 2 and aromatic_ccnt != ring.Size():
                    for ai in ring._path:
                        a = mol.GetAtom(ai)
                        a.SetAromatic(True)

        for bond in ob.OBMolBondIter(mol):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if a1.IsAromatic() and a2.IsAromatic():
                bond.SetAromatic(True)

        mol.PerceiveBondOrders()
        rd_mol = self.convert_ob_mol_to_rd_mol(mol, require_H=require_H)
        try:
            rd_mol = postprocess_rd_mol_1(rd_mol, remove_hs=not require_H)
            rd_mol = postprocess_rd_mol_2(rd_mol)
        except Exception as e:
            if raise_on_error:
                raise MolReconError()
            else:
                # don't interrupt; keep rd_mol as-is and continue
                print(f"Warning: postprocessing failed but continue (raise_on_error=False): {e}")

        # final aromatic consistency fix (as in reconstruct.py)
        for bond in rd_mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if bond.GetIsAromatic():
                if not a1.GetIsAromatic() or not a2.GetIsAromatic():
                    bond.SetIsAromatic(False)
            elif a1.GetIsAromatic() and a2.GetIsAromatic():
                bond.SetIsAromatic(True)

        # attempt sanitize, raising MolReconError on failure
        try:
            AllChem.SanitizeMol(rd_mol, AllChem.SANITIZE_ALL ^ AllChem.SANITIZE_KEKULIZE)
        except Exception as e:
            if raise_on_error:
                raise MolReconError()
            else:
                # attempt to continue even if sanitize failed
                print(f"Warning: sanitize failed but continue (raise_on_error=False): {e}")

        return rd_mol

    def compute_iou(self, original_mol, generated_mol):
        """计算 IOU 使用重原子计数"""
        # 移除氢原子以确保 MCS 和总原子数的一致性
        m1 = Chem.RemoveHs(original_mol, sanitize=False)
        m2 = Chem.RemoveHs(generated_mol, sanitize=False)
        
        mcs_result = rdFMCS.FindMCS([m1, m2], timeout=2, bondCompare=rdFMCS.BondCompare.CompareAny)
        mcs_atom_count = mcs_result.numAtoms
        
        n1 = m1.GetNumAtoms()
        n2 = m2.GetNumAtoms()
        union = n1 + n2 - mcs_atom_count
        return mcs_atom_count / union if union > 0 else 0.0

    def reachable(self, a, b):
        if a.GetExplicitDegree() == 1 or b.GetExplicitDegree() == 1:
            return False
        seenbonds = set([a.GetBond(b).GetIdx()])
        return self.reachable_r(a, b, seenbonds)

    def reachable_r(self, a, b, seenbonds):
        for nbr in ob.OBAtomAtomIter(a):
            bond = a.GetBond(nbr).GetIdx()
            if bond not in seenbonds:
                seenbonds.add(bond)
                if nbr == b:
                    return True
                elif self.reachable_r(nbr, b, seenbonds):
                    return True
        return False

    def forms_small_angle(self, a, b, cutoff=60):
        for nbr in ob.OBAtomAtomIter(a):
            if nbr != b:
                degrees = b.GetAngle(a, nbr)
                if degrees < cutoff:
                    return True
        return False

    def count_nbrs_of_elem(self, atom, atomic_num):
        count = 0
        for nbr in ob.OBAtomAtomIter(atom):
            if nbr.GetAtomicNum() == atomic_num:
                count += 1
        return count