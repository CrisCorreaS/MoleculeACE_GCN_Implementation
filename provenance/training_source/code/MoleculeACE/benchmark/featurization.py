"""
Author: Derek van Tilborg -- TU/e -- 20-05-2022

All code for featurizing a molecules into molecular descriptors and molecular graphs.

    - Featurizer:                   Main molecule featurizer class that converts SMILES strings to computable info.
                                        example:    x_train = Featurizer().ecfp(list_of_smiles)
        - ecfp()
        - graphs()

    - featurize_graph():            Construct a molecular graph from a SMILES string

Reduced to GCN_GRAPH (see docs/upstream_changes.md): graphs() and ecfp() are kept unchanged (ECFP is part of the
activity-cliff definition). Removed: maccs(), whim(), physchem(), one_hot(), tokens(), OneHotEncodeSMILES,
compute_whim() (it wrote whim_database.pkl in the working directory) and compute_physchem().

"""
from MoleculeACE.benchmark.const import Descriptors
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from typing import List
import numpy as np
import torch

x_map = {'symbol': ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'],
         'atomic_weight': float,
         'n_valence': int,
         'num_hs': int,
         'degree': list(range(0, 7)),
         'hybridization': ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'],
         'partial_charge': float,
         'is_aromatic': [True, False],
         'is_in_ring': [True, False]}

e_map = {'bond_type': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
         'is_conjugated': [True, False]}


class Featurizer:
    def __init__(self):
        pass

    @staticmethod
    def ecfp(smiles: List[str], radius: int = 2, nbits: int = 1024):
        """ Convert SMILES to ECFP fingerprints """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

        fp = [GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols_from_smiles(smiles)]
        return rdkit_numpy_convert(fp)

    @staticmethod
    def graphs(smiles: List[str]):
        return [featurize_graph(smi) for smi in smiles]

    def __call__(self, descriptor: Descriptors, **kwargs):
        if descriptor.name == 'ECFP':
            return self.ecfp(**kwargs)
        if descriptor.name == 'GRAPH':
            return self.graphs( **kwargs)
        # Upstream silently returned None for unknown descriptors
        raise ValueError(f"Descriptor {descriptor} is not available in this GCN-only version")

    def __repr__(self):
        return "Molecule Featurizer that converts SMILES strings to: ecfp(), graphs()"


def mols_from_smiles(smiles: List[str]):
    """ Create a list of RDkit mol objects from a list of SMILES strings """
    from rdkit.Chem import MolFromSmiles
    return [MolFromSmiles(m) for m in smiles]


def rdkit_numpy_convert(fp):
    """ Convert a RDkit fingerprint object to simple numpy array """
    from rdkit.DataStructs import ConvertToNumpyArray
    output = []
    for f in fp:
        arr = np.zeros((1,))
        ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def sigmoid(number: float):
    """ numerically semi-stable sigmoid function to map charge between 0 and 1 """
    return 1.0 / (1.0 + float(np.exp(-number)))


def molecule_from_smiles(smiles: str):
    """ Sanitize a molecule from a SMILES string"""

    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    Chem.rdPartialCharges.ComputeGasteigerCharges(molecule)

    return molecule


class GenFeatures(object):
    """ Adapted by Luke Rossen """

    def __init__(self):
        self.symbols = [
            "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S", "Si", 'Se', 'Te'
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

    def __call__(self, data):
        # Create rdkit mol object
        mol = molecule_from_smiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.

            atomic_weight = sigmoid(Chem.GetPeriodicTable().GetAtomicWeight(atom.GetSymbol()))

            n_valence = float(atom.GetTotalValence())
            n_hydrogens = float(atom.GetTotalNumHs())

            degree = [0.] * 7
            degree[atom.GetDegree()] = 1.

            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.

            partial_charge = sigmoid(float(atom.GetProp('_GasteigerCharge')))

            aromatic = [0.] * 2
            if atom.GetIsAromatic():
                aromatic[0] = 1.
            else:
                aromatic[1] = 1.

            in_ring = [0.] * 2
            if atom.IsInRing():
                in_ring[0] = 1.
            else:
                in_ring[1] = 1.

            x = torch.tensor(symbol + [atomic_weight] + [n_valence] +
                             [n_hydrogens] + degree + hybridization +
                             [partial_charge] + aromatic + in_ring)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []

        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.

            conjugation = [0.] * 2
            if bond.GetIsConjugated():
                conjugation[0] = 1.
            else:
                conjugation[1] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic] + conjugation)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


def featurize_graph(smiles: str):
    # Create RDkit molobject
    mol = molecule_from_smiles(smiles)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['symbol'].index(atom.GetSymbol()))
        x.append(0)
        x.append(0)
        x.append(0)
        x.append(x_map['degree'].index(atom.GetDegree()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(0)
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, len(xs))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    # Sort indices.
    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    # y = torch.tensor(label, dtype=torch.float).view(1, -1)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    graph = GenFeatures()(graph)

    return graph
