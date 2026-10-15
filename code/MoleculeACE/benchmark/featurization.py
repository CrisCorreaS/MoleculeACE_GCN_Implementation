"""Convert molecular SMILES to the retained graph or ECFP representations.

The benchmark uses 37 node features and two directed edges per bond. Each
bonded edge has six features, although GCNmodel.forward uses only node features
and connectivity. No response or activity-cliff label enters these features.

Inputs are SMILES from the corrected benchmark; outputs are in-memory graph
objects or fingerprint arrays. Import Featurizer from this module in the pinned
RDKit/Torch/PyG environment. Even its ECFP method shares those module imports;
verify_cliffs.py computes its fingerprints independently without Torch.

Author: Derek van Tilborg -- TU/e -- 20-05-2022. Graph features adapted by
Luke Rossen. The inherited feature calculations are retained; bondless graphs
use the same six-column edge representation as bonded graphs.
See docs/protocol.md and docs/upstream_changes.md.
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
    """Expose the retained GRAPH and ECFP representations in input order."""
    def __init__(self):
        pass

    @staticmethod
    def ecfp(smiles: List[str], radius: int = 2, nbits: int = 1024):
        """Convert SMILES to fixed-length radius-based fingerprints.

        Parameters
        ----------
        smiles : list of str
            Molecular structures in desired row order.
        radius : int
            Morgan fingerprint radius; two for the benchmark cliff definition.
        nbits : int
            Fingerprint length; 1024 by default.

        Returns
        -------
        numpy.ndarray, shape (n_molecules, nbits)
            Binary ECFP features. This method does not use activity values.
        """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

        fp = [GetMorganFingerprintAsBitVect(m, radius, nBits=nbits) for m in mols_from_smiles(smiles)]
        return rdkit_numpy_convert(fp)

    @staticmethod
    def graphs(smiles: List[str]):
        """Build one retained molecular graph per SMILES.

        Parameters
        ----------
        smiles : list of str
            Structures supported by the inherited atom and bond vocabularies.

        Returns
        -------
        list of torch_geometric.data.Data
            Graphs in input order, with no response or cliff labels attached.
        """
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
    """Construct and sanitise a molecule and calculate Gasteiger charges.

    Parameters
    ----------
    smiles : str
        A valid molecular SMILES supported by the retained RDKit version.

    Returns
    -------
    rdkit.Chem.Mol
        Molecule with stereochemistry and per-atom Gasteiger charge properties.

    Notes
    -----
    The inherited fallback retries sanitisation without the operation that failed.
    Invalid/unsupported molecular input is not converted into a placeholder graph.
    """

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
    """Replace temporary tensors with 37 node features and six features per edge."""

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
        """Populate feature tensors from the graph's SMILES.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Temporary graph carrying a smiles attribute.

        Returns
        -------
        torch_geometric.data.Data
            The same object with x of shape (n_atoms, 37), edge_index of shape
            (2, 2 * n_bonds), and edge_attr of shape (2 * n_bonds, 6).
            Node features encode element, atomic weight, valence, hydrogen count,
            degree, hybridisation, partial charge, aromaticity and ring membership.

        Notes
        -----
        Bondless graphs retain edge_attr shape (0, 6), allowing them to share a batch
        with bonded graphs. Bond features are not consumed by the retained GCN
        forward pass.
        """
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
            data.edge_attr = torch.zeros((0, 6), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


def featurize_graph(smiles: str):
    """Construct a graph and replace its temporary features with GenFeatures.

    Parameters
    ----------
    smiles : str
        A valid, non-empty molecule in the retained atom/bond vocabulary.

    Returns
    -------
    torch_geometric.data.Data
        Molecular graph with SMILES, 37 final node features and directed connectivity.
        Bond attributes follow GenFeatures, including its documented no-bond case.

    Notes
    -----
    The initial integer tensors, including their inherited intermediate reshape,
    are overwritten by GenFeatures before return. This function does not fit a
    transform, read activity values or write files.
    """
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


    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    graph = GenFeatures()(graph)

    return graph
