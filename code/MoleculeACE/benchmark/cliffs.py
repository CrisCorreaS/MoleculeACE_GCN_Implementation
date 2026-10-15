"""Recompute activity-cliff membership from structures and concentrations.

ActivityCliffs compares all supplied molecules, normally the complete published
train-plus-test dataset. Default structural similarity is the OR of radius-2,
1024-bit ECFP Tanimoto, generic-scaffold Tanimoto and normalised SMILES
Levenshtein similarity, each at least 0.9. Concentration ratios must be
strictly greater than 10. Molecule membership counts each molecule once.

Inputs are aligned SMILES and positive concentrations in nM, not the model's
negative logarithmic response. Outputs are in-memory pair matrices and labels;
the CLI verify_cliffs.py records comparisons without replacing published data.
NumPy, RDKit, Levenshtein and tqdm are needed on CPU; Torch is not used here.
Package initialisation also requires pandas and PyYAML. See docs/protocol.md.

Author: Derek van Tilborg -- TU/e -- 25-05-2022. Optional matched-molecular-pair
helpers are retained upstream interfaces, not the completed campaign definition.
The pairwise helper uses the same exclusive concentration-ratio threshold.
"""

from typing import List, Callable, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdMMPA import FragmentMol
from Levenshtein import distance as levenshtein
from tqdm import tqdm

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


class ActivityCliffs:
    """Build pairwise cliff relations and derive per-molecule membership."""
    def __init__(self, smiles: List[str], bioactivity: Union[List[float], np.array]):
        """Store aligned structures and untransformed bioactivities.

        Parameters
        ----------
        smiles : list of str
            Molecular SMILES in the dataset's row order.
        bioactivity : list of float or numpy.ndarray, shape (n_molecules,)
            Positive concentrations in a common unit, nM for this benchmark. Do not
            supply y = -log10(nM). The constructor does not validate these preconditions.
        """
        self.smiles = smiles
        self.bioactivity = list(bioactivity) if type(bioactivity) is not list else bioactivity
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9, potency_fold: float = 10, mmp: bool = False,
                    custom_cliff_function: Callable = None):
        """Calculate a symmetric activity-cliff adjacency matrix.

        Parameters
        ----------
        similarity : float
            Inclusive structural-similarity threshold for the default three measures.
        potency_fold : float
            Exclusive concentration-ratio threshold; ten in the benchmark.
        mmp : bool
            Select the retained matched-pair alternative instead of the default definition.
        custom_cliff_function : callable, optional
            Function returning a binary structural-similarity matrix, combined by OR
            with the selected similarity definition.

        Returns
        -------
        numpy.ndarray, shape (n_molecules, n_molecules)
            Binary cliff-pair matrix, also stored in self.cliffs. The default benchmark
            path uses structural similarity >= 0.9 and concentration ratio > 10.
        """

        if mmp:
            sim = mmp_similarity(self.smiles)
        else:
            sim = moleculeace_similarity(self.smiles, similarity)

        if custom_cliff_function is not None:
            custom_sim = custom_cliff_function(self.smiles, similarity)
            sim = np.logical_or(sim == 1, custom_sim == 1).astype(int)

        fc = (get_fc(self.bioactivity) > potency_fold).astype(int)

        self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)

        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        """Return molecules that participate in at least one cliff pair.

        Parameters
        ----------
        return_smiles : bool
            Return selected SMILES when True; otherwise return a row-aligned binary mask.
        **kwargs
            Arguments for find_cliffs, used only when no matrix has been calculated.

        Returns
        -------
        list
            Selected SMILES or one 0/1 entry per input molecule. Membership counts a
            molecule once, however many cliff partners it has. The cached matrix is
            reused on later calls; changed thresholds require a new calculation.
        """
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            return list((sum(self.cliffs) > 0).astype(int))

    def __repr__(self):
        return "Activity cliffs"


def find_fc(a: float, b: float):
    """Calculate the concentration ratio of two positive activities.

    Parameters
    ----------
    a, b : float
        Positive concentrations in the same unit, not logarithmic responses.

    Returns
    -------
    float
        max(a, b) / min(a, b). Positivity is a caller precondition.
    """

    return max([a, b]) / min([a, b])


def get_fc(bioactivity: List[float]):
    """Calculate all pairwise concentration ratios with a zero diagonal.

    Parameters
    ----------
    bioactivity : list of float
        Positive concentrations in one common unit.

    Returns
    -------
    numpy.ndarray, shape (n_molecules, n_molecules)
        Symmetric fold-change matrix; self-pairs are set to zero.
    """

    act_len = len(bioactivity)
    m = np.zeros([act_len, act_len])
    # Calculate upper triangle of matrix
    for i in range(act_len):
        for j in range(i, act_len):
            m[i, j] = find_fc(bioactivity[i], bioactivity[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_levenshtein_matrix(smiles: List[str], normalize: bool = True, hide: bool = False, top_n: int = None):
    """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string"""

    smi_len = len(smiles)

    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            if normalize:
                m[i, j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            else:
                m[i, j] = levenshtein(smiles[i], smiles[j])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Get from a distance to a similarity
    m = 1 - m

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""

    # Make a fingerprint database
    db_fp = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]],
                                                     db_fp[smiles[j]])
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def get_scaffold_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024, hide: bool = False, top_n: int = None):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string """

    # Make scaffold database
    db_scaf = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(m)
        except Exception:  # In the very rare case this doesn't work, use a normal scaffold
            print(f"Could not create a generic scaffold of {smi}, used a normal scaffold instead")
            skeleton = GetScaffoldForMol(m)
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        db_scaf[smi] = skeleton_fp

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]],
                                                     db_scaf[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def find_fragments(smiles: List[str]):
    """ Build a database of molecular fragments for matched molecular pair analysis. Molecular fragmentation from the
    Hussain and Rae algorithm is used. We only use a single cut (true to the original MMP idea) """

    db = {}
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        cuts = FragmentMol(m, maxCuts=1, resultsAsMols=False)

        # extract all not None fragments into a flat list
        fragments = sum([[i for i in cut if i != ''] for cut in cuts], [])

        # Keep the largest fragment as the core structure.
        for i, frag in enumerate(fragments):
            split_frags = frag.split('.')
            if Chem.MolFromSmiles(split_frags[0]).GetNumAtoms() >= Chem.MolFromSmiles(split_frags[-1]).GetNumAtoms():
                core = split_frags[0]
            else:
                core = split_frags[-1]

            # Ignore dummy variables for matching (otherwise you will never get a match). Dummies are introduced during
            # fragmentation
            qp = Chem.AdjustQueryParameters()
            qp.makeDummiesQueries = True
            qp.adjustDegreeFlags = Chem.ADJUST_IGNOREDUMMIES
            fragments[i] = Chem.AdjustQueryProperties(Chem.MolFromSmiles(core), qp)
        # Add all core fragments to the dictionary
        db[smi] = fragments

    return db


def mmp_match(smiles: str, fragments: List):
    """ Check if fragments (provided as rdkit Mol objects are substructures of another molecule (from SMILES string) """

    m = Chem.MolFromSmiles(smiles)
    for frag in fragments:
        # Match fragment on molecule
        if m.HasSubstructMatch(frag):
            return 1
    return 0


def get_mmp_matrix(smiles: List[str], hide: bool = False):
    """ Calculates a matrix of matched molecular pairs for a list of SMILES string"""

    # Make a fingerprint database
    db_frags = find_fragments(smiles)

    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix.
    for i in tqdm(range(smi_len), disable=hide):
        for j in range(i, smi_len):
            m[i, j] = mmp_match(smiles[i], db_frags[smiles[j]])

    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))

    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m


def mmp_similarity(smiles: List[str], similarity=None):
    """ Calculate which pairs of molecules are matched molecular pairs """

    return (get_mmp_matrix(smiles) > 0).astype(int)


def moleculeace_similarity(smiles: List[str], similarity: float = 0.9, hide: bool = False):
    """Combine the three benchmark structural-similarity thresholds.

    Parameters
    ----------
    smiles : list of str
        Structures in dataset order.
    similarity : float
        Inclusive threshold, 0.9 for the benchmark.
    hide : bool
        Suppress progress displays when True.

    Returns
    -------
    numpy.ndarray, shape (n_molecules, n_molecules)
        Binary OR of ECFP, generic-scaffold and normalised Levenshtein similarity
        masks. The diagonal is zero at the benchmark's positive threshold.
    """

    m_tani = get_tanimoto_matrix(smiles, hide=hide) >= similarity
    m_scaff = get_scaffold_matrix(smiles, hide=hide) >= similarity
    m_leve = get_levenshtein_matrix(smiles, hide=hide) >= similarity

    return (m_tani + m_scaff + m_leve).astype(int)


def is_cliff(smiles1, smiles2, y1, y2, similarity: float = 0.9, potency_fold: float = 10):
    """Apply the structural-similarity rule and strict potency threshold to a pair.

    Parameters
    ----------
    smiles1, smiles2 : str
        Structures compared by the default three-measure similarity rule.
    y1, y2 : float
        Despite their names, positive concentrations in the same unit, not model y.
    similarity : float
        Inclusive structural-similarity threshold.
    potency_fold : float
        Exclusive concentration-ratio threshold, matching ActivityCliffs.

    Returns
    -------
    bool or numpy.bool_
        Whether the pair passes the helper's criteria.

    Notes
    -----
    The pairwise helper and ActivityCliffs both require ratio > potency_fold.
    A change of exactly tenfold is not a cliff at the default threshold.
    """
    sim = moleculeace_similarity([smiles1, smiles2], similarity=similarity, hide=True)[0][1]
    fc = get_fc([y1, y2])[0][1]

    return sim == 1 and fc > potency_fold
