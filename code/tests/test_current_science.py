"""Check current cliff-metric, potency-boundary and bondless-graph behaviour.

Tests import active submission code and use synthetic values or small real
molecules. No training, saved prediction changes or historical-source imports
occur. Missing optional chemistry or graph libraries produce explicit skips;
installed but broken dependencies remain test failures.
"""

import importlib.util
import math
import unittest


def needs(*modules: str):
    """Skip a dependency group only when its top-level package is absent."""
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    return unittest.skipIf(
        bool(missing), "Missing optional packages: " + ", ".join(missing)
    )


@needs("numpy", "pandas", "yaml")
class ExplicitCliffMask(unittest.TestCase):
    """Keep response errors separate from concentration-based cliff detection."""

    def test_explicit_integer_and_boolean_masks_select_only_cliff_errors(self) -> None:
        from MoleculeACE.benchmark.utils import calc_cliff_rmse

        observed = [1.0, -2.0, 4.0, -5.0]
        predicted = [3.0, 50.0, 1.0, 200.0]
        for mask in ([1, 0, 1, 0], [True, False, True, False]):
            with self.subTest(mask=mask):
                self.assertAlmostEqual(
                    calc_cliff_rmse(predicted, observed, cliff_mols_test=mask),
                    math.sqrt((2.0**2 + 3.0**2) / 2.0),
                    places=12,
                )

    def test_no_cliff_molecules_returns_none(self) -> None:
        from MoleculeACE.benchmark.utils import calc_cliff_rmse

        self.assertIsNone(calc_cliff_rmse([2.0, 3.0], [1.0, 4.0], [0, 0]))

    def test_missing_mask_requires_an_explicit_mask(self) -> None:
        from MoleculeACE.benchmark.utils import calc_cliff_rmse

        with self.assertRaisesRegex(ValueError, "cliff_mols_test.*explicitly"):
            calc_cliff_rmse([-1.0, -2.0], [-1.0, -2.0])

    def test_legacy_recomputation_arguments_cannot_replace_a_mask(self) -> None:
        from MoleculeACE.benchmark.utils import calc_cliff_rmse

        with self.assertRaisesRegex(ValueError, "cliff_mols_test.*explicitly"):
            calc_cliff_rmse(
                [-1.0, -2.0],
                [-1.0, -2.0],
                smiles_test=["c1ccccc1", "c1ccccc1"],
                y_train=[-3.0],
                smiles_train=["c1ccccc1"],
                potency_fold=10,
            )


@needs("numpy", "pandas", "yaml", "rdkit", "Levenshtein", "tqdm")
class PotencyBoundary(unittest.TestCase):
    """Compare real molecular similarities with the bulk cliff definition."""

    def test_exact_tenfold_ratio_is_not_a_cliff_in_either_direction(self) -> None:
        from MoleculeACE.benchmark.cliffs import is_cliff

        for concentrations in ((1.0, 10.0), (10.0, 1.0)):
            with self.subTest(concentrations=concentrations):
                self.assertFalse(is_cliff("c1ccccc1", "c1ccccc1", *concentrations))

    def test_pairwise_and_bulk_rules_agree_around_threshold(self) -> None:
        from MoleculeACE.benchmark.cliffs import ActivityCliffs, is_cliff

        for concentration, expected in ((9.99, False), (10.0, False), (10.01, True)):
            with self.subTest(concentration=concentration):
                pair = is_cliff(
                    "c1ccccc1", "c1ccccc1", 1.0, concentration, similarity=1.0
                )
                bulk = ActivityCliffs(
                    ["c1ccccc1", "c1ccccc1"], [1.0, concentration]
                ).find_cliffs(similarity=1.0)
                self.assertEqual(bool(pair), expected)
                self.assertEqual(bool(bulk[0, 1]), expected)
                self.assertEqual(bool(bulk[1, 0]), expected)

    def test_custom_potency_threshold_is_also_strict(self) -> None:
        from MoleculeACE.benchmark.cliffs import is_cliff

        self.assertFalse(is_cliff("c1ccccc1", "c1ccccc1", 2.0, 6.0, potency_fold=3))
        self.assertTrue(is_cliff("c1ccccc1", "c1ccccc1", 2.0, 6.01, potency_fold=3))

    def test_high_potency_difference_still_requires_structural_similarity(self) -> None:
        from MoleculeACE.benchmark.cliffs import is_cliff

        self.assertFalse(is_cliff("c1ccccc1", "CCO", 1.0, 100.0))


@needs("numpy", "pandas", "yaml", "rdkit", "torch", "torch_geometric")
class BondlessGraphs(unittest.TestCase):
    """Exercise real RDKit featurisation and PyG batching without a model."""

    def test_bondless_graph_has_six_edge_feature_columns(self) -> None:
        import torch
        from MoleculeACE.benchmark.featurization import featurize_graph

        graph = featurize_graph("C")
        self.assertEqual(tuple(graph.x.shape), (1, 37))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 0))
        self.assertEqual(tuple(graph.edge_attr.shape), (0, 6))
        self.assertEqual(graph.edge_index.dtype, torch.long)
        self.assertEqual(graph.edge_attr.dtype, torch.float)
        self.assertTrue(torch.isfinite(graph.x).all().item())

    def test_bondless_and_bonded_graphs_batch_in_both_orders(self) -> None:
        import torch
        from torch_geometric.data import Batch
        from MoleculeACE.benchmark.featurization import featurize_graph

        bondless = featurize_graph("C")
        bonded = featurize_graph("CCO")
        for graphs, expected_batch, offset in (
            ([bondless, bonded], [0, 1, 1, 1], 1),
            ([bonded, bondless], [0, 0, 0, 1], 0),
        ):
            with self.subTest(bondless_first=offset == 1):
                batch = Batch.from_data_list(graphs)
                self.assertEqual(batch.num_graphs, 2)
                self.assertEqual(tuple(batch.x.shape), (4, 37))
                self.assertEqual(tuple(batch.edge_index.shape), (2, 4))
                self.assertEqual(tuple(batch.edge_attr.shape), (4, 6))
                self.assertEqual(batch.batch.tolist(), expected_batch)
                self.assertTrue(torch.equal(batch.edge_attr, bonded.edge_attr))
                self.assertTrue(
                    torch.equal(batch.edge_index, bonded.edge_index + offset)
                )
                self.assertTrue(torch.isfinite(batch.x).all().item())


if __name__ == "__main__":
    unittest.main()
