"""The 30 corrected CSVs against the corrected Table 1, and the a priori target rule (local, stdlib only)."""
import unittest

from MoleculeACE.benchmark.campaign import default_project_root
from verify_datasets import check_all, propose_targets


class Datasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = check_all(default_project_root())

    def test_all_match_corrected_table1_and_label_scale(self):
        self.assertEqual(len(self.rows), 30)
        self.assertEqual([r['dataset'] for r in self.rows if not r['matches_table1']], [])
        self.assertEqual(sum(r['n'] for r in self.rows), 48714)

    def test_target_proposal_uses_only_data_and_is_stable(self):
        p = propose_targets(self.rows)
        self.assertEqual([r['dataset'] for _, r in p['demo']],
                         ['CHEMBL4203_Ki', 'CHEMBL4616_EC50', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL244_Ki'])
        self.assertEqual(p['pilot']['dataset'], 'CHEMBL2835_Ki')


if __name__ == '__main__':
    unittest.main()
