"""Manifest schema and stage rules (local)."""
import csv
import unittest

from MoleculeACE.benchmark.campaign import (CampaignError, build_manifest, load_manifest, validate_manifest,
                                             write_manifest)
from MoleculeACE.benchmark.const import datasets
from MoleculeACE.models.training import PAPER_NONCONSECUTIVE
from helpers import CampaignTestCase

FIVE = datasets[:5]


class StageRules(CampaignTestCase):
    def test_valid_stages(self):
        pilot = build_manifest('gcn_pilot_t', 'pilot', ['CHEMBL2034_Ki'], folds=[0, 1], max_epochs=20)
        demo = build_manifest('gcn_demo_t', 'demo', FIVE)
        full = build_manifest('gcn_full_t', 'full', datasets)
        self.assertEqual((len(pilot), len(demo), len(full)), (2, 25, 150))
        self.assertEqual({r['scientific'] for r in pilot}, {0})
        self.assertEqual({(r['scientific'], r['max_epochs']) for r in demo + full}, {(1, 300)})
        self.assertEqual([r['task_id'] for r in full], list(range(150)))

    def test_stage_violations(self):
        cases = [
            lambda: build_manifest('gcn_pilot_t', 'pilot', ['CHEMBL2034_Ki'], folds=[0], max_epochs=300),
            lambda: build_manifest('gcn_pilot_t', 'pilot', FIVE[:2], folds=[0], max_epochs=5),
            lambda: build_manifest('gcn_demo_t', 'demo', FIVE, folds=[0, 1, 2]),
            lambda: build_manifest('gcn_demo_t', 'demo', FIVE, max_epochs=100),
            lambda: build_manifest('gcn_demo_t', 'demo', FIVE[:4]),
            lambda: build_manifest('gcn_full_t', 'full', datasets[:29]),
            lambda: build_manifest('gcn_pilot_t', 'demo', FIVE),              # campaign/stage mismatch
            lambda: build_manifest('gcn_demo_../x', 'demo', FIVE),            # unsafe name
            lambda: build_manifest('gcn_demo_t', 'demo', FIVE[:4] + ['CHEMBL0_Ki']),
            lambda: build_manifest('gcn_demo_t', 'demo', FIVE, early_stopping_mode='consecutive'),
        ]
        for i, case in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(CampaignError):
                case()

    def test_mixed_protocol_and_duplicates(self):
        rows = build_manifest('gcn_demo_t', 'demo', FIVE)
        mixed = [dict(r) for r in rows]
        mixed[3]['early_stopping_mode'] = PAPER_NONCONSECUTIVE
        duplicated = [dict(r) for r in rows]
        duplicated[1]['fold'] = 0
        reordered = [dict(r) for r in rows]
        reordered[0]['task_id'] = 7
        for bad in (mixed, duplicated, reordered):
            with self.assertRaises(CampaignError):
                validate_manifest(bad)

    def test_paper_mode_is_selectable_per_campaign(self):
        rows = build_manifest('gcn_demo_paper', 'demo', FIVE, early_stopping_mode=PAPER_NONCONSECUTIVE)
        self.assertEqual({r['early_stopping_mode'] for r in rows}, {PAPER_NONCONSECUTIVE})

    def test_roundtrip_and_file_errors(self):
        path = self.root / 'jobs' / 'm.csv'
        rows = build_manifest('gcn_demo_t', 'demo', FIVE)
        write_manifest(path, rows)
        self.assertEqual(load_manifest(path), rows)
        with self.assertRaises(CampaignError):
            load_manifest(self.root / 'jobs' / 'missing.csv')
        text = path.read_text(encoding='utf-8').replace(rows[0]['data_sha256'], 'f' * 64, 1)
        path.write_text(text, encoding='utf-8')
        from MoleculeACE.benchmark.campaign import check_input_hashes
        with self.assertRaises(CampaignError):
            check_input_hashes(load_manifest(path)[0])
        with path.open('w', newline='', encoding='utf-8') as handle:
            csv.writer(handle).writerows([['task_id', 'campaign'], ['0', 'gcn_demo_t']])
        with self.assertRaises(CampaignError):
            load_manifest(path)


class CommittedManifests(unittest.TestCase):
    def test_jobs_manifests_are_valid_and_match_current_inputs(self):
        from MoleculeACE.benchmark.campaign import check_input_hashes, default_project_root
        found = sorted((default_project_root() / 'jobs').glob('manifest_*.csv'))
        self.assertTrue(found)
        for path in found:
            rows = load_manifest(path)
            self.assertEqual(path.stem, f"manifest_{rows[0]['stage']}")
            for row in rows:
                check_input_hashes(row)


if __name__ == '__main__':
    unittest.main()
