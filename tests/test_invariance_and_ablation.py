"""Unit tests for physical invariances, parameter accounting, and readout ablations."""

import os
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chemprop.args import TrainArgs
from chemprop.features import set_polymer, MolGraph, BatchMolGraph
from chemprop.models import MoleculeModel
from run_ablation import VARIANTS, build_variant_model


SAMPLE_CSV = os.path.join(REPO_ROOT, 'data', 'sample_copolymers.csv')
CKPT_PATH = os.path.join(REPO_ROOT, 'checkpoints', 'synergy_wdmpnn_fold0_seed0.pt')


class TestSynergyWDMPNN(unittest.TestCase):
    def test_parameter_counts_match_manuscript(self):
        _, m_canon = build_variant_model('canonical', SAMPLE_CSV, epochs=1, gpu=-1)
        _, m_pm = build_variant_model('canonical_pm', SAMPLE_CSV, epochs=1, gpu=-1)
        _, m_syn = build_variant_model('synergy_full', SAMPLE_CSV, epochs=1, gpu=-1)

        p_canon = sum(p.numel() for p in m_canon.parameters() if p.requires_grad)
        p_pm = sum(p.numel() for p in m_pm.parameters() if p.requires_grad)
        p_syn = sum(p.numel() for p in m_syn.parameters() if p.requires_grad)

        self.assertEqual(p_canon, 355202)
        self.assertEqual(p_pm, 1250402)
        self.assertEqual(p_syn, 1255802)

    def test_residual_head_norm_nonzero(self):
        _, m_syn = build_variant_model('synergy_full', SAMPLE_CSV, epochs=1, gpu=-1)
        set_polymer(True)
        smi = '[*:1]c1cc(F)c([*:2])cc1F.[*:3]c1c(O)cc(O)c([*:4])c1O|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
        bm = BatchMolGraph([MolGraph(smi)])
        m_syn.eval()
        with torch.no_grad():
            out = m_syn([bm])
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_monomer_order_permutation_invariance(self):
        """Swapping Monomer A and Monomer B in the extended SMILES must yield identical predictions."""
        _, m_syn = build_variant_model('synergy_full', SAMPLE_CSV, epochs=1, gpu=-1)
        set_polymer(True)
        m_syn.eval()

        s1 = '[*:1]c1cc(F)c([*:2])cc1F.[*:3]c1c(O)cc(O)c([*:4])c1O|0.25|0.75|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
        s2 = '[*:3]c1c(O)cc(O)c([*:4])c1O.[*:1]c1cc(F)c([*:2])cc1F|0.75|0.25|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
        with torch.no_grad():
            y1 = m_syn([BatchMolGraph([MolGraph(s1)])])
            y2 = m_syn([BatchMolGraph([MolGraph(s2)])])
        max_diff = (y1 - y2).abs().max().item()
        self.assertLess(max_diff, 1e-5, f'Permutation invariance error too large: {max_diff}')

    def test_degree_of_polymerization_scaling(self):
        _, m_syn = build_variant_model('synergy_full', SAMPLE_CSV, epochs=1, gpu=-1)
        set_polymer(True)
        m_syn.eval()
        base = '[*:1]CC([*:2])c1ccccc1|1.0|<1-2:1.0:1.0'
        with_xn = base + '~100.0'  # 1 + log10(100) = 3.0
        with torch.no_grad():
            z1 = m_syn.encoder([BatchMolGraph([MolGraph(base)])])
            z2 = m_syn.encoder([BatchMolGraph([MolGraph(with_xn)])])
        ratio = (z2 / z1).mean().item()
        self.assertAlmostEqual(ratio, 3.0, places=4)

    def test_all_ablation_variants_forward(self):
        set_polymer(True)
        smi = '[*:1]c1cc(F)c([*:2])cc1F.[*:3]c1cc(C#N)c([*:4])cc1F|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5'
        bm = BatchMolGraph([MolGraph(smi)])
        for name in VARIANTS:
            _, model = build_variant_model(name, SAMPLE_CSV, epochs=1, gpu=-1)
            model.eval()
            with torch.no_grad():
                out = model([bm])
            self.assertEqual(out.shape, (1, 2), f'Failed shape for {name}')
            self.assertTrue(torch.isfinite(out).all(), f'Non-finite output for {name}')

    def test_pretrained_checkpoint_load_and_infer(self):
        if not os.path.exists(CKPT_PATH):
            self.skipTest('Pretrained checkpoint not present')
        from evaluate import evaluate_checkpoint
        rmses, preds, _ = evaluate_checkpoint(CKPT_PATH, SAMPLE_CSV, batch_size=30, gpu=-1)
        self.assertEqual(preds.shape, (30, 2))
        self.assertTrue(np.isfinite(rmses).all())


if __name__ == '__main__':
    unittest.main()
