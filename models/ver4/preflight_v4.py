# SPDX-License-Identifier: Apache-2.0
"""CPU preflight for v4; framework training verification is separate."""
import pathlib
import unittest

import yaml

from preflight_v2 import TestDynamicSampling, TestWallPhysics  # noqa: F401


class TestV4Config(unittest.TestCase):
    def test_gradient_aggregator_contract(self):
        root = pathlib.Path(__file__).resolve().parent
        config = yaml.safe_load((root / 'conf' / 'config.yaml').read_text())
        self.assertIn({'loss': 'lr_annealing'}, config['defaults'])
        self.assertEqual(config['loss']['ref_key'], 'inlet_u')
        self.assertGreaterEqual(config['loss']['update_freq'], 1)
        self.assertTrue(0 < config['loss']['alpha'] <= 1)
        self.assertGreater(config['loss']['eps'], 0)
        self.assertFalse(config['cuda_graphs'])
        self.assertNotEqual(config['network_dir'], 'network_checkpoint_ls_v2')

    def test_all_stage_reference(self):
        source = (pathlib.Path(__file__).resolve().parent /
                  'airfoil_ke_wf.py').read_text()
        self.assertIn("domain.add_constraint(inlet, 'inlet')", source)
        self.assertIn("outvar={'u': u_inf, 'v': 0.0", source)
        self.assertIn('LRAnnealing', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
