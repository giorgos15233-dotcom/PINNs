# SPDX-License-Identifier: Apache-2.0
"""Framework-independent tests for online sampling and composite wall laws."""

import unittest

import numpy as np
from sympy import Symbol, lambdify, log

from dynamic_sampling import FarSeedGeometry, OffsetWallGeometry, ResolvedNearGeometry
from wall_geometry import disk_envelope_sdf
from wall_laws import C_MU, E, KAPPA, blended_epsilon, composite_u_plus


class FakeAirfoil:
    def sample_interior(self, n_points, rand_seed=0, **kwargs):
        rng = np.random.default_rng(rand_seed)
        x = rng.uniform(-0.32, 1.28, (n_points, 1))
        y = rng.uniform(-0.5, 0.5, (n_points, 1))
        sdf, _, _ = disk_envelope_sdf(x, y)
        return {
            'x': x, 'y': y, 'sdf': sdf,
            'alpha': np.zeros_like(x), 'camber': np.zeros_like(x),
            'thickness': np.full_like(x, 0.12),
        }

    def sample_boundary(self, n_points, rand_seed=0, **kwargs):
        rng = np.random.default_rng(rand_seed)
        x = rng.uniform(-0.005, 1.005, (n_points, 1))
        # These are only candidate points; the real disk-envelope surface
        # and outward normals are found by repeated SDF projection.
        xi = np.clip(x, 0.0, 1.0)
        thickness = 0.6 * (
            0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2
            + 0.2843 * xi**3 - 0.1036 * xi**4
        )
        y = rng.choice([-1.0, 1.0], size=(n_points, 1)) * thickness
        return {'x': x, 'y': y}


class FakeChannel:
    def sample_interior(self, n_points, criteria=None, **kwargs):
        rng = np.random.default_rng()
        x = rng.uniform(-2.25, -0.33, (n_points, 1))
        y = rng.uniform(-2.5, 2.5, (n_points, 1))
        return {'x': x, 'y': y, 'alpha': np.zeros_like(x),
                'camber': np.zeros_like(x),
                'thickness': np.full_like(x, 0.12)}


class TestWallPhysics(unittest.TestCase):
    def test_composite_velocity_limits(self):
        variable = Symbol('eta', positive=True)
        f = lambdify(variable, composite_u_plus(variable), 'numpy')
        for eta in (1e-4, 1e-3, 1e-2):
            self.assertAlmostEqual(f(eta) / eta, 1.0, delta=0.04)
        for eta in (300.0, 1000.0):
            self.assertAlmostEqual(
                f(eta) / (np.log(E * eta) / KAPPA), 1.0, delta=0.02
            )

    def test_viscous_shear_limit_and_reverse_sign(self):
        eta = Symbol('eta', positive=True)
        factor = lambdify(eta, eta / composite_u_plus(eta), 'numpy')
        for low in (1e-3, 1e-2):
            self.assertAlmostEqual(factor(low), 1.0, delta=0.05)
        self.assertLess((-0.25) * factor(40.0), 0.0)
        self.assertGreater((+0.25) * factor(40.0), 0.0)

    def test_blended_epsilon_limits(self):
        k = Symbol('k', positive=True)
        eta = Symbol('eta', positive=True)
        expr = blended_epsilon(k, 1e-6, 0.0018685, eta)
        f = lambdify((k, eta), expr, 'numpy')
        k_value = 1e-4
        viscous = 2e-6 * k_value / 0.0018685**2
        self.assertAlmostEqual(float(f(k_value, 0.01) / viscous), 1.0, delta=1e-5)
        logarithmic = C_MU**0.75 * k_value**1.5 / (KAPPA * 0.0018685)
        self.assertAlmostEqual(float(f(k_value, 1000) / logarithmic), 1.0, delta=1e-4)


class TestDynamicSampling(unittest.TestCase):
    def test_near_rejects_unresolved_layer_and_resamples(self):
        sampler = ResolvedNearGeometry(
            FakeAirfoil(), delta_y=0.0018685, k_in=1.5e-6,
            ep_in=2.025e-8, k_wall_seed=0.006,
            ep_wall_seed=0.1, k_ref=1e-3, ep_ref=1e-2,
        )
        first = sampler.sample_interior(256)
        second = sampler.sample_interior(256)
        self.assertEqual(first['x'].shape, (256, 1))
        self.assertTrue(np.all(first['sdf'] >= 0.0018685))
        self.assertTrue(np.all(second['sdf'] >= 0.0018685))
        self.assertFalse(np.array_equal(first['x'], second['x']))
        self.assertTrue(np.all(np.isfinite(first['k_seed_log'])))

    def test_far_seed_and_wall_points_resample(self):
        far = FarSeedGeometry(FakeChannel(), None, 1.5e-6, 2.025e-8, 1e-3, 1e-2)
        f1, f2 = far.sample_interior(128), far.sample_interior(128)
        self.assertFalse(np.array_equal(f1['x'], f2['x']))
        self.assertTrue(np.allclose(f1['k_seed_log'], np.log(1.5e-3)))

        wall = OffsetWallGeometry(FakeAirfoil(), None, 0.0018685)
        w1, w2 = wall.sample_boundary(64), wall.sample_boundary(64)
        self.assertFalse(np.array_equal(w1['x'], w2['x']))
        self.assertTrue(np.all(w1['normal_distance'] > 0.8 * 0.0018685))
        self.assertTrue(np.all(w1['normal_distance'] < 1.2 * 0.0018685))
        norm = w1['normal_x']**2 + w1['normal_y']**2
        self.assertTrue(np.allclose(norm, 1.0, rtol=1e-5))
        orthogonal = w1['normal_x']*w1['tangent_x'] + w1['normal_y']*w1['tangent_y']
        self.assertTrue(np.allclose(orthogonal, 0.0, atol=1e-6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
