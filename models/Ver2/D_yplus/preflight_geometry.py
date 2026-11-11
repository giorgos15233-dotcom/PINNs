"""Fast, CPU-only consistency checks for disk-envelope point projections."""
import numpy as np
from wall_geometry import disk_envelope_sdf, make_wall_points


class FakeSampler:
    def sample_boundary(self, n_points, rand_seed, parameterization):
        rng = np.random.default_rng(rand_seed)
        return {
            'x': rng.uniform(-0.01, 1.01, (n_points, 1)),
            'y': rng.choice([-1., 1.], (n_points, 1)) * 0.048,
        }


p = make_wall_points(FakeSampler(), None, 512, 0.001868567)
sdf, _, _ = disk_envelope_sdf(p['x'], p['y'])
norm = np.hypot(p['normal_x'], p['normal_y'])
tnorm = np.hypot(p['tangent_x'], p['tangent_y'])
assert p['x'].shape == (512,1)
assert np.all(np.abs(sdf-0.001868567) < 0.0004)
assert np.allclose(norm, 1, atol=1e-6)
assert np.allclose(tnorm, 1, atol=1e-6)
assert np.allclose(p['normal_x']*p['tangent_x'] + p['normal_y']*p['tangent_y'], 0, atol=1e-6)
print('PASS: wall P is off-wall, normals and tangents are orthonormal')
