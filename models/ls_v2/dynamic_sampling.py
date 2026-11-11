# SPDX-License-Identifier: Apache-2.0
"""Online sampling adapters for native PhysicsNeMo pointwise constraints.

These objects deliberately implement the same sampling methods as Geometry.
The underlying samplers produce new coordinates when called; a separate cache
holds each constraint's batch until the solver's 1000-step inference event
invalidates it. The original airfoil Warp sampler does not honor an interior
``criteria`` expression, so resolved-near rejection remains explicit.
"""

import numpy as np

from wall_geometry import make_wall_points


INPUT_KEYS = ('x', 'y', 'alpha', 'camber', 'thickness')


class CachedGeometry:
    """Reuse a sampled training batch until the solver resets the cache.

    Each constraint has its own cache. The first request after reset generates
    its coordinates; subsequent requests return the same arrays. The solver
    performs reset at the actual training step where inferencers are recorded.
    """

    def __init__(self, source):
        self.source = source
        self.reset()

    @property
    def dims(self):
        return self.source.dims

    def __getattr__(self, name):
        return getattr(self.source, name)

    def reset(self):
        """Invalidate points so the next request samples a new training batch."""
        self._cache = {}

    def _sample(self, kind, n_points, *args, **kwargs):
        count = int(n_points)
        key = (kind, count)
        if key not in self._cache:
            method = getattr(self.source, kind)
            self._cache[key] = method(count, *args, **kwargs)
        return self._cache[key]

    def sample_interior(self, n_points, *args, **kwargs):
        return self._sample('sample_interior', n_points, *args, **kwargs)

    def sample_boundary(self, n_points, *args, **kwargs):
        return self._sample('sample_boundary', n_points, *args, **kwargs)


def refresh_due(step, interval):
    """Match the solver's step-numbered inference-recording cadence."""
    return int(step) > 0 and int(step) % int(interval) == 0


def _float_column(value):
    return np.asarray(value, dtype=np.float32).reshape(-1, 1)


def _unit_area(count):
    # Equal pointwise integration weight, matching the previous NumPy losses.
    return np.ones((count, 1), dtype=np.float32)


class ResolvedNearGeometry:
    """Dynamically reject samples in the unresolved wall strip, sdf < dy."""

    dims = ['x', 'y']

    def __init__(self, source, delta_y, k_in, ep_in, k_wall_seed,
                 ep_wall_seed, k_ref, ep_ref, seed=1701):
        if delta_y <= 0:
            raise ValueError('delta_y must be positive')
        self.source = source
        self.delta_y = float(delta_y)
        self.k_in, self.ep_in = float(k_in), float(ep_in)
        self.k_wall_seed, self.ep_wall_seed = float(k_wall_seed), float(ep_wall_seed)
        self.k_ref, self.ep_ref = float(k_ref), float(ep_ref)
        self._seed = int(seed)

    def sample_interior(self, n_points, bounds=None, criteria=None,
                        parameterization=None, compute_sdf_derivatives=False,
                        quasirandom=False):
        if criteria is not None or bounds is not None:
            raise ValueError('ResolvedNearGeometry uses its own exact SDF rejection')
        count = int(n_points)
        accepted = {key: [] for key in (*INPUT_KEYS, 'sdf')}
        n_collected = 0
        for _ in range(20):
            # Collect a small oversample rather than filtering a fixed pool.
            required = count - n_collected
            raw = self.source.sample_interior(
                n_points=max(128, int(required * 1.25) + 16),
                rand_seed=self._seed,
                parameterization=parameterization,
                compute_sdf_derivatives=False,
                quasirandom=False,
            )
            self._seed = (self._seed + 104729) % 2147483647
            sdf = np.asarray(raw['sdf']).reshape(-1)
            good = np.flatnonzero(np.isfinite(sdf) & (sdf >= self.delta_y))[:required]
            if good.size == 0:
                continue
            for key in accepted:
                accepted[key].append(_float_column(raw[key])[good])
            n_collected += good.size
            if n_collected == count:
                break
        if n_collected != count:
            raise RuntimeError(f'Near-field online sampling accepted {n_collected}/{count}')

        out = {key: np.concatenate(blocks) for key, blocks in accepted.items()}
        distance = out['sdf'].astype(np.float64)
        k_seed = self.k_in + (self.k_wall_seed - self.k_in) * np.exp(-distance / 0.07)
        ep_seed = self.ep_in + (self.ep_wall_seed - self.ep_in) * np.exp(-distance / 0.035)
        out['k_seed_log'] = np.log(k_seed / self.k_ref).astype(np.float32)
        out['ep_seed_log'] = np.log(ep_seed / self.ep_ref).astype(np.float32)
        out['area'] = _unit_area(count)
        return out


class FarSeedGeometry:
    """Online far-field samples carrying the uniform initialization targets."""

    dims = ['x', 'y']

    def __init__(self, channel, far_criteria, k_in, ep_in, k_ref, ep_ref):
        self.channel = channel
        self.far_criteria = far_criteria
        self.log_k = float(np.log(k_in / k_ref))
        self.log_ep = float(np.log(ep_in / ep_ref))

    def sample_interior(self, n_points, bounds=None, criteria=None,
                        parameterization=None, compute_sdf_derivatives=False,
                        quasirandom=False):
        if criteria is not None or bounds is not None:
            raise ValueError('The far region is already defined in FarSeedGeometry')
        raw = self.channel.sample_interior(
            int(n_points), criteria=self.far_criteria,
            parameterization=parameterization,
            quasirandom=False,
        )
        out = {key: _float_column(raw[key]) for key in INPUT_KEYS}
        n = len(out['x'])
        out['k_seed_log'] = np.full((n, 1), self.log_k, dtype=np.float32)
        out['ep_seed_log'] = np.full((n, 1), self.log_ep, dtype=np.float32)
        out['area'] = _unit_area(n)
        return out


class OffsetWallGeometry:
    """Continuously regenerate points displaced into the fluid by delta_y."""

    dims = ['x', 'y']

    def __init__(self, source, parameterization, delta_y,
                 k_wall_seed=None, ep_wall_seed=None,
                 k_ref=None, ep_ref=None, seed=2303):
        self.source = source
        self.parameterization = parameterization
        self.delta_y = float(delta_y)
        self._seed = int(seed)
        self.k_seed = None if k_wall_seed is None else float(np.log(k_wall_seed / k_ref))
        self.ep_seed = None if ep_wall_seed is None else float(np.log(ep_wall_seed / ep_ref))

    def sample_boundary(self, n_points, criteria=None, parameterization=None,
                        quasirandom=False):
        if criteria is not None:
            raise ValueError('OffsetWallGeometry already restricts to the airfoil')
        out = make_wall_points(
            self.source,
            parameterization if parameterization is not None else self.parameterization,
            count=int(n_points), delta_y=self.delta_y,
            seed=self._seed, verbose=False,
        )
        self._seed = (self._seed + 104729) % 2147483647
        n = len(out['x'])
        if self.k_seed is not None:
            out['k_seed_log'] = np.full((n, 1), self.k_seed, dtype=np.float32)
            out['ep_seed_log'] = np.full((n, 1), self.ep_seed, dtype=np.float32)
        out['area'] = _unit_area(n)
        return out
