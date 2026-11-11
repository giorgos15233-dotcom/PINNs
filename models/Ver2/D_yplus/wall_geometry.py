"""NACA0012 wall points consistent with the original Warp disk-union SDF.

The original NVIDIA-derived AirfoilInChannel geometry represents the airfoil as
501 disks: min_i(distance_to_center_i - thickness_i). Its sample_boundary()
can return points hidden inside another disk. Project and validate against the
very same disk envelope instead of treating the raw samples as exact boundary.

This helper intentionally supports only alpha=0, camber=0, thickness=0.12.
Parametric studies require re-computing the disk centers after rotation/camber.
"""
import numpy as np


def disk_envelope_sdf(x, y, chord=1.0, thickness=0.12, resolution=501):
    shape = np.shape(x)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xc = chord * np.arange(resolution, dtype=np.float64) / resolution
    xi = xc / chord
    radii = 5.0 * thickness * chord * (
        0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2
        + 0.2843 * xi**3 - 0.1036 * xi**4
    )
    radii = np.maximum(radii, 0.0)
    distance = np.hypot(x[:, None] - xc[None, :], y[:, None])
    raw = distance - radii[None, :]
    closest = np.argmin(raw, axis=1)
    sdf = raw[np.arange(len(x)), closest]
    norm = np.maximum(distance[np.arange(len(x)), closest], 1e-14)
    nx = (x - xc[closest]) / norm
    ny = y / norm
    return (sdf.reshape(shape), nx.reshape(shape), ny.reshape(shape))


def make_wall_points(geometry, parameterization, count, delta_y, seed=23):
    """Sample, project and return P at positive outward normal distance delta_y.

    Uses the unchanged geometry.sample_boundary() as initial candidate source.
    Raises rather than silently accepting points inside the unresolved wall.
    """
    accepted = []
    total = 0
    for attempt in range(14):
        n = max(2 * count, 512)
        raw = geometry.sample_boundary(
            n_points=n, rand_seed=seed + attempt * 1237,
            parameterization=parameterization,
        )
        x = np.asarray(raw['x'], dtype=np.float64).reshape(-1)
        y = np.asarray(raw['y'], dtype=np.float64).reshape(-1)
        total += len(x)
        # Repeated closest-disk projection to the disk-union zero level set.
        for _ in range(6):
            ds, nx, ny = disk_envelope_sdf(x, y)
            x -= ds * nx
            y -= ds * ny
        ds, nx, ny = disk_envelope_sdf(x, y)
        xp, yp = x + delta_y * nx, y + delta_y * ny
        dsp, _, _ = disk_envelope_sdf(xp, yp)
        # Avoid pathological high-curvature or ambiguous disk-intersection points.
        good = (np.abs(ds) < 3e-5) & (dsp > 0.80 * delta_y) & (
            dsp < 1.20 * delta_y) & (np.hypot(nx, ny) > 0.99)
        # Tangent orientation from leading to trailing edge, for attached flow.
        sign = np.where(ny >= 0, 1.0, -1.0)
        tx, ty = sign * ny, -sign * nx
        for idx in np.flatnonzero(good):
            accepted.append((xp[idx], yp[idx], -nx[idx], -ny[idx], tx[idx], ty[idx], dsp[idx]))
            if len(accepted) >= count:
                break
        if len(accepted) >= count:
            break
    if len(accepted) != count:
        raise RuntimeError(f'Only {len(accepted)}/{count} physically valid P points after {total} candidates')
    vals = np.asarray(accepted, dtype=np.float32)
    fields = ('x', 'y', 'normal_x', 'normal_y', 'tangent_x', 'tangent_y', 'normal_distance')
    out = {key: vals[:, i:i+1] for i, key in enumerate(fields)}
    # Use the *actual* closest-wall distance at P (the disk-envelope SDF).
    out['alpha'] = np.zeros((count, 1), np.float32)
    out['camber'] = np.zeros((count, 1), np.float32)
    out['thickness'] = np.full((count, 1), 0.12, np.float32)
    # Stored normal_x/y are from FLUID INTO SOLID, matching AirfoilInChannel.
    print(f'Wall points P: {count} valid / {total} candidates; '
          f'distance range=[{out["normal_distance"].min():.8g}, '
          f'{out["normal_distance"].max():.8g}] m')
    return out
