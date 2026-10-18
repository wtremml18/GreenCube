"""Load pinned public measurements and optical products without replicated labels."""
from pathlib import Path
import datetime as dt
import io
import h5py
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import varmats_from_mat

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'auroral_data/data'
EVENTS = ['20230210', '20230212', '20230304a', '20230304b', '20230314', '20230319']


def plane(lon, lat, lon0=-147.43, lat0=65.12, radius=6481.):
    dl = (np.asarray(lon) - lon0 + 180) % 360 - 180
    return np.stack([np.deg2rad(dl) * radius * np.cos(np.deg2rad(lat0)), np.deg2rad(np.asarray(lat) - lat0) * radius], axis=-1)


def electric_output_scale(x):
    """Convert chart potential derivatives to physical east/north components.

    At constant altitude, dx=R*cos(lat0)*dlon, ds_east=R*cos(lat)*dlon.
    Therefore E_chart_x=cos(lat)/cos(lat0)*E_east. Noise transforms likewise.
    """
    lat = 65.12 + np.rad2deg(np.asarray(x)[..., 1] / 6481.)
    return np.stack([np.cos(np.deg2rad(65.12)) / np.cos(np.deg2rad(lat)), np.ones_like(lat)], axis=-1)


def image_ridge(xy, intensity, xlim=(-180., 180.), ylim=(-100., 420.)):
    """Image-only quadratic brightness ridge; proxy, not a measured flow boundary."""
    x, y = xy[..., 0].ravel(), xy[..., 1].ravel()
    image = np.asarray(intensity).ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(image) & (y >= ylim[0]) & (y <= ylim[1])
    xs, ys = [], []
    edges = np.linspace(*xlim, 26)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = valid & (x >= left) & (x < right)
        if mask.sum() < 20:
            continue
        val = image[mask]
        w = np.maximum(val - np.quantile(val, .8), 0.) ** 2
        if w.sum() <= 0:
            continue
        xs.append(np.sum(x[mask] * w) / w.sum())
        ys.append(np.sum(y[mask] * w) / w.sum())
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 8:
        raise ValueError('Insufficient image support for a ridge')
    p = np.polynomial.Polynomial.fit(xs, ys, 2).convert()
    # Fixed robust reweighting, using optical coordinates only.
    for _ in range(4):
        residual = ys - p(xs)
        sigma = max(1., 1.4826 * np.median(abs(residual - np.median(residual))))
        weight = np.minimum(1., 2.5 * sigma / np.maximum(abs(residual), 1e-12))
        p = np.polynomial.Polynomial.fit(xs, ys, 2, w=np.sqrt(weight)).convert()
    return p.coef, np.column_stack([xs, ys]), float(np.sqrt(np.mean((p(xs) - ys) ** 2)))


def radar_event(tag, sigma_limit=30.):
    with h5py.File(DATA / f'dasc_{tag}.h5') as f:
        utc = float(f['Time/Unix'][()])
        image_xy = plane(f['Coordinates/Longitude'][:], f['Coordinates/Latitude'][:])
        image = f['Optical/Blue'][:]
        color_times = f['Optical/Times'][:]
        image_alt = float(f['Coordinates/Altitude'][()]) / 1000
    polynomial, ridge, ridge_rmse = image_ridge(image_xy, image)
    with h5py.File(DATA / f'pfisr_{tag}.h5') as f:
        time = f['Time/UnixTime'][:]
        containing = np.flatnonzero((time[:, 0] <= utc) & (time[:, 1] >= utc))
        if len(containing) != 1:
            raise ValueError(f'{tag}: expected one radar interval containing image')
        i = int(containing[0])
        alt = f['VvelsGeoCoords/Altitude'][:]
        z = int(np.argmin(abs(np.nanmean(alt, axis=1) - image_alt)))
        assert np.allclose(alt[z], image_alt, atol=.01)
        x = plane(f['VvelsGeoCoords/Longitude'][z], f['VvelsGeoCoords/Latitude'][z])
        y = f['VvelsGeoCoords/ElectricField'][i, z, :, :2] * 1000
        cov = f['VvelsGeoCoords/CovarianceE'][i, z, :, :2, :2] * 1e6
        chi2 = f['ProcessingParams/Chi2'][i]
        number = f['ProcessingParams/NumPoints'][i]
        finite = np.isfinite(x).all(1) & np.isfinite(y).all(1) & np.isfinite(cov).all((1, 2))
        positive = np.zeros(len(y), bool)
        positive[finite] = (np.linalg.eigvalsh(cov[finite]) > 0).all(1)
        std = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0))
        keep = finite & positive & (std.max(1) < sigma_limit) & (number >= 30) & (chi2 > 0) & (chi2 < 3)
        metadata = dict(event=tag, utc=dt.datetime.fromtimestamp(utc, dt.timezone.utc).isoformat(),
                        image_color_unix_times=color_times.tolist(), radar_unix_interval=time[i].tolist(),
                        radar_image_time_offsets_s=(time[i] - utc).tolist(), radar_time_index=i,
                        altitude_km=image_alt, retained_bins=np.flatnonzero(keep).tolist(), total_bins=len(y),
                        sigma_limit_mVm=sigma_limit, image_ridge_coefficients=polynomial.tolist(), image_ridge_residual_km=ridge_rmse)
    scale = electric_output_scale(x[keep])
    physical = y[keep].copy()
    chart_y = physical / scale
    chart_cov = cov[keep] / scale[:, :, None] / scale[:, None, :]
    metadata['east_component_chart_to_physical_scale_range'] = [float(scale[:,0].min()),float(scale[:,0].max())]
    return dict(name=tag, x=x[keep], y=chart_y, physical_y=physical, noise=chart_cov, output_scale=scale, ids=np.flatnonzero(keep),
                polynomial=polynomial, image_xy=image_xy, image=image, ridge=ridge, metadata=metadata, flow=False)


def decode_rocket_metadata(f):
    """Decode this pinned file's MCOS payload, not a general MATLAB decoder.

    Its datetime payload is Unix milliseconds with a sub-millisecond residual.
    Cross-check all 195 values against the separately stored elapsed seconds
    and the documented 2017-03-02 07:50 UTC launch before using the result.
    """
    b = f['__function_workspace__'].tobytes()
    streams = varmats_from_mat(io.BytesIO(b' ' * 124 + b'\x00\x01IM' + b[8:]))
    structure = loadmat(streams[0][1], struct_as_record=False, squeeze_me=True)['__function_workspace__'].ravel()[0]
    payload = structure.MCOS['_ObjectMetadata'][0]
    unix = np.real(payload[3]) / 1000.
    launch = dt.datetime(2017, 3, 2, 7, 50, tzinfo=dt.timezone.utc).timestamp()
    if unix.shape != f['time'].shape or not np.allclose(unix - f['time'], launch, rtol=0, atol=.001):
        raise ValueError('MCOS times do not agree with documented launch and elapsed time')
    strings = [v[5:].tobytes().decode('utf-16le')[:int(v[4])] for v in [payload[2], payload[4]]]
    return unix, strings


def rocket_event(frame=90, sigma=100., max_time_offset=None):
    a = loadmat(DATA / 'particles_ig.mat', simplify_cells=True)
    f = loadmat(DATA / 'flow_data_ig.mat', simplify_cells=True)
    unix, source_notes = decode_rocket_metadata(f)
    image_unix = dt.datetime(*map(int, a['outputdate'][frame]), tzinfo=dt.timezone.utc).timestamp()
    lon0, lat0 = float(np.mean(a['mlon'])), float(np.mean(a['mlat']))
    lon, lat = np.meshgrid(a['mlon'], a['mlat'], indexing='ij')
    image_xy = plane(lon, lat, lon0, lat0, 6471.)
    image = a['Qit'][:, :, frame]
    x = plane(f['mlon'], f['mlat'], lon0, lat0, 6471.)
    y = np.column_stack([f['v_geom_east'], f['v_geom_north']])
    xl = (float(image_xy[..., 0].min()), float(image_xy[..., 0].max()))
    yl = (float(image_xy[..., 1].min()), float(image_xy[..., 1].max()))
    keep = (x[:, 0] >= xl[0]) & (x[:, 0] <= xl[1]) & (x[:, 1] >= yl[0]) & (x[:, 1] <= yl[1])
    if max_time_offset is not None:
        keep &= abs(unix - image_unix) <= max_time_offset
    polynomial, ridge, residual = image_ridge(image_xy, image, xl, yl)
    return dict(name='isinglass', x=x[keep], y=y[keep], physical_y=y[keep], output_scale=np.ones_like(y[keep]), noise=np.tile(np.eye(2)*sigma**2,(keep.sum(),1,1)), ids=np.flatnonzero(keep),
                polynomial=polynomial, image_xy=image_xy, image=image, ridge=ridge, flow=True,
                metadata=dict(event='isinglass', optical_datetime=a['outputdate'][frame].tolist(), frame=frame,
                              retained_samples=int(keep.sum()), original_samples=len(y), original_row_ids=np.flatnonzero(keep).tolist(),
                              elapsed_seconds_range=[float(f['time'][keep].min()),float(f['time'][keep].max())],
                              rocket_unix_range=[float(unix[keep].min()),float(unix[keep].max())],
                              rocket_image_offset_range_s=[float((unix-image_unix)[keep].min()),float((unix-image_unix)[keep].max())],
                              maximum_time_offset_filter_s=max_time_offset, source_notes=source_notes,
                              assumed_measurement_sigma_ms=sigma, image_ridge_coefficients=polynomial.tolist(), image_ridge_residual_km=residual,
                              limitation='Static optical inversion applied to moving track. Nominal 100 km local geomagnetic plane; upstream vector rotation is approximate, per source note. No independent track.'))
