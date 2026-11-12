"""
Build the DDQ cloud optics files from band-resolved RRTMGP-format cloud optics.

Each DDQ wavenumber takes the cloud optical properties of the source band that contains
it, and becomes a band of its own whose limits are that wavenumber. The band limits are
copied from the gas optics file so that cloud and gas optics share the same spectral
discretization exactly; RTE requires matching band limits to within a few ULPs.

Two sources are converted:
  rrtmgp  rrtmgp-clouds-{lw,sw}-bnd.nc from rrtmgp-data
          -> rrtmgp-clouds-{lw,sw}-ddq.nc
  echam6  ECHAM6_CldOptProps_rrtmgp_{lw,sw}.nc distributed with ICON
          -> ECHAM6-rrtmgp-clouds-{lw,sw}-ddq.nc


Adapted from https://github.com/pczarnecki/ddq-implementation-sprint/blob/main/Notebooks/DDQ_cloud_class.ipynb

Usage:
  python make_cloud_optics.py --rrtmgp-data <rrtmgp-data dir> --echam6-data <dir> [--out-dir <dir>]
"""

import argparse
import hashlib
import os

import numpy as np
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = {"lw": "longwave", "sw": "shortwave"}
SOURCES = {
    "rrtmgp": {
        "src": "rrtmgp-clouds-{band}-bnd.nc",
        "out": "rrtmgp-clouds-{band}-ddq.nc",
        "title": "RRTMGP cloud optics on the DDQ {band} wavenumber grid",
        "reference": "https://github.com/earth-system-radiation/rrtmgp-data",
    },
    "echam6": {
        "src": "ECHAM6_CldOptProps_rrtmgp_{band}.nc",
        "out": "ECHAM6-rrtmgp-clouds-{band}-ddq.nc",
        "title": "ECHAM6 cloud optics (RRTMGP format) on the DDQ {band} wavenumber grid",
        "reference": "ECHAM6_CldOptProps_rrtmgp_{band}.nc distributed with ICON",
    },
}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def band_of_wavenumber(nu, limits):
    """Index of the source band with limits[:, 0] <= nu <= limits[:, 1] (first match)."""
    inside = (limits[None, :, 0] <= nu[:, None]) & (nu[:, None] <= limits[None, :, 1])
    if not inside.any(axis=1).all():
        missing = nu[~inside.any(axis=1)]
        raise ValueError(f"wavenumbers outside the source bands: {missing}")
    return inside.argmax(axis=1)


def to_ddq(src, nu, band, source, src_path, gas_path):
    """Source cloud optics subset to the band containing each DDQ wavenumber."""
    limits = src["bnd_limits_wavenumber"].values
    ibnd = band_of_wavenumber(np.asarray(nu, dtype=np.float64), limits)

    out = src.isel(nband=ibnd)
    # Only the values change; attributes stay as in the source, and the gas optics
    # precision is kept rather than the source's on-disk type
    lims = out["bnd_limits_wavenumber"].copy(data=np.stack([nu, nu], axis=1))
    lims.encoding.pop("dtype", None)
    out["bnd_limits_wavenumber"] = lims

    meta = SOURCES[source]
    out.attrs = {
        "title": meta["title"].format(band=BANDS[band]),
        "description": "Cloud optical properties of the source band that contains each DDQ "
        "wavenumber, one band per DDQ wavenumber. Both band limits equal the DDQ wavenumber "
        f"nu of {os.path.basename(gas_path)} exactly.",
        "source": os.path.basename(src_path),
        "source_md5": md5(src_path),
        "source_reference": meta["reference"].format(band=band),
        "gas_optics_file": os.path.basename(gas_path),
        "history": "Created by scripts/make_cloud_optics.py in ddq-data",
    }
    return out


def check(path, gas_path):
    """Band limits read back from disk equal the gas optics wavenumbers."""
    with xr.open_dataset(path) as c, xr.open_dataset(gas_path) as g:
        nu = g["nu"].values.astype(np.float64)
        lims = c["bnd_limits_wavenumber"].values.astype(np.float64)
        if not (np.array_equal(lims[:, 0], nu) and np.array_equal(lims[:, 1], nu)):
            raise RuntimeError(f"{path}: band limits differ from nu in {gas_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--rrtmgp-data", required=True, help="directory with rrtmgp-clouds-*-bnd.nc"
    )
    p.add_argument(
        "--echam6-data",
        required=True,
        help="directory with ECHAM6_CldOptProps_rrtmgp_*.nc",
    )
    p.add_argument(
        "--gas-optics-dir", default=REPO, help="directory with gas_optics_{lw,sw}.nc"
    )
    p.add_argument("--out-dir", default=REPO)
    args = p.parse_args()
    src_dirs = {"rrtmgp": args.rrtmgp_data, "echam6": args.echam6_data}

    for band in BANDS:
        gas_path = os.path.join(args.gas_optics_dir, f"gas_optics_{band}.nc")
        with xr.open_dataset(gas_path) as g:
            nu = g["nu"].values  # keep the gas optics precision
        for source, meta in SOURCES.items():
            src_path = os.path.join(src_dirs[source], meta["src"].format(band=band))
            out_path = os.path.join(args.out_dir, meta["out"].format(band=band))
            with xr.open_dataset(src_path) as src:
                out = to_ddq(src.load(), nu, band, source, src_path, gas_path)
            out.to_netcdf(out_path)
            check(out_path, gas_path)
            print(f"{out_path}: {out.sizes['nband']} bands")


if __name__ == "__main__":
    main()
