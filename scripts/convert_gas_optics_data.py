"""Flatten trained fax gas optics datatrees into the DDQ gas optics files for RTE.

Reads one trained datatree per band (LW and SW) and writes gas_optics_lw.nc and
gas_optics_sw.nc in the flat layout for Fortran DDQ gas optics.

Usage:
  python convert_gas_optics_data.py --lw gas_optics_DDQ_LW.nc --sw gas_optics_DDQ_SW.nc [--out-dir DIR]
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

LIGHT_SPEED = 2.99792458e8  # m s^-1
CM_TO_M = 100.0
KAYSER_TO_HZ = LIGHT_SPEED * CM_TO_M

FAX_GROUP = "ShiftedReciprocalLaurent_Rational"
CONT_GROUP = "both_continuum_MT_CKD_4_3"

# Model variables the Fortran reader does not consume; the file layout is fixed.
DROP_VARS = [
    "temperature_coeffs",
    "t_order",
    "x_p_range",
    "x_t_range",
    "bound",
]

TERM_NAMES = ["p00", "p10", "p20", "p01"]

LINE_RENAME = {
    "xsec0": "fax_sigma0",
    "p_order": "fax_p_nterms",
    "ref_pressure": "fax_p0",
    "ref_temperature": "fax_T0",
    "ref_vmr": "fax_vmr0",
    "self_scaling": "fax_S",
    "pressure_coeffs": "fax_c",
    "species": "fax_species_names",
}

CONT_RENAME = {
    "ref_pressure": "mtckd_p0",
    "ref_temperature": "mtckd_T0",
    "species": "mtckd_species_names",
    "self_absco_ref": "mtckd_cself",
    "self_texp": "mtckd_n",
    "for_absco_ref": "mtckd_cfrgn",
}

TRANSPOSE_ORDER = (
    "nu",
    "fax_species_names",
    "fax_p_nterms",
    "fax_t_order",
    "xsec_species_names",
    "xsec_nterms",
    "mtckd_species_names",
)

LW_ORDER = [
    "nu",
    "weights",
    "fax_nspecies",
    "fax_species_names",
    "fax_p0",
    "fax_T0",
    "fax_S",
    "fax_vmr0",
    "fax_sigma0",
    "fax_p_nterms",
    "fax_c",
    "fax_t_order",
    "fax_a",
    "fax_b",
    "xsec_nspecies",
    "xsec_species_names",
    "xsec_nterms",
    "xsec_p",
    "mtckd_nspecies",
    "mtckd_species_names",
    "mtckd_p0",
    "mtckd_T0",
    "mtckd_cself",
    "mtckd_n",
    "mtckd_cfrgn",
]

SW_ORDER = [
    "nu",
    "weights",
    "solar_spectral_irradiance",
    *LW_ORDER[2:],
    "rayleigh_xsec",
]

PRESSURE_TERMS = (
    "Order: c0, shift, lin. With x = (p/fax_p0)*(1 + vmr*fax_S)/"
    "(1 + fax_vmr0*fax_S) and w = (x + shift)/(1 + shift) the pressure factor "
    "is lin*x + (1 - lin)/Q(w), where Q(w) = c0/w + (1 - c0)*w is 1 at x = 1."
)

VARIABLE_ATTRS = {
    "nu": {"units": "cm^-1", "description": "DDQ wavenumber grid."},
    "weights": {"units": "", "description": "DDQ weights."},
    "solar_spectral_irradiance": {
        "units": "W m^-2",
        "description": "Top-of-atmosphere solar spectral irradiance. weights * solar_spectral_irradiance integrates to 1361 W m^-2.",
    },
    "fax_nspecies": {"units": "1", "description": "Fax species index."},
    "fax_species_names": {
        "units": "",
        "description": "Species names for which fax model is available.",
    },
    "fax_p0": {"units": "Pa", "description": "Reference pressure for fax_sigma0."},
    "fax_T0": {"units": "K", "description": "Reference temperature for fax_sigma0."},
    "fax_S": {"units": "1", "description": "Self-broadening pressure scaling factor."},
    "fax_vmr0": {
        "units": "1",
        "description": "Reference volume mixing ratio for fax_sigma0.",
    },
    "fax_sigma0": {
        "units": "m^2 molecule^-1",
        "description": "Reference cross section at fax_p0, fax_T0 and fax_vmr0.",
    },
    "fax_p_nterms": {"units": "1", "description": PRESSURE_TERMS},
    "fax_c": {"units": "", "description": PRESSURE_TERMS},
    "fax_t_order": {
        "units": "1",
        "description": "Polynomial term index for the temperature rational function. Order: const, x, x^2.",
    },
    "fax_a": {
        "units": "",
        "description": "Polynomial coefficients for numerator of Temperature rational function.",
    },
    "fax_b": {
        "units": "",
        "description": "Polynomial coefficients for denominator of Temperature rational function.",
    },
    "xsec_nspecies": {"units": "1", "description": "XFIT species index."},
    "xsec_species_names": {
        "units": "",
        "description": "Species names for which XFIT model is available.",
    },
    "xsec_nterms": {
        "units": "1",
        "description": "XFIT coefficients index. order: p0 + p1 * T + p2 * T^2 + p3 * pressure",
    },
    "xsec_p": {
        "units": "result: m^2 molecule^-1, T, pressure in SI units",
        "description": "XFIT polynomial coefficients. order: p0 + p1 * T + p2 * T^2 + p3 * pressure",
    },
    "mtckd_nspecies": {"units": "1", "description": "MT_CKD species index."},
    "mtckd_species_names": {
        "units": "",
        "description": "Species names for which MT_CKD4.3 model is available.",
    },
    "mtckd_p0": {
        "units": "Pa",
        "description": "Reference pressure for the MT_CKD continuum.",
    },
    "mtckd_T0": {
        "units": "K",
        "description": "Reference temperature for the MT_CKD continuum.",
    },
    "mtckd_cself": {
        "units": "m^2 molecule^-1 cm",
        "description": "Self-continuum coefficient.",
    },
    "mtckd_n": {"units": "1", "description": "Self-continuum temperature exponent."},
    "mtckd_cfrgn": {
        "units": "m^2 molecule^-1 cm",
        "description": "Foreign-continuum coefficient.",
    },
    "rayleigh_xsec": {
        "units": "m^2 molecule^-1",
        "description": "Rayleigh scattering cross section.",
    },
}


def clear_all_attrs(ds):
    ds = ds.copy()
    ds.attrs.clear()
    for var in ds.variables:
        ds[var].attrs.clear()
    return ds


def pad_species_names(da, width=32):
    """Right-pad a string DataArray with spaces to a fixed width (Fortran-style)."""
    return da.str.pad(width=width, side="right", fillchar=" ")


def apply_variable_attrs(ds, metadata):
    ds = ds.copy()
    for name, attrs in metadata.items():
        if name in ds.variables:
            ds[name].attrs.update(attrs)
    return ds


def hz_to_kayser(hz):
    """Convert frequencies in Hz to wavenumbers in kaysers (cm^-1)."""
    return hz / KAYSER_TO_HZ


def flatten(path, band):
    """One band of a trained fax datatree as the flat Fortran layout."""
    data = xr.open_datatree(path).copy()

    lines = data[FAX_GROUP].to_dataset().rename(LINE_RENAME)
    lines["fax_p_nterms"] = ("fax_p_nterms", range(lines.sizes["fax_p_nterms"]))
    lines["fax_a"] = (
        lines["temperature_coeffs"]
        .isel(t_order=[0, 1, 2])
        .rename({"t_order": "fax_t_order"})
        .assign_coords(fax_t_order=[0, 1, 2])
    )
    ones = xr.ones_like(lines["temperature_coeffs"].isel(t_order=0, drop=True))
    ones = ones.expand_dims(fax_t_order=[0])
    rest = (
        lines["temperature_coeffs"]
        .isel(t_order=[3, 4])
        .rename({"t_order": "fax_t_order"})
        .assign_coords(fax_t_order=[1, 2])
    )
    lines["fax_b"] = xr.concat([ones, rest], dim="fax_t_order")
    lines = lines.drop_vars(DROP_VARS, errors="ignore")

    cont = data[CONT_GROUP].to_dataset().rename(CONT_RENAME)

    xsec = (
        xr.concat(
            [data["XFIT"].to_dataset()[name] for name in TERM_NAMES],
            dim="xsec_nterms",
        )
        .assign_coords(xsec_nterms=range(4))
        .to_dataset()
        .rename({"species": "xsec_species_names", "p00": "xsec_p"})
    )

    flat = xr.merge([lines, cont, xsec])
    flat["frequency"] = hz_to_kayser(flat["frequency"].astype(np.float64))
    flat = flat.rename({"frequency": "nu"})
    flat["weights"] = ("nu", hz_to_kayser(data["DDQ"]["weights_hz"].values))

    if band == "SW":
        flat["solar_spectral_irradiance"] = (
            "nu",
            data["DDQ"]["spectral_solar_irradiance"].values * LIGHT_SPEED * CM_TO_M,
        )
        flat["rayleigh_xsec"] = ("nu", data["DDQ"]["xsec_rayleigh"].values)

    flat = clear_all_attrs(flat).transpose(*TRANSPOSE_ORDER)
    flat = flat.assign_coords(
        fax_nspecies=("fax_species_names", range(flat.sizes["fax_species_names"])),
        xsec_nspecies=("xsec_species_names", range(flat.sizes["xsec_species_names"])),
        mtckd_nspecies=(
            "mtckd_species_names",
            range(flat.sizes["mtckd_species_names"]),
        ),
    )
    flat = (
        flat.swap_dims({"fax_species_names": "fax_nspecies"})
        .swap_dims({"xsec_species_names": "xsec_nspecies"})
        .swap_dims({"mtckd_species_names": "mtckd_nspecies"})
        .reset_coords(
            ["fax_species_names", "xsec_species_names", "mtckd_species_names"]
        )
    )
    flat = flat[LW_ORDER if band == "LW" else SW_ORDER]
    flat["mtckd_p0"] = flat["mtckd_p0"][0]
    flat["mtckd_T0"] = flat["mtckd_T0"][0]
    for names in ["xsec_species_names", "fax_species_names", "mtckd_species_names"]:
        flat[names] = pad_species_names(flat[names].str.lower()).astype("S32")
        flat[names].encoding["dtype"] = "S1"

    flat = apply_variable_attrs(flat, VARIABLE_ATTRS)
    return flat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lw", required=True, type=Path, help="Trained fax datatree, LW"
    )
    parser.add_argument(
        "--sw", required=True, type=Path, help="Trained fax datatree, SW"
    )
    parser.add_argument(
        "--out-dir", default=Path(__file__).resolve().parents[1], type=Path
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for band, path in (("LW", args.lw), ("SW", args.sw)):
        out = args.out_dir / f"gas_optics_{band.lower()}.nc"
        flatten(path, band).to_netcdf(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
