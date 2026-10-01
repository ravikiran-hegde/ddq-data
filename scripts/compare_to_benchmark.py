"""DDQ and RRTMGP fluxes against a line-by-line ARTS benchmark.

Models compared:

    ddq       DDQ quadrature with the FAX gas optics
    rrtmgp    RRTMGP with 256 longwave, 224 shortwave goints
    arts      ARTS line-by-line on ~1e5 frequencies, used as the reference

Input files:

    rfmip-states.nc           RFMIP atmospheres, from rte-examples
    ddq-{lw,sw}-rfmip.nc      written by the ctest rte-examples driver
    rrtmgp-{lw,sw}-rfmip.nc   written by the ctest rte-examples driver
    arts-{lw,sw}-rfmip.nc     benchmark, from rte-examples/benchmark

Set the three paths below and run:
    python compare_to_benchmark.py
"""

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import LogNorm

# --- configuration ---

HERE = Path(__file__).resolve().parent

# Fluxes written by the rte-examples driver (ddq-*.nc, rrtmgp-*.nc) and the
# ARTS line-by-line benchmark (arts-*.nc) both live here.
BUILD_DIR = HERE / "../../rte-rrtmgp/build"
FLUX_DIR = BUILD_DIR / "examples/rte-examples"
BENCHMARK_DIR = BUILD_DIR / "rte-examples-data/benchmark"
STATES_FILE = BUILD_DIR / "rte-examples-data/rfmip-states.nc"

OUT_DIR = HERE / "figures"

REFERENCE = "arts"
MODELS = ["ddq", "rrtmgp"]
LABELS = {"arts": "ARTS", "ddq": "DDQ", "rrtmgp": "RRTMGP"}

# Profiles and flux distributions use one RFMIP experiment; forcing is the
# difference between two. Experiment order is fixed by RFMIP.
VARIANT = 0
VARIANT_LABEL = "Present day"
FORCING_VARIANTS = (2, 1)
FORCING_LABEL = "4xCO2 - Pre-industrial"

SOURCES = {
    "arts": (BENCHMARK_DIR / "arts-lw-rfmip.nc", BENCHMARK_DIR / "arts-sw-rfmip.nc"),
    "ddq": (FLUX_DIR / "ddq-lw-rfmip.nc", FLUX_DIR / "ddq-sw-rfmip.nc"),
    "rrtmgp": (FLUX_DIR / "rrtmgp-lw-rfmip.nc", FLUX_DIR / "rrtmgp-sw-rfmip.nc"),
}

needed = [STATES_FILE, *(p for paths in SOURCES.values() for p in paths)]
missing = [p for p in needed if not p.exists()]
if missing:
    raise SystemExit(
        "cannot find:\n  "
        + "\n  ".join(str(p) for p in missing)
        + "\nedit FLUX_DIR, BENCHMARK_DIR and STATES_FILE at the top of this file."
    )

# %% Plotting setup
# --- physics ---

GRAVITY = 9.80665  # m s-2
CP_AIR = 1004.0  # J kg-1 K-1
SECONDS_PER_DAY = 86400.0
TSI_REF = 1361.0  # W m-2, the irradiance the fluxes were computed at

FLUX_UNIT = r"W m$^{-2}$"
HEATING_UNIT = r"K day$^{-1}$"


def heating_rate(pressure: xr.DataArray, net_flux: xr.DataArray) -> xr.DataArray:
    """Heating rate in K day-1, from the divergence of a net flux profile."""
    dp = pressure.diff(dim="level").rename({"level": "layer"})
    dflux = net_flux.diff(dim="level").rename({"level": "layer"})
    divergence = xr.where(dp != 0, dflux / dp, np.nan)
    return -divergence * GRAVITY / CP_AIR * SECONDS_PER_DAY


def load_model(paths: tuple[Path, ...]) -> xr.Dataset:
    """One model's fluxes on common names, plus net flux and heating rate."""
    parts = []
    for path in paths:
        ds = xr.open_dataset(path)
        parts.append(ds[[v for v in CORE_VARS if v in ds.variables]].load())
        ds.close()
    flux = xr.merge(parts)

    # The examples run at a fixed solar constant; rescale to the irradiance
    # the RFMIP atmospheres specify.
    solar_scale = atm["total_solar_irradiance"] / TSI_REF
    flux["sw_flux_dn"] = flux["sw_flux_dn"] * solar_scale
    flux["sw_flux_up"] = flux["sw_flux_up"] * solar_scale

    for band in ("lw", "sw"):
        flux[f"{band}_flux_net"] = flux[f"{band}_flux_dn"] - flux[f"{band}_flux_up"]
        flux[f"{band}_heating_rate"] = heating_rate(
            atm["pres_level"], flux[f"{band}_flux_net"]
        )
    return flux


# --- colours ---

BAND_PALETTE = {"Longwave": "tab:blue", "Shortwave": "tab:orange"}
BIAS_COLOR = "black"
RMSD_COLOR = "tab:red"


def save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


# --- load ---

atm = xr.load_dataset(STATES_FILE)

# Pressure runs top-down in the RFMIP files; check rather than assume.
_top_first = float(atm["pres_level"][0, 0]) < float(atm["pres_level"][-1, 0])
TOA, SFC = (0, -1) if _top_first else (-1, 0)

CORE_VARS = ("lw_flux_dn", "lw_flux_up", "sw_flux_dn", "sw_flux_up")


ALL_MODELS = [REFERENCE, *MODELS]
fluxes = xr.concat(
    [load_model(SOURCES[m]) for m in ALL_MODELS],
    dim=xr.IndexVariable("model", ALL_MODELS),
)

# Forcing is a difference between two experiments, so a bias shared by both
# cancels.
_perturbed, _control = FORCING_VARIANTS
forcing = xr.Dataset(
    {
        f"{band}_forcing": fluxes[f"{band}_flux_net"].isel(variant=_perturbed)
        - fluxes[f"{band}_flux_net"].isel(variant=_control)
        for band in ("lw", "sw")
    }
)

# --- statistics ---


def error_table(dataset: xr.Dataset, rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    """Bias +- sd against the reference, one row per quantity and level."""
    table = {}
    for var, level, title in rows:
        reference = dataset[var].sel(model=REFERENCE).isel(level=level)
        table[title] = {}
        for model in MODELS:
            error = dataset[var].sel(model=model).isel(level=level) - reference
            table[title][
                LABELS[model]
            ] = f"{float(error.mean()):+.3f} +- {float(error.std()):.3f}"
    return pd.DataFrame(table).T


flux_rows = [
    ("lw_flux_up", TOA, "LW up, TOA"),
    ("sw_flux_up", TOA, "SW up, TOA"),
    ("lw_flux_dn", SFC, "LW down, surface"),
    ("sw_flux_dn", SFC, "SW down, surface"),
    ("lw_flux_net", TOA, "LW net, TOA"),
    ("sw_flux_net", TOA, "SW net, TOA"),
    ("lw_flux_net", SFC, "LW net, surface"),
    ("sw_flux_net", SFC, "SW net, surface"),
]
forcing_rows = [
    ("lw_forcing", TOA, "LW forcing, TOA"),
    ("sw_forcing", TOA, "SW forcing, TOA"),
    ("lw_forcing", SFC, "LW forcing, surface"),
    ("sw_forcing", SFC, "SW forcing, surface"),
]

print(f"\nRFMIP {VARIANT_LABEL}, error against {LABELS[REFERENCE]}")
print("bias +- sd over the 100 profiles, in W m-2\n")
print(error_table(fluxes.isel(variant=VARIANT), flux_rows).to_string())

print(f"\nRFMIP {FORCING_LABEL}, error against {LABELS[REFERENCE]}")
print("bias +- sd over the 100 profiles, in W m-2\n")
print(error_table(forcing, forcing_rows).to_string())

# --- vertical error density ---

# One panel per model. Each pressure layer is normalised to 100 % of its own
# columns, so tight and broad spreads are equally visible; bias and RMSD
# profiles are drawn on top.


def finite_together(*arrays: xr.DataArray) -> tuple[np.ndarray, ...]:
    """Broadcast, flatten, and keep points that are finite in all of them."""
    flat = [a.values.ravel() for a in xr.broadcast(*arrays)]
    keep = np.logical_and.reduce([np.isfinite(a) for a in flat])
    return tuple(a[keep] for a in flat)


def log_midpoints(profile: np.ndarray) -> np.ndarray:
    """Bin edges halfway between layers, in log pressure."""
    p = np.sort(profile[np.isfinite(profile)])
    return np.concatenate(
        [
            [p[0] / np.sqrt(p[1] / p[0])],
            np.sqrt(p[:-1] * p[1:]),
            [p[-1] * np.sqrt(p[-1] / p[-2])],
        ]
    )


def draw_density(ax, values, p_hpa, p_edges, cmap):
    """2-D histogram of values against pressure, normalised per layer."""
    x, y = finite_together(values, p_hpa)
    lo, hi = np.percentile(x, [1, 99])
    counts, x_edges, y_edges = np.histogram2d(
        x, y, bins=[np.linspace(lo, hi, 60), p_edges]
    )
    per_layer = counts.sum(axis=0, keepdims=True)
    counts = np.where(per_layer > 0, counts / np.where(per_layer > 0, per_layer, 1), 0)
    counts *= 100
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_where(counts <= 0, counts).T,
        cmap=cmap,
        shading="auto",
        rasterized=True,
    )
    positive = counts[counts > 0]
    return mesh, (positive.min(), positive.max()), (lo, hi)


def density_row(axs, var: str, show_reference: bool) -> None:
    """One row of a profile figure: the error density for each model."""
    on_layers = "layer" in fluxes[var].dims
    pressure = atm["pres_layer" if on_layers else "pres_level"]
    p_min, p_max = 10.0, float(pressure.max())
    in_range = (pressure > p_min) & (pressure < p_max)
    p_hpa = pressure.where(in_range) / 100.0
    p_profile = p_hpa.mean(dim="col").values
    p_edges = log_midpoints(p_profile)

    values = fluxes[var].isel(variant=VARIANT).where(in_range)
    reference = values.sel(model=REFERENCE)
    errors = [values.sel(model=m) - reference for m in MODELS]

    for ax in axs:
        ax.set_yscale("log")
        ax.set_ylim(p_max / 100, p_min / 100)
    axs[0].set_ylabel("Pressure / hPa")

    fig = axs[0].figure
    quantity = ("Longwave" if var.startswith("lw") else "Shortwave") + (
        " Heating Rate" if on_layers else " Net Flux"
    )
    unit = HEATING_UNIT if on_layers else FLUX_UNIT

    # The leftmost panel carries the reference's own values, for scale.
    first = 0
    if show_reference:
        mesh, (low, high), _ = draw_density(
            axs[0], reference, p_hpa, p_edges, "Oranges"
        )
        mesh.set_norm(LogNorm(vmin=low, vmax=high))
        fig.colorbar(mesh, ax=axs[0], label="% per layer", pad=0.02, shrink=0.6)
        axs[0].set_xlabel(f"{quantity} / {unit}")
        axs[0].set_title(LABELS[REFERENCE])
        first = 1

    meshes, spans, ranges = [], [], []
    for ax, error in zip(axs[first:], errors):
        mesh, span, x_range = draw_density(ax, error, p_hpa, p_edges, "Blues")
        meshes.append(mesh)
        spans.append(span)
        ranges.append(x_range)

    # One colour scale across the model panels, so they can be read against
    # each other.
    shared = LogNorm(vmin=min(s[0] for s in spans), vmax=max(s[1] for s in spans))
    for mesh in meshes:
        mesh.set_norm(shared)
    fig.colorbar(
        meshes[-1], ax=list(axs[first:]), label="% per layer", pad=0.02, shrink=0.6
    )

    profiles = [(e.mean(dim="col"), np.sqrt((e**2).mean(dim="col"))) for e in errors]

    # RMSD is a magnitude and can sit outside the density's 1-99 % range, so
    # the shared x axis is widened until both lines fit.
    edges = [x for pair in ranges for x in pair]
    edges += [float(np.nanmin(p)) for pair in profiles for p in pair]
    edges += [float(np.nanmax(p)) for pair in profiles for p in pair]
    pad = 0.03 * (max(edges) - min(edges))

    for model, ax, (bias, rmsd) in zip(MODELS, axs[first:], profiles):
        ax.set_xlim(min(edges) - pad, max(edges) + pad)
        ax.set_title(LABELS[model])
        ax.set_xlabel(rf"$\Delta$ {quantity} / {unit}")
        ax.axvline(0, color="gray", linewidth=1, zorder=1)
        ax.plot(
            bias.values, p_profile, color=BIAS_COLOR, lw=1.6, label="Bias", zorder=4
        )
        ax.plot(
            rmsd.values, p_profile, color=RMSD_COLOR, lw=1.6, label="RMSD", zorder=4
        )
    axs[first].legend(loc="lower left")


def profile_figure(variables: list[str], show_reference: bool, title: str, stem: str):
    ncols = len(MODELS) + int(show_reference)
    fig, axs = plt.subplots(
        len(variables),
        ncols,
        figsize=(4.6 * ncols + 1.8, 4.6 * len(variables)),  # + room for the colourbar
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    for row, var in zip(axs, variables):
        density_row(row, var, show_reference)
    fig.suptitle(title)
    save(fig, stem)
    return fig


# --- split violins ---

# Distribution over the 100 columns, one violin per model, split between its
# longwave and shortwave halves.


def at_level(da: xr.DataArray, level: int) -> np.ndarray:
    return da.isel(level=level).values.ravel()


def violin_frame(dataset, models, reference, var_names, level) -> pd.DataFrame:
    """Long-form values at one level; with a reference, errors against it."""
    rows = []
    for var in var_names:
        band = "Longwave" if var.startswith("lw") else "Shortwave"
        baseline = 0.0
        if reference is not None:
            baseline = at_level(dataset[var].sel(model=reference), level)
        for model in models:
            values = at_level(dataset[var].sel(model=model), level) - baseline
            rows.append(
                pd.DataFrame({"model": LABELS[model], "band": band, "value": values})
            )
    return pd.concat(rows, ignore_index=True)


def draw_violins(ax, frame, order, ylabel, title, legend) -> None:
    sns.violinplot(
        data=frame,
        x="model",
        y="value",
        hue="band",
        hue_order=["Longwave", "Shortwave"],
        order=order,
        split=True,
        palette=BAND_PALETTE,
        inner="quart",
        ax=ax,
    )
    ax.axhline(0, color="gray", linewidth=0.9, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(title="", loc="best")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()


def violin_figure(dataset, models, reference, var_names, ylabel, title, stem):
    fig, axs = plt.subplots(
        1, 2, figsize=(3.0 * len(models) + 4.0, 5), layout="constrained"
    )
    for ax, (level, panel) in zip(axs, [(TOA, "Top of atmosphere"), (SFC, "Surface")]):
        frame = violin_frame(dataset, models, reference, var_names, level)
        draw_violins(
            ax,
            frame,
            order=[LABELS[m] for m in models],
            ylabel=ylabel,
            title=panel,
            legend=ax is axs[0],
        )
    axs[1].set_ylabel("")
    fig.suptitle(title)
    save(fig, stem)
    return fig


# --- figures ---

print()
TITLE = f"RFMIP {VARIANT_LABEL} | reference {LABELS[REFERENCE]}"

# Where in the column the net-flux error sits.
profile_figure(
    ["lw_flux_net", "sw_flux_net"],
    show_reference=False,
    title=TITLE,
    stem="flux_profiles",
)

# Heating-rate error with height, beside the reference's own profile.
profile_figure(
    ["lw_heating_rate", "sw_heating_rate"],
    show_reference=True,
    title=TITLE,
    stem="heating_profiles",
)

# Net-flux error over columns, at both ends of the column.
violin_figure(
    fluxes.isel(variant=VARIANT),
    MODELS,
    REFERENCE,
    ("lw_flux_net", "sw_flux_net"),
    ylabel=rf"$\Delta$ Net Flux / {FLUX_UNIT}",
    title=TITLE,
    stem="flux_violin",
)

# The forcing itself, reference included, so its size is visible.
violin_figure(
    forcing,
    ALL_MODELS,
    None,
    ("lw_forcing", "sw_forcing"),
    ylabel=rf"Forcing / {FLUX_UNIT}",
    title=f"RFMIP {FORCING_LABEL}",
    stem="forcing",
)

# Forcing error: the part of the bias that does not cancel between experiments.
violin_figure(
    forcing,
    MODELS,
    REFERENCE,
    ("lw_forcing", "sw_forcing"),
    ylabel=rf"$\Delta$ Forcing / {FLUX_UNIT}",
    title=f"RFMIP {FORCING_LABEL} | reference {LABELS[REFERENCE]}",
    stem="forcing_error",
)

plt.close("all")

# %%
