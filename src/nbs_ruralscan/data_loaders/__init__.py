"""Dataset loaders for Earth Engine and other geospatial inputs.

Pulls the pixels a T1 row points to into an in-memory `xarray.DataArray`, clipped to an AOI, at
the pipeline's analysis resolution. This is the *data-plane* half of the schema/data split: T1
never stores a pixel (see `runtime.schema_loader`), this module is where a `dataset_id` actually
becomes numbers.

Real path implemented: `access_type = gee_asset` via `xee` (Earth Engine <-> xarray). Requires,
on the machine actually running this (NOT this sandbox, which has no network path to Google):

    pip install earthengine-api xee
    earthengine authenticate      # once, opens a browser OAuth flow
    # then in Python, before calling load_variable:
    import ee; ee.Initialize(project="your-gcp-project")

Everything else (`direct_download`, `api`, `proprietary_licensed`) has no loader yet and falls
back to a synthetic placeholder raster, loudly flagged (`is_synthetic=True` in the returned
DataArray's attrs) with the dataset's own T1 metadata (`download_url`/`citation`) surfaced as
the recommended source to wire up next -- never silently treated as real data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rioxarray  # noqa: F401 -- registers the .rio accessor on xarray.DataArray
import xarray as xr

logger = logging.getLogger(__name__)


def _synthetic_raster(
    variable: str,
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
    value_range: tuple[float, float] = (0.0, 1.0),
    seed: int | None = None,
) -> xr.DataArray:
    """Deterministic placeholder raster over `bbox` (minx, miny, maxx, maxy) in degrees. Not
    real data -- only so the rest of the pipeline can be built/tested before every dataset has
    a real loader wired up (or before this environment has network access to the real source).

    `value_range` matters: a variable like annual_precipitation lives in mm (roughly 0-2000),
    not [0,1] -- uninformed [0,1] noise would fuzzy-standardise to a constant (always below
    every real threshold), which then breaks CRITIC weighting downstream (zero variance -> NaN
    correlation). Callers should pass the variable's own plausible range (e.g. derived from its
    T4 relationship_params) rather than accepting the [0,1] default for anything but an
    already-normalised variable.
    """
    minx, miny, maxx, maxy = bbox
    lons = np.arange(minx, maxx, resolution_deg)
    lats = np.arange(miny, maxy, resolution_deg)
    rng = np.random.default_rng(seed)
    low, high = value_range
    data = rng.uniform(low, high, size=(len(lats), len(lons)))
    da = xr.DataArray(
        data, coords={"y": lats, "x": lons}, dims=("y", "x"), name=variable
    )
    da.attrs["is_synthetic"] = True
    da.attrs["variable"] = variable
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da


def _recommended_source_note(dataset_row: dict[str, Any]) -> str:
    """One-line pointer to where a human should go look, built from T1's own metadata fields --
    never invented. Surfaced in the resolution-audit table so a synthetic row still tells you
    exactly what to wire up next."""
    parts = [f"dataset_id={dataset_row.get('dataset_id')!r}"]
    if dataset_row.get("dataset_name"):
        parts.append(str(dataset_row["dataset_name"]))
    if dataset_row.get("gee_asset_id"):
        parts.append(f"GEE asset: {dataset_row['gee_asset_id']}")
    if dataset_row.get("download_url"):
        parts.append(f"source: {dataset_row['download_url']}")
    if dataset_row.get("citation"):
        parts.append(f"cite: {dataset_row['citation']}")
    return " | ".join(parts)


def _load_local_raster(
    dataset_row: dict[str, Any],
    bbox: tuple[float, float, float, float],
    resolution_m: int,
) -> xr.DataArray:
    """Open a manually-downloaded GeoTIFF from `data/raw/<dataset_id>/` and clip+reproject it
    onto the requested bbox/resolution. The generic "Group B" pattern: one function, reused by
    every raster the team downloads by hand (WorldClim v2, CGIAR-CSI aridity, HWSD v2) instead
    of writing a bespoke loader per provider.

    Reads `dataset_row["access_params"]` (a JSON string, T1's existing extensibility column --
    reused rather than adding new ad-hoc T1 columns, which would need a matching update to the
    frozen schema/structure/columns.json manifest). Expected keys:
        local_filename (required) -- exact file to open inside `data/raw/<dataset_id>/`.
        band (optional, 1-indexed) -- for a provider that ships one multi-band file.
    """
    import json as _json

    import rioxarray as rxr

    dataset_id = dataset_row["dataset_id"]
    raw_params = dataset_row.get("access_params")
    params = (
        _json.loads(raw_params) if raw_params and isinstance(raw_params, str) else {}
    )
    filename = params.get("local_filename")
    band = params.get("band")
    if not filename:
        raise FileNotFoundError(
            f"T1 row for {dataset_id!r} has no access_params.local_filename set -- add "
            '{"local_filename": "<exact filename>"} before this loader can find it.'
        )
    path = Path("data") / "raw" / dataset_id / filename
    if not path.exists():
        raise FileNotFoundError(
            f"expected downloaded file at {path} for dataset_id={dataset_id!r} -- not found. "
            f"Download it from {dataset_row.get('download_url')} and place it there first."
        )

    da = rxr.open_rasterio(path, masked=True)
    if band is not None:
        da = da.isel(band=int(band) - 1)
    elif "band" in da.dims:
        da = da.isel(band=0)

    minx, miny, maxx, maxy = bbox
    da = da.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    if da.rio.crs is None:
        da = da.rio.write_crs("EPSG:4326")
    if da.rio.crs.to_epsg() != 4326:
        da = da.rio.reproject("EPSG:4326")

    da.attrs["is_synthetic"] = False
    return da


def _load_gee_asset(
    gee_asset_id: str,
    bbox: tuple[float, float, float, float],
    resolution_m: int,
    band: str | None = None,
) -> xr.DataArray:
    """Real pull via xee. Handles both `ee.Image` assets (e.g. `USGS/SRTMGL1_003`) and
    `ee.ImageCollection` assets (e.g. climate time series like ERA5) by reducing a collection
    to its temporal mean before opening -- adequate for a baseline scoping pull; a real
    scenario-aware pull (per-scenario compositing, seasonal aggregation) is recipe-specific
    logic that belongs in a later iteration, not this generic loader.

    xee 0.1.2's `open_dataset` does NOT take `scale`/`geometry` (confirmed against the
    installed version's real signature -- an earlier draft of this function assumed a newer
    xee API and broke with `TypeError: unexpected keyword argument 'geometry'`). This version
    instead builds the request the way 0.1.2 actually wants it: an explicit affine transform
    (`crs_transform`) plus an explicit output raster shape (`shape_2d`), computed here from
    `bbox` + `resolution_m` -- the same information `scale`/`geometry` used to carry, just
    spelled out by hand instead of inferred by xee.
    """
    import math

    import ee
    import xee  # noqa: F401 -- registers the "ee" xarray backend

    minx, miny, maxx, maxy = bbox
    resolution_deg = resolution_m / 111_320  # rough metres-to-degrees at the equator

    width = max(1, math.ceil((maxx - minx) / resolution_deg))
    height = max(1, math.ceil((maxy - miny) / resolution_deg))
    # GDAL-style affine: (pixel_width, row_rotation, x_origin(upper-left),
    #                      col_rotation, pixel_height(negative), y_origin(upper-left))
    # Must be a real tuple -- xee's ext.py does `isinstance(crs_transform, tuple)` and
    # rejects a list with the same six values (confirmed against the installed xee source).
    crs_transform = (resolution_deg, 0, minx, 0, -resolution_deg, maxy)

    try:
        image = ee.Image(gee_asset_id)
        image.getInfo()  # forces evaluation; raises if this asset_id is actually a collection
    except Exception:  # noqa: BLE001 -- intentional type probe (Image vs Collection vs Vector), any failure means "try the next type"
        try:
            image = ee.ImageCollection(gee_asset_id).mean()
            image.getInfo()  # forces evaluation; raises if this is actually a FeatureCollection
        except Exception:  # noqa: BLE001 -- last type probe in the chain, falls through to FeatureCollection
            # Vector asset (e.g. WDPA polygons: WCMC/WDPA/current/polygons) -- confirmed
            # against a real run: neither ee.Image nor ee.ImageCollection accepts it, GEE
            # calls it an "EECollection" (its generic term for FeatureCollection). Rasterise
            # to a binary indicator -- 1 inside a feature (e.g. inside a protected area),
            # 0 outside -- via .paint(), which is what a threshold-type T4 row (like
            # protected_area_status) actually needs: a 0/1 mask, not a continuous field.
            fc = ee.FeatureCollection(gee_asset_id)
            image = ee.Image(0).paint(fc, 1).rename("value")

    if band:
        image = image.select(band)

    collection = ee.ImageCollection([image])
    ds = xr.open_dataset(
        collection,
        engine="ee",
        crs="EPSG:4326",
        crs_transform=crs_transform,
        # xee's own docs (ext.py line ~181/964) say shape_2d is "(width, height)" -- the
        # OPPOSITE order from rioxarray/rasterio's usual (height, width)/(rows, cols)
        # convention used a few steps later in _align_to_grid. Confirmed as a real bug against
        # a live run: with (height, width) here, xee returned x (longitude, should be width=65
        # points) truncated to 47 points, and y (latitude, should be height=47) inflated to 65
        # -- silently swapping which axis got which point count, which is exactly why the
        # eastern ~28% of every GEE-pulled variable came back as NaN (the requested longitude
        # range was cut short without erroring).
        shape_2d=(width, height),
    )
    var_name = next(iter(ds.data_vars))
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    da.attrs["is_synthetic"] = False
    return da


def _load_reference_table(
    variable: str,
    country_iso3: str | None,
    bbox: tuple[float, float, float, float],
    resolution_deg: float,
    table_path: str = "schema/reference/country_statistics.csv",
) -> tuple[xr.DataArray | None, float | None]:
    """Look up a single country-level statistic (WGI, Findex, FAOSTAT, ILOSTAT-style data) and
    broadcast it uniformly across the AOI grid. No reprojection needed -- there is no spatial
    variation to resample, the whole point is that this variable is the SAME value everywhere
    within one country.

    Returns (DataArray, value) on a real hit -- both, since the notebook may want to print the
    scalar directly. Returns (None, None) if the table has no row, or the row's value is empty
    (still a TODO -- see schema/reference/country_statistics.csv), so the caller can fall back
    to synthetic without silently treating a placeholder as real data.
    """
    import csv as _csv
    from pathlib import Path

    path = Path(table_path)
    if not path.exists() or country_iso3 is None:
        return None, None
    with open(path, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if row["variable"] == variable and row["country_iso3"] == country_iso3:
                if not row["value"]:
                    return None, None  # TODO row -- not filled in yet, not a real value
                value = float(row["value"])
                minx, miny, maxx, maxy = bbox
                lons = np.arange(minx, maxx, resolution_deg)
                lats = np.arange(miny, maxy, resolution_deg)
                data = np.full((len(lats), len(lons)), value)
                da = xr.DataArray(
                    data, coords={"y": lats, "x": lons}, dims=("y", "x"), name=variable
                )
                da.attrs["is_synthetic"] = False
                da.attrs["variable"] = variable
                da.rio.write_crs("EPSG:4326", inplace=True)
                return da, value
    return None, None


def load_variable(
    variable: str,
    dataset_row: dict[str, Any],
    bbox: tuple[float, float, float, float],
    resolution_m: int = 1000,
    band: str | None = None,
    synthetic_value_range: tuple[float, float] = (0.0, 1.0),
    country_iso3: str | None = None,
) -> xr.DataArray:
    """Load one variable's raster for an AOI bbox, per its T1 row.

    `dataset_row` is one row of T1_data_registry (as a dict -- e.g. `df.loc[i].to_dict()`).
    `bbox` is (minx, miny, maxx, maxy) in EPSG:4326 degrees. Falls back to a synthetic raster
    (attrs["is_synthetic"] = True, attrs["recommended_source"] = <T1 pointer>) whenever the
    real pull fails or isn't implemented for this dataset's access_type -- callers (the
    resolution-audit step) should check `is_synthetic` and surface `recommended_source`.
    `synthetic_value_range` should be the variable's own plausible raw-value range (e.g. from
    its T4 relationship_params) so the placeholder actually exercises fuzzy standardisation
    meaningfully instead of collapsing to a constant. `country_iso3` is only used by the
    `reference_table` access_type branch (country-level statistics with no spatial variation).
    """
    access_type = dataset_row.get("access_type")
    resolution_deg = resolution_m / 111_320  # rough metres-to-degrees at the equator
    source_note = _recommended_source_note(dataset_row)

    if access_type == "reference_table":
        da, _ = _load_reference_table(variable, country_iso3, bbox, resolution_deg)
        if da is not None:
            return da
        logger.warning(
            "%r: no real value yet in schema/reference/country_statistics.csv for "
            "country_iso3=%r -- fill in that row (or download it) rather than trusting a "
            "synthetic placeholder for this one; a made-up governance/finance score is worse "
            "than an honest gap. Recommended source: %s",
            variable,
            country_iso3,
            source_note,
        )
        da = _synthetic_raster(
            variable, bbox, resolution_deg, value_range=synthetic_value_range
        )
        da.attrs["recommended_source"] = source_note
        return da

    if access_type == "gee_asset":
        gee_asset_id = dataset_row.get("gee_asset_id")
        if not gee_asset_id:
            logger.warning(
                "%r: access_type=gee_asset but T1.gee_asset_id is empty. Recommended source "
                "to fill in: %s",
                variable,
                source_note,
            )
            da = _synthetic_raster(
                variable, bbox, resolution_deg, value_range=synthetic_value_range
            )
            da.attrs["recommended_source"] = source_note
            return da
        try:
            return _load_gee_asset(gee_asset_id, bbox, resolution_m, band=band)
        except ImportError:
            logger.warning(
                "%r: earthengine-api/xee not installed in this environment. Install with "
                "`pip install earthengine-api xee` and run `earthengine authenticate` once, "
                "then `ee.Initialize()`, to pull the real data. Recommended source: %s",
                variable,
                source_note,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any GEE/network/auth failure should fall back to synthetic, not crash the pipeline
            logger.warning(
                "%r: GEE pull failed (%s: %s) -- falling back to synthetic. Common causes: "
                "`ee.Initialize()` not called yet, no internet path to Google from this "
                "environment, or the asset id is stale. Recommended source: %s",
                variable,
                type(exc).__name__,
                exc,
                source_note,
            )
        da = _synthetic_raster(
            variable, bbox, resolution_deg, value_range=synthetic_value_range
        )
        da.attrs["recommended_source"] = source_note
        return da

    if access_type == "direct_download":
        try:
            return _load_local_raster(dataset_row, bbox, resolution_m)
        except FileNotFoundError as exc:
            logger.warning(
                "%r: %s Falling back to synthetic. Recommended source: %s",
                variable,
                exc,
                source_note,
            )
        except Exception as exc:  # noqa: BLE001 -- any local-file/raster read failure should fall back to synthetic, not crash the pipeline
            logger.warning(
                "%r: local raster load failed (%s: %s) -- falling back to synthetic. "
                "Recommended source: %s",
                variable,
                type(exc).__name__,
                exc,
                source_note,
            )
        da = _synthetic_raster(
            variable, bbox, resolution_deg, value_range=synthetic_value_range
        )
        da.attrs["recommended_source"] = source_note
        return da

    # api / proprietary_licensed -- no loader built yet for either (each needs per-provider
    # handling: an API call + point-to-grid rasterisation for something like ACLED, etc.).
    # Same synthetic-fallback contract as the other branches above.
    logger.warning(
        "%r: no loader implemented yet for access_type=%r. Recommended source to explore "
        "next: %s",
        variable,
        access_type,
        source_note,
    )
    da = _synthetic_raster(
        variable, bbox, resolution_deg, value_range=synthetic_value_range
    )
    da.attrs["recommended_source"] = source_note
    return da
