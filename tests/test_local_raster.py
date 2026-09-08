"""Tests for the direct_download / local-raster loading pattern (Group B: WorldClim, HWSD,
CGIAR-CSI aridity)."""

from __future__ import annotations

import numpy as np
import pytest

from nbs_ruralscan.data_loaders import _load_local_raster


@pytest.fixture
def local_raster_file(tmp_path):
    """A real single-band GeoTIFF, written to disk, matching the data/raw/<dataset_id>/
    convention -- so this test exercises the real rasterio/rioxarray read path, not a mock."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    dataset_dir = tmp_path / "data" / "raw" / "test_dataset"
    dataset_dir.mkdir(parents=True)
    path = dataset_dir / "test.tif"

    data = np.linspace(0, 100, 50 * 50, dtype="float32").reshape(50, 50)
    transform = from_bounds(-74.5, 18.0, -71.6, 20.1, 50, 50)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=50,
        width=50,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    return tmp_path


def test_load_local_raster_reads_real_file(local_raster_file, monkeypatch):
    monkeypatch.chdir(local_raster_file)
    dataset_row = {
        "dataset_id": "test_dataset",
        "access_params": '{"local_filename": "test.tif"}',
        "download_url": "http://example.com",
    }
    da = _load_local_raster(dataset_row, (-74.5, 18.0, -71.6, 20.1), 5000)
    assert da.attrs["is_synthetic"] is False
    assert str(da.rio.crs) == "EPSG:4326"
    assert da.values.min() >= 0 and da.values.max() <= 100


def test_load_local_raster_missing_file_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset_row = {
        "dataset_id": "nonexistent",
        "access_params": '{"local_filename": "missing.tif"}',
        "download_url": "http://example.com",
    }
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_local_raster(dataset_row, (-74.5, 18.0, -71.6, 20.1), 5000)


def test_load_local_raster_no_access_params_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset_row = {
        "dataset_id": "no_params",
        "access_params": None,
        "download_url": "x",
    }
    with pytest.raises(FileNotFoundError, match="access_params"):
        _load_local_raster(dataset_row, (-74.5, 18.0, -71.6, 20.1), 5000)
