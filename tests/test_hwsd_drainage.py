"""Tests for the HWSD2 SMU-ID -> drainage-class attribute join (_load_hwsd_drainage).

Schema confirmed against the real HWSD2.sqlite (ISRIC): the HWSD2_SMU table has
HWSD2_SMU_ID, DRAINAGE, SHARE columns -- no mock schema guessing, this matches the actual
database structure reported from a real download.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from nbs_ruralscan.data_loaders import _load_hwsd_drainage

HAITI_BBOX = (-74.5, 18.0, -71.6, 20.1)


@pytest.fixture
def hwsd_files(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    dataset_dir = tmp_path / "data" / "raw" / "hwsd_v2"
    dataset_dir.mkdir(parents=True)

    transform = from_bounds(*HAITI_BBOX, 50, 50)
    rng = np.random.default_rng(0)
    smu_ids = rng.choice([1001, 1002, 1003, 9999], size=(50, 50)).astype("float32")
    with rasterio.open(
        dataset_dir / "HWSD2.bil",
        "w",
        driver="EHdr",
        height=50,
        width=50,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(smu_ids, 1)

    con = sqlite3.connect(dataset_dir / "HWSD2.sqlite")
    con.execute(
        "CREATE TABLE HWSD2_SMU (ID INTEGER, HWSD2_SMU_ID INTEGER, DRAINAGE REAL, SHARE REAL)"
    )
    con.executemany(
        "INSERT INTO HWSD2_SMU (ID, HWSD2_SMU_ID, DRAINAGE, SHARE) VALUES (?,?,?,?)",
        [
            (1, 1001, 3.0, 70.0),  # dominant component for SMU 1001
            (2, 1001, 5.0, 30.0),  # minority component -- must NOT win
            (3, 1002, 1.0, 100.0),
            (4, 1003, 6.0, 100.0),
            # 9999 deliberately has no row -- tests the "no match" -> NaN path
        ],
    )
    con.commit()
    con.close()
    return tmp_path


def test_dominant_component_wins_by_share(hwsd_files, monkeypatch):
    monkeypatch.chdir(hwsd_files)
    row = {
        "dataset_id": "hwsd_v2",
        "access_params": '{"local_filename": "HWSD2.bil", "sqlite_filename": "HWSD2.sqlite"}',
        "download_url": "x",
    }
    da = _load_hwsd_drainage(row, HAITI_BBOX, 5000)
    present = set(da.values[~np.isnan(da.values)].tolist())
    assert present == {1.0, 3.0, 6.0}  # 3.0, not 5.0, for SMU 1001


def test_unmatched_smu_id_becomes_nan(hwsd_files, monkeypatch):
    monkeypatch.chdir(hwsd_files)
    row = {
        "dataset_id": "hwsd_v2",
        "access_params": '{"local_filename": "HWSD2.bil", "sqlite_filename": "HWSD2.sqlite"}',
        "download_url": "x",
    }
    da = _load_hwsd_drainage(row, HAITI_BBOX, 5000)
    assert np.isnan(da.values).any()


def test_missing_sqlite_filename_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = {
        "dataset_id": "hwsd_v2",
        "access_params": '{"local_filename": "HWSD2.bil"}',
        "download_url": "x",
    }
    with pytest.raises(FileNotFoundError, match="sqlite_filename"):
        _load_hwsd_drainage(row, HAITI_BBOX, 5000)
