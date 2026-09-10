"""Tests for the OSM road-network -> distance-to-nearest-road loader (_load_road_distance).

Confirmed against T4's own relationship_params for distance_to_road (abs_min=0, opt_high=3,
abs_max=20) that the expected unit is KILOMETRES, not metres -- a 20-metre cutoff would be
absurd for agroforestry market access.
"""

from __future__ import annotations

import pytest

from nbs_ruralscan.data_loaders import _load_road_distance

geopandas = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")


@pytest.fixture
def road_shapefile(tmp_path):
    from shapely.geometry import LineString

    dataset_dir = tmp_path / "data" / "raw" / "osm_road_network"
    dataset_dir.mkdir(parents=True)
    # A long straight road running north-south exactly along the western edge of the test bbox.
    road = LineString([(-72.50, 18.5), (-72.50, 20.0)])
    gdf = geopandas.GeoDataFrame({"geometry": [road]}, crs="EPSG:4326")
    gdf.to_file(dataset_dir / "gis_osm_roads_free_1.shp")
    return tmp_path


def test_distance_increases_away_from_road(road_shapefile, monkeypatch):
    monkeypatch.chdir(road_shapefile)
    bbox = (
        -72.50,
        19.00,
        -72.00,
        19.50,
    )  # road sits exactly on the western edge (x=-72.50)
    row = {
        "dataset_id": "osm_road_network",
        "access_params": '{"local_filename": "gis_osm_roads_free_1.shp"}',
        "download_url": "x",
    }
    da = _load_road_distance(row, bbox, 5000)
    west_edge = float(da.values[:, 0].mean())
    east_edge = float(da.values[:, -1].mean())
    assert da.attrs["is_synthetic"] is False
    assert west_edge < east_edge  # closer to the road on the western side, as built
    # West edge pixel centres sit half a pixel (2.5km) from the road -- should be small, not 0
    # (pixel centres, not edges) and not large either.
    assert 0 < west_edge < 5
    # East edge is ~0.5 degrees (~50-55km at this latitude, allowing for UTM projection
    # distortion away from the zone's central meridian) from the road.
    assert 45 < east_edge < 60


def test_missing_local_filename_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    row = {"dataset_id": "osm_road_network", "access_params": None, "download_url": "x"}
    with pytest.raises(FileNotFoundError, match="local_filename"):
        _load_road_distance(row, (-72.5, 19.0, -72.0, 19.5), 5000)
