import pytest
from resqpy.model import new_model
from resqpy.crs import Crs
from resqpy.grid import RegularGrid
from resqpy.surface import Surface
import numpy as np
from resqpy.olio.triangulation import dt
from pathlib import Path
from typing import Tuple


@pytest.fixture
def small_grid_and_surface(tmp_path: Path) -> Tuple[RegularGrid, Surface, str]:
    """Creates a small RegularGrid and a random triangular surface saved in a temporary epc file."""
    epc_file = f"{tmp_path}/test.epc"
    model = new_model(epc_file)

    crs = Crs(model)
    crs.create_xml()

    extent = 10
    extent_kji = (extent, extent, extent)
    dxyz = (1.0, 1.0, 1.0)
    crs_uuid = crs.uuid
    grid = RegularGrid(model, extent_kji = extent_kji, dxyz = dxyz, crs_uuid = crs_uuid, title = "small_grid")
    grid.create_xml()

    n_points = 100
    points = np.random.rand(n_points, 3) * extent
    triangles = dt(points)
    surface = Surface(model, crs_uuid = crs_uuid, title = "small_surface")
    surface.set_from_triangles_and_points(triangles, points)
    surface.triangles_and_points()
    surface.write_hdf5()
    surface.create_xml()

    model.store_epc()
    return grid, surface, epc_file
