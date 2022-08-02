"""High level add single cell grid function."""

import os
import numpy as np

import resqpy.olio.grid_functions as gf
import resqpy.rq_import as rqi


def add_single_cell_grid(points,
                         new_grid_title = None,
                         new_epc_file = None,
                         xy_units = 'm',
                         z_units = 'm',
                         z_inc_down = True):
    """Creates a model with a single cell IJK Grid, with a cuboid cell aligned with x,y,z axes, enclosing points."""

    assert new_epc_file is not None

    # determine range of points
    min_xyz = np.nanmin(points.reshape((-1, 3)), axis = 0)
    max_xyz = np.nanmax(points.reshape((-1, 3)), axis = 0)
    assert not np.any(np.isnan(min_xyz)) and not np.any(np.isnan(max_xyz))

    # create corner point array in pagoda protocol
    cp = np.array([[min_xyz[0], min_xyz[1], min_xyz[2]], [max_xyz[0], min_xyz[1], min_xyz[2]],
                   [min_xyz[0], max_xyz[1], min_xyz[2]], [max_xyz[0], max_xyz[1], min_xyz[2]],
                   [min_xyz[0], min_xyz[1], max_xyz[2]], [max_xyz[0], min_xyz[1], max_xyz[2]],
                   [min_xyz[0], max_xyz[1], max_xyz[2]], [max_xyz[0], max_xyz[1], max_xyz[2]]]).reshape(
                       (1, 1, 1, 2, 2, 2, 3))

    # switch to nexus ordering
    gf.resequence_nexus_corp(cp)

    # write cp to temp pure binary file
    temp_file = new_epc_file[:-4] + '.temp.db'
    with open(temp_file, 'wb') as fp:
        fp.write(cp.data)

    # use_rq_import to create a new model
    one_cell_model = rqi.import_nexus(new_epc_file[:-4],
                                      extent_ijk = (1, 1, 1),
                                      corp_file = temp_file,
                                      corp_xy_units = xy_units,
                                      corp_z_units = z_units,
                                      corp_z_inc_down = z_inc_down,
                                      ijk_handedness = 'left',
                                      resqml_xy_units = xy_units,
                                      resqml_z_units = z_units,
                                      resqml_z_inc_down = z_inc_down,
                                      use_binary = True,
                                      split_pillars = False,
                                      grid_title = new_grid_title)
    grid = one_cell_model.grid()

    os.remove(temp_file)

    return grid
