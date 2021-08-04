import pytest
import os
import numpy as np

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.fault as rqf
import resqpy.lines as rql
import resqpy.derived_model as rqdm
import resqpy.olio.transmission as rqtr


def test_fault_connection_set(tmp_path):

   gm = os.path.join(tmp_path, 'resqpy_test_fgcs.epc')

   model = rq.Model(gm, new_epc = True, create_basics = True, create_hdf5_ext = True)

   # unsplit grid

   g0 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))
   g0.write_hdf5()
   g0.create_xml(title = 'G0 unsplit')
   g0_fcs, g0_fa = rqtr.fault_connection_set(g0)

   assert g0_fcs is None
   assert g0_fa is None

   # J face split with no juxtaposition
   throw = 2.0

   g1 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g1.grid_representation = 'IjkGrid'

   pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
   pu = g1.points_ref(masked = False).reshape(g1.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g1.nk + 1, pu_pillar_count + g1.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, g1.ni + 1:2 * (g1.ni + 1), :]
   p[:, 2 * (g1.ni + 1):, 2] += throw
   g1.points_cached = p
   g1.has_split_coordinate_lines = True
   g1.split_pillars_count = g1.ni + 1
   g1.split_pillar_indices_cached = np.array([i for i in range(g1.ni + 1, 2 * (g1.ni + 1))], dtype = int)
   g1.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g1.write_hdf5()
   g1.create_xml(title = 'G1 J no juxtaposition')

   # model.store_epc()
   # model.h5_release()

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is None
   assert g1_fa is None

   # I face split with no juxtaposition
   throw = 2.0

   g1 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g1.grid_representation = 'IjkGrid'

   pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
   pr = g1.points_ref(masked = False)
   pr[:, :, -1, 2] += throw
   pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
   p[:, pu_pillar_count:, 2] += throw
   g1.points_cached = p
   g1.has_split_coordinate_lines = True
   g1.split_pillars_count = g1.ni + 1
   g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
   g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
   g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g1.write_hdf5()
   g1.create_xml(title = 'G1 I no juxtaposition')

   # model.store_epc()

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is None
   assert g1_fa is None

   # J face split with full juxtaposition of kji0 (1, 0, *) with (0, 1, *)
   # pattern 4, 4 (or 3, 3)

   throw = 1.0

   g2 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g2.grid_representation = 'IjkGrid'

   pu_pillar_count = (g2.nj + 1) * (g2.ni + 1)
   pu = g2.points_ref(masked = False).reshape(g2.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g2.nk + 1, pu_pillar_count + g2.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, g2.ni + 1:2 * (g2.ni + 1), :]
   p[:, 2 * (g2.ni + 1):, 2] += throw
   g2.points_cached = p
   g2.has_split_coordinate_lines = True
   g2.split_pillars_count = g2.ni + 1
   g2.split_pillar_indices_cached = np.array([i for i in range(g2.ni + 1, 2 * (g2.ni + 1))], dtype = int)
   g2.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   g2.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g2.write_hdf5()
   g2.create_xml(title = 'G2 J full juxtaposition of kji0 (1, 0, *) with (0, 1, *)')

   # model.store_epc()

   g2_fcs, g2_fa = rqtr.fault_connection_set(g2)

   assert g2_fcs is not None
   assert g2_fa is not None

   # show_fa(g2, g2_fcs, g2_fa)

   assert g2_fcs.count == 2
   assert np.all(np.isclose(g2_fa, 1.0, atol = 0.01))

   # I face split with full juxtaposition of kji0 (1, *, 0) with (0, *, 1)
   # pattern 4, 4 (or 3, 3) diagram 1

   throw = 1.0

   g1 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g1.grid_representation = 'IjkGrid'

   pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
   pr = g1.points_ref(masked = False)
   pr[:, :, -1, 2] += throw
   pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
   p[:, pu_pillar_count:, 2] += throw
   g1.points_cached = p
   g1.has_split_coordinate_lines = True
   g1.split_pillars_count = g1.ni + 1
   g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
   g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
   g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g1.write_hdf5()
   g1.create_xml(title = 'G2 I full juxtaposition of kji0 (1, *, 0) with (0, *, 1)')

   # model.store_epc()

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is not None
   assert g1_fa is not None

   # show_fa(g1, g1_fcs, g1_fa)

   assert g1_fcs.count == 2
   assert np.all(np.isclose(g1_fa, 1.0, atol = 0.01))

   # J face split with half juxtaposition of kji0 (*, 0, *) with (*, 1, *); and (1, 0, *) with (0, 1, *)
   # pattern 5, 5 (or 2, 2) diagram 2

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]
   p[:, 2 * (grid.ni + 1):, 2] += throw
   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G3 J half juxtaposition of kji0 (*, 0, *) with (*, 1, *); and (1, 0, *) with (0, 1, *)')

   # model.store_epc()

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(np.isclose(grid_fa, 0.5, atol = 0.01))

   # I face split with half juxtaposition of kji0 (*, *, 0) with (*, *, 1); and (1, *, 0) with (0, *, 1)
   # pattern 5, 5 (or 2, 2) diagram 2

   throw = 0.5

   g1 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g1.grid_representation = 'IjkGrid'

   pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
   pr = g1.points_ref(masked = False)
   pr[:, :, -1, 2] += throw
   pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
   p[:, pu_pillar_count:, 2] += throw
   g1.points_cached = p
   g1.has_split_coordinate_lines = True
   g1.split_pillars_count = g1.ni + 1
   g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
   g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
   g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g1.write_hdf5()
   g1.create_xml(title = 'G3 I half juxtaposition of kji0 (*, *, 0) with (*, *, 1); and (1, *, 0) with (0, *, 1)')

   # model.store_epc()

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is not None
   assert g1_fa is not None

   # show_fa(g1, g1_fcs, g1_fa)

   assert g1_fcs.count == 6
   assert np.all(np.isclose(g1_fa, 0.5, atol = 0.01))

   # J face split with 0.25 juxtaposition of kji0 (*, 0, *) with (*, 1, *); and 0.75 of (1, 0, *) with (0, 1, *)
   # pattern 5, 5 (or 2, 2) diagram 2

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]
   p[:, 2 * (grid.ni + 1):, 2] += throw
   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(
      title = 'G4 J 0.25 juxtaposition of kji0 (*, 0, *) with (*, 1, *); and 0.75 of (1, 0, *) with (0, 1, *)')

   # model.store_epc()

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.25], [0.25, 0.25], [0.75, 0.75], [0.25, 0.25]]),
                 atol = 0.01))

   # I face split with 0.25 juxtaposition of kji0 (*, *, 0) with (*, *, 1); and 0.75 of (1, *, 0) with (0, *, 1)
   # pattern 5, 5 (or 2, 2) diagram 2

   throw = 0.75

   g1 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g1.grid_representation = 'IjkGrid'

   pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
   pr = g1.points_ref(masked = False)
   pr[:, :, -1, 2] += throw
   pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
   p[:, pu_pillar_count:, 2] += throw
   g1.points_cached = p
   g1.has_split_coordinate_lines = True
   g1.split_pillars_count = g1.ni + 1
   g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
   g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
   g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g1.write_hdf5()
   g1.create_xml(
      title = 'G4 I 0.25 juxtaposition of kji0 (*, *, 0) with (*, *, 1); and 0.75 of (1, *, 0) with (0, *, 1)')

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is not None
   assert g1_fa is not None

   # show_fa(g1, g1_fcs, g1_fa)

   assert g1_fcs.count == 6
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(
      np.isclose(g1_fa,
                 np.array([[0.25, 0.25], [0.75, 0.75], [0.25, 0.25], [0.25, 0.25], [0.75, 0.75], [0.25, 0.25]]),
                 atol = 0.01))

   # J face split with full full (1, 0, 0) with (0, 1, 0); and 0.5 of (*, 0, 1) with (*, 1, 1) and layer crossover
   # diagrams 4 & 2

   throw = 1.0

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]
   p[:, 2 * (grid.ni + 1):-1, 2] += throw
   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(
      title = 'G5 J full (1, 0, 0) with (0, 1, 0); and 0.5 of (*, 0, 1) with (*, 1, 1) and layer crossover')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(np.isclose(grid_fa, np.array([[1., 1.], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), atol = 0.01))

   # I face split with full (1, 0, 0) with (0, 0, 1); and 0.5 of (*, 1, 0) with (*, 1, 1) and layer crossover

   throw = 1.0

   g1 = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   g1.grid_representation = 'IjkGrid'

   pu_pillar_count = (g1.nj + 1) * (g1.ni + 1)
   pr = g1.points_ref(masked = False)
   pr[:, :, -1, 2] += throw
   pu = pr.reshape(g1.nk + 1, pu_pillar_count, 3)
   p = np.zeros((g1.nk + 1, pu_pillar_count + g1.nj + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pr[:, :, 1, :].reshape(g1.nk + 1, g1.nj + 1, 3)
   p[:, pu_pillar_count:-1, 2] += throw
   g1.points_cached = p
   g1.has_split_coordinate_lines = True
   g1.split_pillars_count = g1.ni + 1
   g1.split_pillar_indices_cached = np.array([i for i in range(1, (g1.nj + 1) * (g1.ni + 1), g1.nj + 1)], dtype = int)
   g1.cols_for_split_pillars = np.array((1, 1, 3, 3), dtype = int)
   g1.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   g1.write_hdf5()
   g1.create_xml(title = 'G5 I full (1, 0, 0) with (0, 0, 1); and 0.5 of (*, 1, 0) with (*, 1, 1) and layer crossover')

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is not None
   assert g1_fa is not None

   # show_fa(g1, g1_fcs, g1_fa)

   assert g1_fcs.count == 4
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(np.isclose(g1_fa, np.array([[1., 1.], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), atol = 0.01))

   grid_fa

   # J face split diagram 4

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):-1, 2] += throw

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G6 J diagram 4')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.75, 0.75], [0.25, 0.25], [0.75, 0.75]]),
                 atol = 0.01))

   # J face split
   # diagram 5

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[1:, 2 * (grid.ni + 1), 2] -= 0.9
   p[1:, 3 * (grid.ni + 1), 2] -= 1.0

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G7 diagram 5')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 7
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[0.375, 0.75], [0.125, 0.125], [0.125, 0.25], [0.75, 0.75], [0.5, 0.5], [0.5, 0.5],
                           [0.5, 0.5]]),
                 atol = 0.01))

   # bl.set_log_level('info')

   # J face split
   # diagram 5m

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[1:, 3 * (grid.ni + 1) - 1, 2] -= 0.9
   p[1:, 4 * (grid.ni + 1) - 1, 2] -= 1.0

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G7m diagram 5m')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 7
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.375, 0.75], [0.125, 0.125], [0.125, 0.25],
                           [0.75, 0.75]]),
                 atol = 0.01))

   # bl.set_log_level('info')

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G8 lower layer half thickness; throw 0.5')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   assert np.all(np.isclose(grid_fa, np.array([[0.5, 0.5], [1.0, 0.5], [0.5, 0.5], [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G9 lower layer half thickness; throw 0.75')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   assert np.all(np.isclose(grid_fa, np.array([[0.25, 0.25], [1.0, 0.5], [0.25, 0.25], [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G10 lower layer half thickness; throw 0.25')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[0.75, 0.75], [0.5, 0.25], [0.5, 0.5], [0.75, 0.75], [0.5, 0.25], [0.5, 0.5]]),
                 atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:, 2 * (grid.ni + 1), 2] -= 1.0
   p[:, 3 * (grid.ni + 1), 2] -= 1.0
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G11 lower layer half thickness; throw 0.75 except for -0.25 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(
         grid_fa,
         np.array([
            [11.0 / 16, 11.0 / 16],  # 0.6875, 0.6875
            [1.0 / 32, 1.0 / 16],  # 0.03125, 0.0625
            [0.5, 0.25],
            [0.4375, 0.4375],
            [0.25, 0.25],
            [1.0, 0.5]
         ]),
         atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:, 3 * (grid.ni + 1) - 1, 2] -= 1.0
   p[:, 4 * (grid.ni + 1) - 1, 2] -= 1.0
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G12 lower layer half thickness; throw 0.75 except for -0.25 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(
         grid_fa,
         np.array([
            [0.25, 0.25],
            [1.0, 0.5],
            [11.0 / 16, 11.0 / 16],  # 0.6875, 0.6875
            [1.0 / 32, 1.0 / 16],  # 0.03125, 0.0625
            [0.5, 0.25],
            [0.4375, 0.4375]
         ]),
         atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:, 3 * (grid.ni + 1) - 1, 2] -= 1.5
   p[:, 4 * (grid.ni + 1) - 1, 2] -= 1.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G13 lower layer half thickness; throw 0.75 except for -0.25 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(
         grid_fa,
         np.array([
            [0.25, 0.25],
            [1.0, 0.5],
            [5.0 / 8, 5.0 / 8],  # 0.625, 0.625
            [1.0 / 6, 1.0 / 3],
            [1.0 / 3, 1.0 / 6],
            [1.0 / 3, 1.0 / 3]
         ]),
         atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[0, 2 * (grid.ni + 1), 2] += 0.5
   p[0, 3 * (grid.ni + 1), 2] += 0.5
   p[1:, 2 * (grid.ni + 1), 2] -= 0.5
   p[1:, 3 * (grid.ni + 1), 2] -= 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G14 lower layer half thickness; throw 0.75 except for top layer pinching on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 5
   assert np.all(
      np.isclose(
         grid_fa,
         np.array([
            [1.0 / 16, 1.0 / 8],  # 0.0625, 0.125
            [0.75, 0.75],
            [0.125, 0.125],
            [0.25, 0.25],
            [1.0, 0.5]
         ]),
         atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = -0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[0, 2 * (grid.ni + 1), 2] += 0.5
   p[0, 3 * (grid.ni + 1), 2] += 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G15 lower layer half thickness; throw -0.25 except for top layer pinching on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[11.0 / 16, 11.0 / 12], [0.25, 0.5], [0.5, 0.5], [0.75, 0.75], [0.25, 0.5], [0.5, 0.5]]),
                 atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = -0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:, 2 * (grid.ni + 1), 2] -= 1.0
   p[:, 3 * (grid.ni + 1), 2] -= 1.0
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G16 lower layer half thickness; throw -0.75 except for -1.75 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   assert np.all(
      np.isclose(grid_fa, np.array([[1.0 / 32, 1.0 / 32], [0.25, 0.5], [0.25, 0.25], [0.5, 1.0]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:, 2 * (grid.ni + 1), 2] += 1.0
   p[:, 3 * (grid.ni + 1), 2] += 1.0
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G17 lower layer half thickness; throw +0.75 except for +1.75 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   assert np.all(
      np.isclose(grid_fa, np.array([[1.0 / 32, 1.0 / 32], [0.5, 0.25], [0.25, 0.25], [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:, 3 * (grid.ni + 1) - 1, 2] += 1.0
   p[:, 4 * (grid.ni + 1) - 1, 2] += 1.0
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G18 lower layer half thickness; throw +0.75 except for +1.75 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   assert np.all(
      np.isclose(grid_fa, np.array([[0.25, 0.25], [1.0, 0.5], [1.0 / 32, 1.0 / 32], [0.5, 0.25]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[0, 2 * (grid.ni + 1), 2] -= 0.5
   p[0, 3 * (grid.ni + 1), 2] -= 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G19 lower layer half thickness; throw +0.25 except for top edge -0.25 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[15.0 / 16, 0.75], [0.5, 0.2], [0.5, 0.5], [0.75, 0.75], [0.5, 0.25], [0.5, 0.5]]),
                 atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[0, 3 * (grid.ni + 1) - 1, 2] -= 0.5
   p[0, 4 * (grid.ni + 1) - 1, 2] -= 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G20 lower layer half thickness; throw +0.25 except for top edge -0.25 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[0.75, 0.75], [0.5, 0.25], [0.5, 0.5], [15.0 / 16, 0.75], [0.5, 0.2], [0.5, 0.5]]),
                 atol = 0.01))

   # bl.set_log_level('info')

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[1, 2 * (grid.ni + 1), 2] -= 0.5
   p[1, 3 * (grid.ni + 1), 2] -= 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G21 lower layer half thickness; throw +0.25 except for mid edge -0.25 on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 7
   assert np.all(
      np.isclose(grid_fa,
                 np.array([[11.0 / 16, 11.0 / 12], [1.0 / 16, 1.0 / 12], [1.0 / 8, 1.0 / 12], [7.0 / 8, 7.0 / 12],
                           [0.75, 0.75], [0.5, 0.25], [0.5, 0.5]]),
                 atol = 0.01))

   # bl.set_log_level('info')

   # bl.set_log_level('debug')
   # local_log_out = bl.log_fresh()
   # display(local_log_out)

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:2, 2 * (grid.ni + 1), 2] -= 0.5
   p[:2, 3 * (grid.ni + 1), 2] -= 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G22 lower layer half thickness; throw +0.5 except for top, mid edge on one wing')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 5
   assert np.all(
      np.isclose(grid_fa, np.array([[0.75, 0.75], [0.5, 0.25], [0.5, 1.0 / 3], [0.5, 0.5], [1.0, 0.5]]), atol = 0.01))

   # bl.set_log_level('debug')
   # local_log_out = bl.log_fresh()
   # display(local_log_out)

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1:2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1):, 2] += throw
   p[:2, 3 * (grid.ni + 1) - 1, 2] -= 0.5
   p[:2, 4 * (grid.ni + 1) - 1, 2] -= 0.5
   p[-1, :, 2] -= 0.5

   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G23 lower layer half thickness; throw +0.5 except for top, mid edge on one wing')

   model.store_epc()
   model.h5_release()

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 5
   assert np.all(
      np.isclose(grid_fa, np.array([[0.5, 0.5], [1.0, 0.5], [0.75, 0.75], [0.5, 0.25], [0.5, 1.0 / 3]]), atol = 0.01))


def test_add_faults(tmp_path):

   def write_poly(filename, a, mode = 'w'):
      nines = 999.0
      with open(filename, mode = mode) as fp:
         for row in range(len(a)):
            fp.write(f'{a[row, 0]:8.3f} {a[row, 1]:8.3f} {a[row, 2]:8.3f}\n')
         fp.write(f'{nines:8.3f} {nines:8.3f} {nines:8.3f}\n')

   def make_poly(model, a, title, crs):
      return [
         rql.Polyline(model, set_bool = False, set_coord = a, set_crs = crs.uuid, set_crsroot = crs.root, title = title)
      ]

   epc = os.path.join(tmp_path, 'tic_tac_toe.epc')

   for test_mode in ['file', 'polyline']:

      model = rq.new_model(epc)
      grid = grr.RegularGrid(model, extent_kji = (1, 3, 3), set_points_cached = True)
      grid.write_hdf5()
      grid.create_xml(write_geometry = True)
      crs = rqc.Crs(model, uuid = grid.crs_uuid)
      model.store_epc()

      # single straight fault
      a = np.array([[-0.2, 2.0, -0.1], [3.2, 2.0, -0.1]])
      f = os.path.join(tmp_path, 'ttt_f1.dat')
      if test_mode == 'file':
         write_poly(f, a)
         lines_file_list = [f]
         polylines = None
      else:
         lines_file_list = None
         polylines = make_poly(model, a, 'ttt_f1', crs)
      g = rqdm.add_faults(epc,
                          source_grid = None,
                          polylines = polylines,
                          lines_file_list = lines_file_list,
                          inherit_properties = False,
                          new_grid_title = 'ttt_f1 straight')

      # single zig-zag fault
      a = np.array([[-0.2, 1.0, -0.1], [1.0, 1.0, -0.1], [1.0, 2.0, -0.1], [3.2, 2.0, -0.1]])
      f = os.path.join(tmp_path, 'ttt_f2.dat')
      if test_mode == 'file':
         write_poly(f, a)
         lines_file_list = [f]
         polylines = None
      else:
         lines_file_list = None
         polylines = make_poly(model, a, 'ttt_f2', crs)
      g = rqdm.add_faults(epc,
                          source_grid = None,
                          polylines = polylines,
                          lines_file_list = lines_file_list,
                          inherit_properties = True,
                          new_grid_title = 'ttt_f2 zig_zag')

      # single zig-zag-zig fault
      a = np.array([[-0.2, 1.0, -0.1], [1.0, 1.0, -0.1], [1.0, 2.0, -0.1], [2.0, 2.0, -0.1], [2.0, 1.0, -0.1],
                    [3.2, 1.0, -0.1]])
      f = os.path.join(tmp_path, 'ttt_f3.dat')
      if test_mode == 'file':
         write_poly(f, a)
         lines_file_list = [f]
         polylines = None
      else:
         lines_file_list = None
         polylines = make_poly(model, a, 'ttt_f3', crs)
      g = rqdm.add_faults(epc,
                          source_grid = None,
                          polylines = polylines,
                          lines_file_list = lines_file_list,
                          inherit_properties = True,
                          new_grid_title = 'ttt_f3 zig_zag_zig')

      # horst block
      a = np.array([[-0.2, 1.0, -0.1], [3.2, 1.0, -0.1]])
      b = np.array([[3.2, 2.0, -0.1], [-0.2, 2.0, -0.1]])
      fa = os.path.join(tmp_path, 'ttt_f4a.dat')
      fb = os.path.join(tmp_path, 'ttt_f4b.dat')
      if test_mode == 'file':
         write_poly(fa, a)
         write_poly(fb, b)
         lines_file_list = [fa, fb]
         polylines = None
      else:
         lines_file_list = None
         polylines = make_poly(model, a, 'ttt_f4a', crs) + make_poly(model, b, 'ttt_f4b', crs)
      g = rqdm.add_faults(epc,
                          source_grid = None,
                          polylines = polylines,
                          lines_file_list = lines_file_list,
                          inherit_properties = True,
                          new_grid_title = 'ttt_f4 horst')

      # asymmetrical horst block
      lr_throw_dict = {'ttt_f4a': (0.0, -0.3), 'ttt_f4b': (0.0, -0.6)}
      g = rqdm.add_faults(epc,
                          source_grid = None,
                          polylines = polylines,
                          lines_file_list = lines_file_list,
                          left_right_throw_dict = lr_throw_dict,
                          inherit_properties = True,
                          new_grid_title = 'ttt_f5 horst')
      assert g is not None

      # scaled version of asymmetrical horst block
      model = rq.Model(epc)
      grid = model.grid(title = 'ttt_f5 horst')
      assert grid is not None
      gcs_uuids = model.uuids(obj_type = 'GridConnectionSetRepresentation', related_uuid = grid.uuid)
      assert gcs_uuids
      scaling_dict = {'ttt_f4a': 3.0, 'ttt_f4b': 1.7}
      for i, gcs_uuid in enumerate(gcs_uuids):
         gcs = rqf.GridConnectionSet(model, uuid = gcs_uuid)
         rqdm.fault_throw_scaling(epc,
                                  source_grid = grid,
                                  scaling_factor = None,
                                  connection_set = gcs,
                                  scaling_dict = scaling_dict,
                                  ref_k0 = 0,
                                  ref_k_faces = 'top',
                                  cell_range = 0,
                                  offset_decay = 0.5,
                                  store_displacement = False,
                                  inherit_properties = True,
                                  inherit_realization = None,
                                  inherit_all_realizations = False,
                                  new_grid_title = f'ttt_f6 scaled {i+1}',
                                  new_epc_file = None)
         model = rq.Model(epc)
         grid = model.grid(title = f'ttt_f6 scaled {i+1}')
         assert grid is not None

      # two intersecting straight faults
      a = np.array([[-0.2, 2.0, -0.1], [3.2, 2.0, -0.1]])
      b = np.array([[1.0, -0.2, -0.1], [1.0, 3.2, -0.1]])
      f = os.path.join(tmp_path, 'ttt_f7.dat')
      write_poly(f, a)
      write_poly(f, b, mode = 'a')
      if test_mode == 'file':
         write_poly(f, a)
         write_poly(f, b, mode = 'a')
         lines_file_list = [f]
         polylines = None
      else:
         lines_file_list = None
         polylines = make_poly(model, a, 'ttt_f7_1', crs) + make_poly(model, b, 'ttt_f7_2', crs)
      g = rqdm.add_faults(epc,
                          source_grid = None,
                          polylines = polylines,
                          lines_file_list = lines_file_list,
                          inherit_properties = True,
                          new_grid_title = 'ttt_f7')

      # re-open and check a few things
      model = rq.Model(epc)
      assert len(model.titles(obj_type = 'IjkGridRepresentation')) == 8
      g1 = model.grid(title = 'ttt_f7')
      assert g1.split_pillars_count == 5
      cpm = g1.create_column_pillar_mapping()
      assert cpm.shape == (3, 3, 2, 2)
      extras = (cpm >= 16)
      assert np.count_nonzero(extras) == 7
      assert np.all(np.sort(np.unique(cpm)) == np.arange(21))
