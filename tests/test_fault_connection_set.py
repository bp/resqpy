import pytest
import os
import numpy as np

import resqpy.model as rq
import resqpy.grid as grr
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
   p[:, pu_pillar_count:, :] = pu[:, g1.ni + 1 : 2 * (g1.ni + 1), :]
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
   p[:, pu_pillar_count:, :] = pu[:, g2.ni + 1 : 2 * (g2.ni + 1), :]
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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]
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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]
   p[:, 2 * (grid.ni + 1):, 2] += throw
   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G4 J 0.25 juxtaposition of kji0 (*, 0, *) with (*, 1, *); and 0.75 of (1, 0, *) with (0, 1, *)')

   # model.store_epc()

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 6
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.25, 0.25],
                                      [0.75, 0.75],
                                      [0.25, 0.25],
                                      [0.25, 0.25],
                                      [0.75, 0.75],
                                      [0.25, 0.25]]), atol = 0.01))

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
   g1.create_xml(title = 'G4 I 0.25 juxtaposition of kji0 (*, *, 0) with (*, *, 1); and 0.75 of (1, *, 0) with (0, *, 1)')

   g1_fcs, g1_fa = rqtr.fault_connection_set(g1)

   assert g1_fcs is not None
   assert g1_fa is not None

   # show_fa(g1, g1_fcs, g1_fa)

   assert g1_fcs.count == 6
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(np.isclose(g1_fa,
                            np.array([[0.25, 0.25],
                                      [0.75, 0.75],
                                      [0.25, 0.25],
                                      [0.25, 0.25],
                                      [0.75, 0.75],
                                      [0.25, 0.25]]), atol = 0.01))

   # J face split with full full (1, 0, 0) with (0, 1, 0); and 0.5 of (*, 0, 1) with (*, 1, 1) and layer crossover
   # diagrams 4 & 2

   throw = 1.0

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]
   p[:, 2 * (grid.ni + 1) : -1, 2] += throw
   grid.points_cached = p
   grid.has_split_coordinate_lines = True
   grid.split_pillars_count = grid.ni + 1
   grid.split_pillar_indices_cached = np.array([i for i in range(grid.ni + 1, 2 * (grid.ni + 1))], dtype = int)
   grid.cols_for_split_pillars = np.array((2, 2, 3, 3), dtype = int)
   grid.cols_for_split_pillars_cl = np.array((1, 3, 4), dtype = int)
   grid.write_hdf5()
   grid.create_xml(title = 'G5 J full (1, 0, 0) with (0, 1, 0); and 0.5 of (*, 0, 1) with (*, 1, 1) and layer crossover')

   grid_fcs, grid_fa = rqtr.fault_connection_set(grid)

   assert grid_fcs is not None
   assert grid_fa is not None

   # show_fa(grid, grid_fcs, grid_fa)

   assert grid_fcs.count == 4
   # following assertion assumes lists are in certain order, which is not a functional requirement
   assert np.all(np.isclose(grid_fa,
                            np.array([[1. , 1. ],
                                      [0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.5, 0.5]]), atol = 0.01))

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
   p[:, pu_pillar_count : -1, 2] += throw
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
   assert np.all(np.isclose(g1_fa,
                            np.array([[1. , 1. ],
                                      [0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.5, 0.5]]), atol = 0.01))

   grid_fa

   # J face split diagram 4

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

   p[:, 2 * (grid.ni + 1) : -1, 2] += throw

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.75, 0.75],
                                      [0.25, 0.25],
                                      [0.75, 0.75]]), atol = 0.01))

   # J face split
   # diagram 5

   throw = 0.5

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.375, 0.75],
                                      [0.125, 0.125],
                                      [0.125, 0.25],
                                      [0.75, 0.75],
                                      [0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.5, 0.5]]), atol = 0.01))

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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.5, 0.5],
                                      [0.375, 0.75],
                                      [0.125, 0.125],
                                      [0.125, 0.25],
                                      [0.75, 0.75]]), atol = 0.01))

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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.5, 0.5],
                                      [1.0, 0.5],
                                      [0.5, 0.5],
                                      [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.25, 0.25],
                                      [1.0, 0.5],
                                      [0.25, 0.25],
                                      [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 0.5],
                                      [0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[11.0/16, 11.0/16],  # 0.6875, 0.6875
                                      [1.0/32, 1.0/16],    # 0.03125, 0.0625
                                      [0.5, 0.25],
                                      [0.4375, 0.4375],
                                      [0.25, 0.25],
                                      [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.25, 0.25],
                                      [1.0, 0.5],
                                      [11.0/16, 11.0/16],  # 0.6875, 0.6875
                                      [1.0/32, 1.0/16],    # 0.03125, 0.0625
                                      [0.5, 0.25],
                                      [0.4375, 0.4375]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.25, 0.25],
                                      [1.0, 0.5],
                                      [5.0/8, 5.0/8],  # 0.625, 0.625
                                      [1.0/6, 1.0/3],
                                      [1.0/3, 1.0/6],
                                      [1.0/3, 1.0/3]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = 0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[1.0/16, 1.0/8],  # 0.0625, 0.125
                                      [0.75, 0.75],
                                      [0.125, 0.125],
                                      [0.25, 0.25],
                                      [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = -0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[11.0/16, 11.0/12],
                                      [0.25, 0.5],
                                      [0.5, 0.5],
                                      [0.75, 0.75],
                                      [0.25, 0.5],
                                      [0.5, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = -0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[1.0/32, 1.0/32],
                                      [0.25, 0.5],
                                      [0.25, 0.25],
                                      [0.5, 1.0]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[1.0/32, 1.0/32],
                                      [0.5, 0.25],
                                      [0.25, 0.25],
                                      [1.0, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.75

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.25, 0.25],
                                      [1.0, 0.5],
                                      [1.0/32, 1.0/32],
                                      [0.5, 0.25]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[15.0/16, 0.75],
                                      [0.5, 0.2],
                                      [0.5, 0.5],
                                      [0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 0.5]]), atol = 0.01))

   # J face split
   # deeper layer half thickness of top layer

   throw = +0.25

   grid = grr.RegularGrid(model, (2, 2, 2), dxyz = (10.0, 10.0, 1.0))

   grid.grid_representation = 'IjkGrid'

   pu_pillar_count = (grid.nj + 1) * (grid.ni + 1)
   pu = grid.points_ref(masked = False).reshape(grid.nk + 1, pu_pillar_count, 3)
   p = np.zeros((grid.nk + 1, pu_pillar_count + grid.ni + 1, 3))
   p[:, :pu_pillar_count, :] = pu
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 0.5],
                                      [15.0/16, 0.75],
                                      [0.5, 0.2],
                                      [0.5, 0.5]]), atol = 0.01))

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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[11.0/16, 11.0/12],
                                      [1.0/16, 1.0/12],
                                      [1.0/8, 1.0/12],
                                      [7.0/8, 7.0/12],
                                      [0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 0.5]]), atol = 0.01))

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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 1.0/3],
                                      [0.5, 0.5],
                                      [1.0, 0.5]]), atol = 0.01))

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
   p[:, pu_pillar_count:, :] = pu[:, grid.ni + 1 : 2 * (grid.ni + 1), :]

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
   assert np.all(np.isclose(grid_fa,
                            np.array([[0.5, 0.5],
                                      [1.0, 0.5],
                                      [0.75, 0.75],
                                      [0.5, 0.25],
                                      [0.5, 1.0/3]]), atol = 0.01))
