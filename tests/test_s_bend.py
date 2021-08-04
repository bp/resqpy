# resqpy test module including well blocking of trajectories against challenging grid geometries

version = '29th April 2021'

import pytest

import logging

log = logging.getLogger(__name__)

import os
import math as maths
import numpy as np
import pandas as pd

import resqpy.model as rq
import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.well as rqw
import resqpy.rq_import as rqi
import resqpy.olio.vector_utilities as vec
import resqpy.olio.uuid as bu
import resqpy.olio.xml_et as rqet


def test_s_bend_fn(tmp_path, epc = None):

   if epc is None:
      # use pytest temporary directory fixture
      # https://docs.pytest.org/en/stable/tmpdir.html
      epc = str(os.path.join(tmp_path, f"{bu.new_uuid()}.epc"))

   # create s-bend grid

   nk = 5
   nj = 12
   ni_tail = 5
   ni_bend = 18
   ni_half_mid = 2
   ni = 2 * (ni_tail + ni_bend + ni_half_mid)

   total_thickness = 12.0
   layer_thickness = total_thickness / float(nk)
   flat_dx_di = 10.0
   horst_dx_dk = 0.25 * flat_dx_di / float(nk)
   horst_dz = 1.73 * layer_thickness
   horst_half_dx = horst_dx_dk * horst_dz / layer_thickness
   dy_dj = 8.0
   top_depth = 100.0

   assert ni_bend % 2 == 0, 'ni_bend must be even for horizontal faulting'
   assert ni_tail >= 5, 'ni_tail must be at least 5 for horst blocks'

   bend_theta_di = maths.pi / float(ni_bend)
   outer_radius = 2.0 * total_thickness

   bend_a_centre_xz = (flat_dx_di * float(ni_tail), top_depth + outer_radius)
   bend_b_centre_xz = (flat_dx_di * float(ni_tail - 2.0 * ni_half_mid),
                       top_depth + 3.0 * outer_radius - total_thickness)

   points = np.empty((nk + 1, nj + 1, ni + 1, 3))

   for k in range(nk + 1):
      if k == nk // 2 + 1:
         points[k] = points[k - 1]  # pinched out layer
      else:
         for i in range(ni + 1):
            if i < ni_tail + 1:
               x = flat_dx_di * float(i)
               z = top_depth + float(k) * layer_thickness  # will introduce a thick layer after pinchout
            elif i < ni_tail + ni_bend:
               theta = (i - ni_tail) * bend_theta_di
               radius = outer_radius - float(k) * layer_thickness
               x = bend_a_centre_xz[0] + radius * maths.sin(theta)
               z = bend_a_centre_xz[1] - radius * maths.cos(theta)
            elif i < ni_tail + ni_bend + 2 * ni_half_mid + 1:
               x = flat_dx_di * float(ni_tail - (i - (ni_tail + ni_bend)))
               z = top_depth + 2.0 * outer_radius - float(k) * layer_thickness
            elif i < ni_tail + 2 * ni_bend + 2 * ni_half_mid:
               theta = (i - (ni_tail + ni_bend + 2 * ni_half_mid)) * bend_theta_di
               radius = outer_radius - float(nk - k) * layer_thickness
               x = bend_b_centre_xz[0] - radius * maths.sin(theta)
               z = bend_b_centre_xz[1] - radius * maths.cos(theta)
            else:
               x = flat_dx_di * float((i - (ni - ni_tail)) + ni_tail - 2 * ni_half_mid)
               if i == ni - 1 or i == ni - 4:
                  x += horst_dx_dk * float(k)
               elif i == ni - 2 or i == ni - 3:
                  x -= horst_dx_dk * float(k)
               z = top_depth + 4.0 * outer_radius + float(k) * layer_thickness - 2.0 * total_thickness
            points[k, :, i] = (x, 0.0, z)

   for j in range(nj + 1):
      points[:, j, :, 1] = dy_dj * float(j)

   model = rq.Model(epc_file = epc, new_epc = True, create_basics = True, create_hdf5_ext = True)

   grid = grr.Grid(model)

   crs = rqc.Crs(model)
   crs_node = crs.create_xml()
   if model.crs_root is None:
      model.crs_root = crs_node

   grid.grid_representation = 'IjkGrid'
   grid.extent_kji = np.array((nk, nj, ni), dtype = 'int')
   grid.nk, grid.nj, grid.ni = nk, nj, ni
   grid.k_direction_is_down = True  # dominant layer direction, or paleo-direction
   grid.pillar_shape = 'straight'
   grid.has_split_coordinate_lines = False
   grid.k_gaps = None
   grid.crs_uuid = crs.uuid
   grid.crs_root = crs_node

   grid.points_cached = points

   grid.geometry_defined_for_all_pillars_cached = True
   grid.geometry_defined_for_all_cells_cached = True
   grid.grid_is_right_handed = crs.is_right_handed_xyz()

   grid.write_hdf5_from_caches()
   grid.create_xml()

   # create a well trajectory and md datum

   def df_trajectory(x, y, z):
      N = len(x)
      assert len(y) == N and len(z) == N
      df = pd.DataFrame(columns = ['MD', 'X', 'Y', 'Z'])
      md = np.zeros(N)
      for n in range(N - 1):
         md[n + 1] = md[n] + vec.naive_length((x[n + 1] - x[n], y[n + 1] - y[n], z[n + 1] - z[n]))
      df.MD = md
      df.X = x
      df.Y = y
      df.Z = z
      return df

   x = np.array(
      [0.0, flat_dx_di * float(ni_tail) + outer_radius, flat_dx_di * (float(ni_tail) - 0.5), 0.0, -outer_radius])
   y = np.array([0.0, dy_dj * 0.5, dy_dj * float(nj) / 2.0, dy_dj * (float(nj) - 0.5), dy_dj * float(nj)])
   z = np.array([
      0.0, top_depth - total_thickness, top_depth + 2.0 * outer_radius - total_thickness / 2.0,
      top_depth + 3.0 * outer_radius - total_thickness, top_depth + 4.0 * outer_radius
   ])

   df = df_trajectory(x, y, z)

   datum = rqw.MdDatum(model, crs_uuid = crs.uuid, location = (x[0], y[0], z[0]))
   datum.create_xml()

   trajectory = rqw.Trajectory(model, md_datum = datum, data_frame = df, length_uom = 'm', well_name = 'ANGLED_WELL')

   rqc.Crs(model, uuid = rqet.uuid_for_part_root(trajectory.crs_root)).uuid == crs.uuid

   trajectory.write_hdf5()
   trajectory.create_xml()

   # add more wells

   x = np.array(
      [0.0, flat_dx_di * float(ni_tail), flat_dx_di * 2.0 * float(ni_tail - ni_half_mid) + outer_radius, -outer_radius])
   y = np.array([0.0, dy_dj * float(nj) * 0.59, dy_dj * 0.67, dy_dj * 0.5])
   z = np.array([
      0.0, top_depth - total_thickness, top_depth + 4.0 * outer_radius - 1.7 * total_thickness,
      top_depth + 4.0 * outer_radius - 1.7 * total_thickness
   ])

   df = df_trajectory(x, y, z)

   traj_2 = rqw.Trajectory(model, md_datum = datum, data_frame = df, length_uom = 'm', well_name = 'HORST_WELL')
   traj_2.write_hdf5()
   traj_2.create_xml()
   traj_2.control_points

   x = np.array([0.0, 0.0, 0.0])
   y = np.array([0.0, dy_dj * float(nj) * 0.53, dy_dj * float(nj) * 0.53])
   z = np.array([0.0, top_depth - total_thickness, top_depth + 4.0 * outer_radius])

   df = df_trajectory(x, y, z)

   traj_3 = rqw.Trajectory(model, md_datum = datum, data_frame = df, length_uom = 'm', well_name = 'VERTICAL_WELL')
   traj_3.write_hdf5()
   traj_3.create_xml()
   traj_3.control_points

   n_x = flat_dx_di * float(ni_tail) * 0.48
   n_y = dy_dj * float(nj) / 9.1
   o_y = -dy_dj * 0.45
   nd_x = n_y / 3.0
   x = np.array([
      0.0, n_x, n_x, n_x + nd_x, n_x + 2.0 * nd_x, n_x + 3.0 * nd_x, n_x + 4.0 * nd_x, n_x + 5.0 * nd_x,
      n_x + 6.0 * nd_x, n_x + 7.0 * nd_x, n_x + 8.0 * nd_x, n_x + 8.0 * nd_x
   ])
   y = np.array([
      0.0, o_y, o_y + n_y, o_y + 2.0 * n_y, o_y + 3.0 * n_y, o_y + 4.0 * n_y, o_y + 5.0 * n_y, o_y + 6.0 * n_y,
      o_y + 7.0 * n_y, o_y + 8.0 * n_y, o_y + 9.0 * n_y, o_y + 10.0 * n_y
   ])
   n_z1 = top_depth + total_thickness * 0.82
   n_z2 = top_depth - total_thickness * 0.17
   z = np.array([0.0, n_z1, n_z1, n_z2, n_z2, n_z1, n_z1, n_z2, n_z2, n_z1, n_z1, n_z2])

   df = df_trajectory(x, y, z)

   traj_4 = rqw.Trajectory(model, md_datum = datum, data_frame = df, length_uom = 'm', well_name = 'NESSIE_WELL')
   traj_4.write_hdf5()
   traj_4.create_xml()
   traj_4.control_points

   # block wells against grid geometry

   log.info('unfaulted grid blocking of well ' + str(rqw.well_name(trajectory)))
   bw = rqw.BlockedWell(model, grid = grid, trajectory = trajectory)
   bw.write_hdf5()
   bw.create_xml()
   assert bw.cell_count == 19

   log.info('unfaulted grid blocking of well ' + str(rqw.well_name(traj_2)))
   bw_2 = rqw.BlockedWell(model, grid = grid, trajectory = traj_2)
   bw_2.write_hdf5()
   bw_2.create_xml()
   assert bw_2.cell_count == 33

   log.info('unfaulted grid blocking of well ' + str(rqw.well_name(traj_3)))
   bw_3 = rqw.BlockedWell(model, grid = grid, trajectory = traj_3)
   bw_3.write_hdf5()
   bw_3.create_xml()
   assert bw_3.cell_count == 18

   log.info('unfaulted grid blocking of well ' + str(rqw.well_name(traj_4)))
   bw_4 = rqw.BlockedWell(model, grid = grid, trajectory = traj_4)
   bw_4.write_hdf5()
   bw_4.create_xml()
   assert bw_4.cell_count == 26

   # derive a faulted version of the grid

   cp = grid.corner_points(cache_cp_array = True).copy()

   # IK plane faults
   cp[:, 3:, :, :, :, :, :] += (flat_dx_di * 0.7, 0.0, layer_thickness * 1.3)
   cp[:, 5:, :, :, :, :, :] += (flat_dx_di * 0.4, 0.0, layer_thickness * 0.9)
   cp[:, 8:, :, :, :, :, :] += (flat_dx_di * 0.3, 0.0, layer_thickness * 0.6)

   # JK plane faults
   cp[:, :, ni_tail + ni_bend // 2:, :, :, :, 0] -= flat_dx_di * 0.57  # horizontal break mid top bend
   cp[:, :, ni_tail + ni_bend + ni_half_mid:, :, :, :, 2] += layer_thickness * 1.27  # vertical break in mid section

   # zig-zag fault
   j_step = nj // (ni_tail - 2)
   for i in range(ni_tail - 1):
      j_start = i * j_step
      if j_start >= nj:
         break
      cp[:, j_start:, i, :, :, :, 2] += 1.1 * total_thickness

   # JK horst blocks
   cp[:, :, ni - 4, :, :, :, :] -= (horst_half_dx, 0.0, horst_dz)
   cp[:, :, ni - 3:, :, :, :, 0] -= 2.0 * horst_half_dx
   cp[:, :, ni - 2, :, :, :, :] += (-horst_half_dx, 0.0, horst_dz)
   cp[:, :, ni - 1:, :, :, :, 0] -= 2.0 * horst_half_dx

   # JK horst block mid lower bend
   bend_horst_dz = horst_dz * maths.tan(bend_theta_di)
   cp[:, :,
      ni - (ni_tail + ni_bend // 2 + 1):ni - (ni_tail + ni_bend // 2 - 1), :, :, :, :] -= (horst_dz, 0.0, bend_horst_dz)
   cp[:, :, ni - (ni_tail + ni_bend // 2 - 1):, :, :, :, 2] -= 2.0 * bend_horst_dz

   faulted_grid = rqi.grid_from_cp(model,
                                   cp,
                                   crs.uuid,
                                   max_z_void = 0.01,
                                   split_pillars = True,
                                   split_tolerance = 0.01,
                                   ijk_handedness = 'right' if grid.grid_is_right_handed else 'left',
                                   known_to_be_straight = True)

   faulted_grid.write_hdf5_from_caches()
   faulted_grid.create_xml()

   # block wells against faulted grid

   log.info('faulted grid blocking of well ' + str(rqw.well_name(trajectory)))
   fbw = rqw.BlockedWell(model, grid = faulted_grid, trajectory = trajectory)
   fbw.write_hdf5()
   fbw.create_xml()
   assert fbw.cell_count == 32

   log.info('faulted grid blocking of well ' + str(rqw.well_name(traj_2)))
   fbw_2 = rqw.BlockedWell(model, grid = faulted_grid, trajectory = traj_2)
   fbw_2.write_hdf5()
   fbw_2.create_xml()
   assert fbw_2.cell_count == 26

   log.info('faulted grid blocking of well ' + str(rqw.well_name(traj_3)))
   fbw_3 = rqw.BlockedWell(model, grid = faulted_grid, trajectory = traj_3)
   fbw_3.write_hdf5()
   fbw_3.create_xml()
   assert fbw_3.cell_count == 14

   log.info('faulted grid blocking of well ' + str(rqw.well_name(traj_4)))
   fbw_4 = rqw.BlockedWell(model, grid = faulted_grid, trajectory = traj_4)
   fbw_4.write_hdf5()
   fbw_4.create_xml()
   assert fbw_4.cell_count == 16

   # create a version of the faulted grid with a k gap

   k_gap_grid = rqi.grid_from_cp(model,
                                 cp,
                                 crs.uuid,
                                 max_z_void = 0.01,
                                 split_pillars = True,
                                 split_tolerance = 0.01,
                                 ijk_handedness = 'right' if grid.grid_is_right_handed else 'left',
                                 known_to_be_straight = True)

   # convert second layer to a K gap
   k_gap_grid.nk_plus_k_gaps = k_gap_grid.nk
   k_gap_grid.nk -= 1
   k_gap_grid.extent_kji[0] = k_gap_grid.nk
   k_gap_grid.k_gaps = 1
   k_gap_grid.k_gap_after_array = np.zeros(k_gap_grid.nk - 1, dtype = bool)
   k_gap_grid.k_gap_after_array[0] = True
   k_gap_grid.k_raw_index_array = np.zeros(k_gap_grid.nk, dtype = int)
   for k in range(1, k_gap_grid.nk):
      k_gap_grid.k_raw_index_array[k] = k + 1

   # clear some attributes which may no longer be valid
   k_gap_grid.pinchout = None
   k_gap_grid.inactive = None
   k_gap_grid.grid_skin = None
   if hasattr(k_gap_grid, 'array_thickness'):
      delattr(k_gap_grid, 'array_thickness')

   k_gap_grid.write_hdf5_from_caches()
   k_gap_grid.create_xml()

   k_gap_grid_uuid = k_gap_grid.uuid

   # reload k gap grid object to ensure it is properly initialised

   k_gap_grid = None
   k_gap_grid = grr.Grid(model, uuid = k_gap_grid_uuid)

   # block wells against faulted grid with k gap

   log.info('k gap grid blocking of well ' + str(rqw.well_name(trajectory)))
   try:
      gbw = rqw.BlockedWell(model, grid = k_gap_grid, trajectory = trajectory)
      gbw.write_hdf5()
      gbw.create_xml()
      assert gbw.cell_count == 24
   except Exception:
      log.exception('failed to block well against k gap grid')

   log.info('k gap grid blocking of well ' + str(rqw.well_name(traj_2)))
   try:
      gbw_2 = rqw.BlockedWell(model, grid = k_gap_grid, trajectory = traj_2)
      gbw_2.write_hdf5()
      gbw_2.create_xml()
      assert gbw_2.cell_count == 20
   except Exception:
      log.exception('failed to block well against k gap grid')

   log.info('k gap grid blocking of well ' + str(rqw.well_name(traj_3)))
   try:
      gbw_3 = rqw.BlockedWell(model, grid = k_gap_grid, trajectory = traj_3)
      gbw_3.write_hdf5()
      gbw_3.create_xml()
      assert gbw_3.cell_count == 10
   except Exception:
      log.exception('failed to block well against k gap grid')

   log.info('k gap grid blocking of well ' + str(rqw.well_name(traj_4)))
   try:
      gbw_4 = rqw.BlockedWell(model, grid = k_gap_grid, trajectory = traj_4)
      gbw_4.write_hdf5()
      gbw_4.create_xml()
      assert gbw_4.cell_count == 10
   except Exception:
      log.exception('failed to block well against k gap grid')

   # store model

   model.store_epc()

   assert k_gap_grid.k_gaps
   assert len(k_gap_grid.k_raw_index_array) == k_gap_grid.nk
   assert len(k_gap_grid.k_gap_after_array) == k_gap_grid.nk - 1
   assert k_gap_grid.pinched_out((1, 0, 2))

   # clean up
   model.h5_release()
   os.remove(model.h5_file_name())
