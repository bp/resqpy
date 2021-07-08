"""Exporting to Nexus WellSpec files"""

import logging
from dataclasses import dataclass

import numpy as np
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec


log = logging.getLogger("__name__")


@dataclass
class WellSpecItems:
   """Definition of which items in a model to use in a WellSpec export"""

   ntg_uuid: str = None
   perm_i_uuid: str = None
   perm_j_uuid: str = None
   perm_k_uuid: str = None
   satw_uuid: str = None
   sato_uuid: str = None
   satg_uuid: str = None
   region_uuid: str = None


@dataclass
class WellSpecConfig:
   """Definition of how a WellSpec file should be laid out"""

   i_col: str = "IW"
   j_col: str = "JW"
   k_col: str = "L"
   one_based: bool = True
   length_mode: str = "MD"
   length_uom: str = None
   anglv_ref: str = "normal ij down"
   angla_plane_ref: str = None
   extra_columns_list: list = []   # TODO: make default arg non-mutable
   radw: float = None
   skin: float = None
   stat: str = None
   depth_inc_down: bool = None
   set_k_face_intervals_vertical: bool = False
   use_face_centres: bool = False
   preferential_perforation: bool = True


@dataclass
class WellSpecLimits:
   """Definition of filters for what should be included in a WellSpec export."""
   
   max_satw: float = None
   min_sato: float = None
   max_satg: float = None
   k0_list: list = None
   min_k0: int = None
   max_k0: int = None
   min_length: float = None
   min_kh: float = None
   max_depth: float = None
   perforation_list: list = None
   region_list: list = None
   active_only: bool = False

   def interval_matches(self, items, grid, grid_crs, cell_kji0, tuple_kji0):
      """Return true if an interval matches the limits"""

      if self.max_depth is not None:
         cell_depth = grid.centre_point(cell_kji0)[2]
         if not grid_crs.z_inc_down: cell_depth = -cell_depth
         if cell_depth > self.max_depth:
            return False
      if self.active_only and grid.inactive is not None and grid.inactive[tuple_kji0]: return False
      if (self.min_k0 is not None and cell_kji0[0] < self.min_k0) or (self.max_k0 is not None and cell_kji0[0] > self.max_k0): return False
      if self.k0_list is not None and cell_kji0[0] not in self.k0_list: return False
      if self.region_list is not None and prop_array(items.region_uuid, grid)[tuple_kji0] not in self.region_list: return False
      if self.max_satw is not None and prop_array(items.satw_uuid, grid)[tuple_kji0] > self.max_satw: return False
      if self.min_sato is not None and prop_array(items.sato_uuid, grid)[tuple_kji0] < self.min_sato: return False
      if self.max_satg is not None and prop_array(items.satg_uuid, grid)[tuple_kji0] > self.max_satg: return False
      return True





def validate_export_settings(
   limits: WellSpecLimits,
   items: WellSpecItems,
   config: WellSpecConfig,
   grid_list: list
):

   # Validate limits

   if limits.min_length is not None and limits.min_length <= 0.0: limits.min_length = None
   if limits.min_kh is not None and limits.min_kh <= 0.0: limits.min_kh = None
   if limits.max_satw is not None and limits.max_satw >= 1.0: limits.max_satw = None
   if limits.min_sato is not None and limits.min_sato <= 0.0: limits.min_sato = None
   if limits.max_satg is not None and limits.max_satg >= 1.0: limits.max_satg = None

   # Validate Items
   if limits.region_list is not None:
      assert items.region_uuid is not None, 'region list specified without region property array'
      
   # Validate config
   assert config.length_mode in ['MD', 'straight']
   assert config.length_uom is None or config.length_uom in ['m', 'ft']
   assert config.anglv_ref in ['gravity', 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down']

   if config.anglv_ref == 'gravity': config.anglv_ref = 'z down'
   if config.angla_plane_ref is None:
      config.angla_plane_ref = config.anglv_ref
   assert config.angla_plane_ref in ['gravity', 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down', 'normal well i+']
   if config.angla_plane_ref == 'gravity':
      config.angla_plane_ref = 'z down'
      
   if config.extra_columns_list:
      for extra in config.extra_columns_list:
         assert extra.upper() in ['GRID', 'ANGLA', 'ANGLV', 'LENGTH', 'KH', 'DEPTH', 'MD', 'X', 'Y',
                                 'SKIN', 'RADW', 'PPERF', 'RADB', 'WI', 'WBC']

   column_list = [config.i_col, config.j_col, config.k_col]
   if config.extra_columns_list:
      for extra in config.extra_columns_list:
         column_list.append(extra.upper())

   # Work out what we are "doing", more validation of settings
   doing_kh = False
   isotropic_perm = None
   if 'KH' in column_list or limits.min_kh is not None or 'WBC' in column_list:
      doing_kh = True
      assert config.perm_i_uuid is not None, 'WBC or KH requested (or minimum specified) without I direction permeabilty being specified'
   do_well_inflow = 'WI' in column_list or 'WBC' in column_list or 'RADB' in column_list
   if do_well_inflow:
      assert config.perm_i_uuid is not None, 'WI, RADB or WBC requested without I direction permeabilty being specified'
   if doing_kh or do_well_inflow:
      if config.perm_j_uuid is None and config.perm_k_uuid is None:
         isotropic_perm = True
      else:
         if config.perm_j_uuid is None: config.perm_j_uuid = config.perm_i_uuid
         if config.perm_k_uuid is None: config.perm_k_uuid = config.perm_i_uuid
         # following line assumes arguments are passed in same form; if not, some unnecessary maths might be done
         isotropic_perm = (
            bu.matching_uuids(config.perm_i_uuid, config.perm_j_uuid)
            and bu.matching_uuids(items.perm_i_uuid, items.perm_k_uuid)
         )
   if limits.max_satw is not None: assert config.satw_uuid is not None, 'water saturation limit specified without saturation property array'
   if limits.min_sato is not None: assert config.sato_uuid is not None, 'oil saturation limit specified without saturation property array'
   if limits.max_satg is not None: assert config.satg_uuid is not None, 'gas saturation limit specified without saturation property array'


   # Append further columns to column_list
   
   if config.radw is not None and 'RADW' not in column_list: column_list.append('RADW')
   if config.radw is None: config.radw = 0.25
   assert config.radw > 0.0
   if config.skin is not None and 'SKIN' not in column_list: column_list.append('SKIN')
   if config.skin is None: config.skin = 0.0
   if config.stat is not None:
      assert str(config.stat).upper() in ['ON', 'OFF']
      config.stat = str(config.stat).upper()
      if 'STAT' not in column_list: column_list.append('STAT')
   else:
      config.stat = 'ON'
   
   # TODO: check this is same as bw.number_of_grids()
   number_of_grids = len(grid_list)

   if 'GRID' not in column_list and number_of_grids > 1:
      log.error('creating blocked well dataframe without GRID column for well that intersects more than one grid')
   if 'LENGTH' in column_list and 'PPERF' in column_list and 'KH' not in column_list and limits.perforation_list is not None:
      log.warning('both LENGTH and PPERF will include effects of partial perforation; only one should be used in WELLSPEC')
   elif (limits.perforation_list is not None and 'LENGTH' not in column_list and 'PPERF' not in column_list and
         'KH' not in column_list and 'WBC' not in column_list):
      log.warning('perforation list supplied but no use of LENGTH, KH, PPERF nor WBC')
   if limits.min_k0 is None: limits.min_k0 = 0
   else: assert limits.min_k0 >= 0
   if limits.max_k0 is not None: assert limits.min_k0 <= limits.max_k0
   if limits.k0_list is not None and len(limits.k0_list) == 0:
      log.warning('no layers included for blocked well dataframe: no rows will be included')
   if limits.perforation_list is not None and len(limits.perforation_list) == 0:
      log.warning('empty perforation list specified for blocked well dataframe: no rows will be included')
   doing_angles = ('ANGLA' in column_list or 'ANGLV' in column_list or doing_kh or do_well_inflow)
   doing_xyz = ('X' in column_list or 'Y' in column_list or 'DEPTH' in column_list)
   doing_entry_exit = doing_angles or ('LENGTH' in column_list and config.length_mode == 'straight')

   return column_list, isotropic_perm, doing_entry_exit, doing_angles, doing_xyz, do_well_inflow, doing_kh


def prop_array(uuid_or_dict, grid):
   assert uuid_or_dict is not None and grid is not None
   if isinstance(uuid_or_dict, dict):
      prop_uuid = uuid_or_dict[grid.uuid]
   else:
      prop_uuid = uuid_or_dict   # uuid either in form of string or uuid.UUID
   return grid.property_collection.single_array_ref(uuid = prop_uuid)


def get_ref_vector(grid, grid_crs, cell_kji0, mode):
   # gravity = np.array((0.0, 0.0, 1.0))
   if mode == 'normal well i+': return None  # ANGLA only: option for no projection onto a plane
   ref_vector = None
   # options for anglv or angla reference: 'z down', 'z+', 'k down', 'k+', 'normal ij', 'normal ij down'
   cell_axial_vectors = None
   if not mode.startswith('z'):
      cell_axial_vectors = grid.interface_vectors_kji(cell_kji0)
   if mode == 'z+':
      ref_vector = np.array((0.0, 0.0, 1.0))
   elif mode == 'z down':
      if grid_crs.z_inc_down: ref_vector = np.array((0.0, 0.0, 1.0))
      else: ref_vector = np.array((0.0, 0.0, -1.0))
   elif mode in ['k+', 'k down']:
      ref_vector = vec.unit_vector(cell_axial_vectors[0])
      if mode == 'k down' and not grid.k_direction_is_down:
         ref_vector = -ref_vector
   else:  # normal to plane of ij axes
      ref_vector = vec.unit_vector(vec.cross_product(cell_axial_vectors[1], cell_axial_vectors[2]))
      if mode == 'normal ij down':
         if grid_crs.z_inc_down:
            if ref_vector[2] < 0.0: ref_vector = -ref_vector
         else:
            if ref_vector[2] > 0.0: ref_vector = -ref_vector
   if ref_vector is None or ref_vector[2] == 0.0:
      if grid_crs.z_inc_down: ref_vector = np.array((0.0, 0.0, 1.0))
      else: ref_vector = np.array((0.0, 0.0, -1.0))
   return ref_vector



