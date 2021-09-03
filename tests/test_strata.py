# resqpy stratigraphy test

import os
import numpy as np
import pandas as pd

import resqpy.model as rq
import resqpy.strata as strata
import resqpy.grid as grr
import resqpy.crs as rqc
import resqpy.organize as rqo
import resqpy.olio.uuid as bu

# the following creates a fully related set of objects, saves to persistent storage, and retrieves


def test_strata(tmp_path):

   epc = os.path.join(tmp_path, 'strata.epc')

   model = rq.new_model(epc_file = epc)

   # earth model organisational objects

   emf = rqo.OrganizationFeature(model, feature_name = 'strata model', organization_kind = 'earth model')
   emf.create_xml()

   emi = rqo.EarthModelInterpretation(model, organization_feature = emf)
   emi.create_xml()

   # stratigraphic organisational objects
   # note: 'intrusive mud' has a spurious trailing space in the RESQML 2.0.1 xsd; resqpy accepts with or without

   unit_list = ['deep unit', 'mudstone unit', 'middle unit', 'shallow unit']
   composition_list = ['sedimentary siliclastic', 'intrusive mud', 'carbonate', 'sedimentary turbidite']
   implacement_list = ['autochtonous', 'allochtonous', 'autochtonous', 'autochtonous']
   mode_list = ['parallel to bottom', None, 'proportional between top and bottom', 'parallel to top']
   suf_list = []
   sui_list = []

   for unit_name, composition, implacement, mode in zip(unit_list, composition_list, implacement_list, mode_list):
      suf = strata.StratigraphicUnitFeature(model, title = unit_name)
      suf.create_xml()
      suf_list.append(suf)
      min_thick, max_thick = (250.0, 300.0) if unit_name == 'mudstone unit' else (None, None)
      sui = strata.StratigraphicUnitInterpretation(model,
                                                   title = unit_name,
                                                   stratigraphic_unit_feature = suf,
                                                   composition = composition,
                                                   material_implacement = implacement,
                                                   deposition_mode = mode,
                                                   min_thickness = min_thick,
                                                   max_thickness = max_thick,
                                                   thickness_uom = 'cm')
      sui.create_xml()
      sui_list.append(sui)

   # horizon organisational objects

   hf_list = []
   hi_list = []

   for unit_name in unit_list:
      base_horizon_name = 'base ' + unit_name
      hf = rqo.GeneticBoundaryFeature(model, feature_name = base_horizon_name, kind = 'horizon')
      hf.create_xml()
      hf_list.append(hf)
      hi = rqo.HorizonInterpretation(model,
                                     title = base_horizon_name,
                                     genetic_boundary_feature = hf,
                                     sequence_stratigraphy_surface = 'maximum flooding')
      hi.create_xml()
      hi_list.append(hi)

   # the main stratigraphic column rank interpretation

   scri = strata.StratigraphicColumnRank(model,
                                         earth_model_feature_uuid = emf.uuid,
                                         strata_uuid_list = [sui.uuid for sui in sui_list],
                                         title = 'stratigraphy')
   scri.set_contacts_from_horizons(horizon_uuids = [hi.uuid for hi in hi_list[1:]],
                                   older_contact_mode = 'erosion',
                                   younger_contact_mode = 'baselap')
   scri.create_xml()

   # the all encompassing stratigraphic column (in case of multiple ranks)

   sc = strata.StratigraphicColumn(model, rank_uuid_list = [scri.uuid], title = scri.title)
   sc.create_xml()

   # a small grid to relate to the stratigraphy, with a few layers per strata but a K gap for the mudstone

   crs = rqc.Crs(model)
   crs.create_xml()

   grid = grr.RegularGrid(model,
                          extent_kji = (10, 2, 3),
                          dxyz = (50.0, 50.0, 3.0),
                          origin = (0.0, 0.0, 1000.0),
                          crs_uuid = crs.uuid,
                          set_points_cached = True,
                          find_properties = False,
                          title = 'tiny grid')

   # convert one layer into a K gap
   grid.nk -= 1
   grid.extent_kji = np.array((grid.nk, grid.nj, grid.ni), dtype = int)
   grid.k_gaps = 1
   grid.nk_plus_k_gaps = grid.nk + grid.k_gaps
   grid.k_gap_after_array = np.array([False, False, False, False, False, True, False, False], dtype = bool)
   grid._set_k_raw_index_array()

   # set the stratigraphy mapping for the grid's layers (and K gap)
   # note that stratigraphic unit indices start deep (old) and increase with shallowness (young)
   grid.stratigraphic_column_rank_uuid = scri.uuid
   grid.stratigraphic_units = np.array([3, 3, 3, 3, 2, 2, 1, 0, 0, 0], dtype = int)

   # write the grid array data and create grid xml, including geometry
   # True is the default for the new stratigraphy argument anyway but included here for emphasis!
   grid.write_hdf5_from_caches(write_active = False, stratigraphy = True)
   grid.create_xml(write_geometry = True, write_active = False)

   # now write the epc

   model.store_epc()

   # re-open the model and check relationships are as they should be

   model = rq.Model(epc)
   assert model is not None
   grid = model.grid()
   assert grid is not None
   assert grid.stratigraphic_column_rank_uuid is not None
   assert grid.stratigraphic_units is not None
   assert len(grid.stratigraphic_units) == grid.nk_plus_k_gaps
   strata_column_uuid = model.uuid(obj_type = 'StratigraphicColumn')
   assert strata_column_uuid is not None and bu.matching_uuids(strata_column_uuid, sc.uuid)
   strata_column = strata.StratigraphicColumn(model, uuid = strata_column_uuid)
   assert strata_column is not None
   assert len(strata_column.ranks) == 1
   strata_column_ri_uuid = model.uuid(obj_type = 'StratigraphicColumnRankInterpretation',
                                      related_uuid = strata_column_uuid)
   assert strata_column_ri_uuid is not None and bu.matching_uuids(strata_column_ri_uuid, scri.uuid)
   assert bu.matching_uuids(strata_column_ri_uuid, strata_column.ranks[0].uuid)
   assert bu.matching_uuids(strata_column_ri_uuid, grid.stratigraphic_column_rank_uuid)
   earth_model_feature_uuid = model.uuid(obj_type = 'OrganizationFeature', related_uuid = strata_column_ri_uuid)
   assert earth_model_feature_uuid is not None
   earth_model_feature = rqo.OrganizationFeature(model, uuid = earth_model_feature_uuid)
   assert earth_model_feature.organization_kind == 'earth model'
   strata_column_ri = strata.StratigraphicColumnRank(model, uuid = strata_column_ri_uuid)
   assert strata_column_ri is not None
   sui_uuid_set = set(model.uuids(obj_type = 'StratigraphicUnitInterpretation', related_uuid = strata_column_ri_uuid))
   assert len(sui_uuid_set) == 4
   assert sui_uuid_set == set([sui.uuid for sui in strata_column_ri.iter_units()])
   hi_uuid_set = set(model.uuids(obj_type = 'HorizonInterpretation', related_uuid = strata_column_ri_uuid))
   assert len(hi_uuid_set) == 3
   pou_set = set()
   for contact in strata_column_ri.iter_contacts():
      if contact.part_of_uuid is not None:
         pou_set.add(contact.part_of_uuid)
   assert hi_uuid_set == pou_set

   # check column rank units and contacts are well ordered

   previous_unit_index = -1
   for i, sui_uuid in strata_column_ri.units:
      assert i > previous_unit_index
      previous_unit_index = i
      assert sui_uuid is not None

   previous_contact_index = -1
   for contact in strata_column_ri.contacts:
      assert contact.index > previous_contact_index
      previous_contact_index = contact.index

   # check that stratigraphy for grid layers is as expected

   # note: K gap is skipped over here
   expected_unit_names_per_layer = 4 * ['shallow unit'] + 2 * ['middle unit'] + 3 * ['deep unit']

   sui_name_list = [sui.title for sui in strata_column_ri.iter_units()]

   for k in range(grid.nk):
      unit = strata_column_ri.unit_for_unit_index(grid.stratigraphic_units[grid.k_raw_index_array[k]])
      assert unit is not None
      assert unit.title == expected_unit_names_per_layer[k]
