import numpy as np
import pytest

import resqpy.grid as grr
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
from resqpy.model import Model
from resqpy.grid import Grid

import os
import resqpy.model as rq

from inspect import getsourcefile


@pytest.fixture
def model_test(tmp_path) -> Model:
    epc = os.path.join(tmp_path, 'test.epc')
    return rq.new_model(epc)


@pytest.fixture
def basic_regular_grid(model_test: Model) -> Grid:
    return grr.RegularGrid(model_test, extent_kji = (2, 2, 2), dxyz = (100.0, 50.0, 20.0), as_irregular_grid = True)


@pytest.fixture
def example_model_with_properties(tmp_path) -> Model:
    """Model with a grid (5x5x3) and properties.
   Properties:
   - Zone (discrete)
   - VPC (discrete)
   - Fault block (discrete)
   - Facies (discrete)
   - NTG (continuous)
   - POR (continuous)
   - SW (continuous)
   """
    model_path = str(tmp_path / 'test_no_rels.epc')
    model = Model(create_basics = True, create_hdf5_ext = True, epc_file = model_path, new_epc = True)
    model.store_epc(model.epc_file)

    grid = grr.RegularGrid(parent_model = model,
                           origin = (0, 0, 0),
                           extent_kji = (3, 5, 5),
                           crs_uuid = rqet.uuid_for_part_root(model.crs_root),
                           set_points_cached = True,
                           as_irregular_grid = True,
                           dxyz = (100.0, 50.0, 20.0))
    grid.cache_all_geometry_arrays()
    grid.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = False), mode = 'w')

    grid.create_xml(ext_uuid = model.h5_uuid(),
                    title = 'grid',
                    write_geometry = True,
                    add_cell_length_properties = False)
    model.store_epc()

    model = Model(model_path)

    zone = np.ones(shape = (5, 5))
    zone_array = np.array([zone, zone + 1, zone + 2], dtype = 'int')

    vpc = np.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]])
    vpc_array = np.array([vpc, vpc, vpc])

    facies = np.array([[1, 1, 1, 2, 2], [1, 1, 2, 2, 2], [1, 2, 2, 2, 3], [2, 2, 2, 3, 3], [2, 2, 3, 3, 3]])
    facies_array = np.array([facies, facies, facies])

    fb = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])
    fb_array = np.array([fb, fb, fb])

    ntg = np.array([[0, 0.5, 0, 0.5, 0], [0.5, 0, 0.5, 0, 0.5], [0, 0.5, 0, 0.5, 0], [0.5, 0, 0.5, 0, 0.5],
                    [0, 0.5, 0, 0.5, 0]])
    ntg_array = np.array([ntg, ntg, ntg])

    por = np.array([[1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5],
                    [1, 1, 1, 1, 1]])
    por_array = np.array([por, por, por])

    sat = np.array([[1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1], [1, 0.5, 1, 0.5, 1],
                    [1, 0.5, 1, 0.5, 1]])
    sat_array = np.array([sat, sat, sat])

    perm = np.array([[1, 10, 10, 100, 100], [1, 10, 10, 100, 100], [1, 10, 10, 100, 100], [1, 10, 10, 100, 100],
                     [1, 10, 10, 100, 100]])
    perm_array = np.array([perm, perm, perm], dtype = 'float')

    collection = rqp.GridPropertyCollection()
    collection.set_grid(grid)
    for array, name, kind, discrete, facet_type, facet in zip(
        [zone_array, vpc_array, fb_array, facies_array, ntg_array, por_array, sat_array, perm_array],
        ['Zone', 'VPC', 'Fault block', 'Facies', 'NTG', 'POR', 'SW', 'Perm'], [
            'discrete', 'discrete', 'discrete', 'discrete', 'net to gross ratio', 'porosity', 'saturation',
            'permeability rock'
        ], [True, True, True, True, False, False, False, False],
        [None, None, None, None, None, None, None, 'direction'], [None, None, None, None, None, None, None, 'I']):
        collection.add_cached_array_to_imported_list(cached_array = array,
                                                     source_info = '',
                                                     keyword = name,
                                                     discrete = discrete,
                                                     uom = None,
                                                     time_index = None,
                                                     null_value = None,
                                                     property_kind = kind,
                                                     facet_type = facet_type,
                                                     facet = facet,
                                                     realization = None)
        collection.write_hdf5_for_imported_list()
        collection.create_xml_for_imported_list_and_add_parts_to_model()
    model.store_epc()

    return model


@pytest.fixture()
def faulted_grid(test_data_path) -> Grid:
    current_filename = os.path.split(getsourcefile(lambda: 0))[0]
    base_folder = os.path.dirname(os.path.dirname(current_filename))
    epc_file = base_folder + '/test_data/wren/wren.epc'
    model = Model(epc_file = epc_file)
    return model.grid(title = 'faulted grid')
