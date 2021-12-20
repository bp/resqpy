import math as maths
import os

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import resqpy.derived_model as rqdm
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.vector_utilities as vec
import resqpy.property as rqp
import resqpy.time_series as rqts
import resqpy.weights_and_measures as bwam
import resqpy.surface as rqs
import resqpy.olio.xml_et as rqet
import resqpy.well as rqw
from resqpy.crs import Crs

from resqpy.property import property_kind_and_facet_from_keyword, guess_uom

# ---- Test PropertyCollection methods ---

# TODO


def test_property(tmp_path):
    epc = os.path.join(tmp_path, 'test_prop.epc')
    model = rq.new_model(epc)
    grid = grr.RegularGrid(model, extent_kji = (2, 3, 4))
    grid.write_hdf5()
    grid.create_xml()
    a1 = np.random.random(grid.extent_kji)
    p1 = rqp.Property.from_array(model,
                                 a1,
                                 source_info = 'random',
                                 keyword = 'NETGRS',
                                 support_uuid = grid.uuid,
                                 property_kind = 'net to gross ratio',
                                 indexable_element = 'cells',
                                 uom = 'm3/m3')
    a3 = np.random.random((grid.nk + 1, grid.nj + 1, grid.ni + 1))
    p3 = rqp.Property.from_array(model,
                                 a3,
                                 source_info = 'random',
                                 keyword = 'jiggle shared nodes',
                                 support_uuid = grid.uuid,
                                 property_kind = 'length',
                                 indexable_element = 'nodes',
                                 uom = 'm3')
    a4 = np.random.random((grid.nk, grid.nj, grid.ni, 2, 2, 2))
    p4 = rqp.Property.from_array(model,
                                 a4,
                                 source_info = 'random',
                                 keyword = 'jiggle nodes per cell',
                                 support_uuid = grid.uuid,
                                 property_kind = 'length',
                                 indexable_element = 'nodes per cell',
                                 uom = 'm3')
    pk = rqp.PropertyKind(model, title = 'facies', parent_property_kind = 'discrete')
    pk.create_xml()
    facies_dict = {0: 'background'}
    for i in range(1, 10):
        facies_dict[i] = 'facies ' + str(i)
    sl = rqp.StringLookup(model, int_to_str_dict = facies_dict, title = 'facies table')
    sl.create_xml()
    shape2 = tuple(list(grid.extent_kji) + [2])
    a2 = (np.random.random(shape2) * 10.0).astype(int)
    p2 = rqp.Property.from_array(model,
                                 a2,
                                 source_info = 'random',
                                 keyword = 'FACIES',
                                 support_uuid = grid.uuid,
                                 local_property_kind_uuid = pk.uuid,
                                 indexable_element = 'cells',
                                 discrete = True,
                                 null_value = 0,
                                 count = 2,
                                 string_lookup_uuid = sl.uuid)
    model.store_epc()
    model = rq.Model(epc)
    ntg_uuid = model.uuid(obj_type = p1.resqml_type, title = 'NETGRS')
    assert ntg_uuid is not None
    p1p = rqp.Property(model, uuid = ntg_uuid)
    assert np.all(p1p.array_ref() == p1.array_ref())
    assert p1p.uom() == 'm3/m3'
    facies_uuid = model.uuid(obj_type = p2.resqml_type, title = 'FACIES')
    assert facies_uuid is not None
    p2p = rqp.Property(model, uuid = facies_uuid)
    assert np.all(p2p.array_ref() == p2.array_ref())
    assert p2p.null_value() is not None and p2p.null_value() == 0
    grid = model.grid()
    jiggle_parts = model.parts(title = 'jiggle', title_mode = 'starts')
    assert len(jiggle_parts) == 2
    jiggle_shared_uuid = model.uuid(parts_list = jiggle_parts, title = 'shared', title_mode = 'contains')
    assert jiggle_shared_uuid is not None
    p3p = rqp.Property(model, uuid = jiggle_shared_uuid)
    assert p3p is not None
    assert p3p.array_ref().shape == (grid.nk + 1, grid.nj + 1, grid.ni + 1)
    jiggle_per_cell_uuid = model.uuid(parts_list = jiggle_parts, title = 'per cell', title_mode = 'ends')
    assert jiggle_per_cell_uuid is not None
    # four properties created here, plus 3 regular grid cell lengths properties
    assert grid.property_collection.number_of_parts() == 7
    collection = rqp.selective_version_of_collection(grid.property_collection,
                                                     property_kind = 'length',
                                                     uuid = jiggle_per_cell_uuid)
    assert collection is not None
    assert collection.number_of_parts() == 1
    p4p = rqp.Property.from_singleton_collection(collection)
    assert p4p is not None
    assert p4p.array_ref().shape == (grid.nk, grid.nj, grid.ni, 2, 2, 2)


def test_create_Property_from_singleton_collection(tmp_model):
    # Arrange
    grid = grr.RegularGrid(tmp_model, extent_kji = (2, 3, 4))
    grid.write_hdf5()
    grid.create_xml(add_cell_length_properties = True)
    collection = rqp.selective_version_of_collection(grid.property_collection,
                                                     property_kind = 'cell length',
                                                     facet_type = 'direction',
                                                     facet = 'J')
    assert collection.number_of_parts() == 1
    # Check property can be created
    prop = rqp.Property.from_singleton_collection(collection)
    assert np.all(np.isclose(prop.array_ref(), 1.0))


# ---- Test property kind parsing ---


def test_property_kinds():
    prop_kinds = bwam.valid_property_kinds()
    assert "angle per time" in prop_kinds
    assert "foobar" not in prop_kinds


# ---- Test infer_property_kind ---


@pytest.mark.parametrize(
    "input_name, input_unit, kind, facet_type, facet",
    [
        ('gamma ray API unit', 'gAPI', 'gamma ray API unit', None, None),  # Just a placeholder for now
        ('', '', 'Unknown', None, None),  # What should default return?
    ])
def test_infer_property_kind(input_name, input_unit, kind, facet_type, facet):
    kind_, facet_type_, facet_ = rqp.infer_property_kind(input_name, input_unit)
    assert kind == kind_
    assert facet_type == facet_type_
    assert facet == facet_


def test_bespoke_property_kind():
    model = rq.Model(create_basics = True)
    em = {'something': 'important', 'and_another_thing': 42}
    pk1 = rqp.PropertyKind(model, title = 'my kind of property', extra_metadata = em)
    pk1.create_xml()
    assert len(model.uuids(obj_type = 'PropertyKind')) == 1
    pk2 = rqp.PropertyKind(model, title = 'my kind of property', extra_metadata = em)
    pk2.create_xml()
    assert len(model.uuids(obj_type = 'PropertyKind')) == 1
    pk3 = rqp.PropertyKind(model, title = 'your kind of property', extra_metadata = em)
    pk3.create_xml()
    assert len(model.uuids(obj_type = 'PropertyKind')) == 2
    pk4 = rqp.PropertyKind(model, title = 'my kind of property')
    pk4.create_xml()
    assert len(model.uuids(obj_type = 'PropertyKind')) == 3
    pk5 = rqp.PropertyKind(model, title = 'my kind of property', parent_property_kind = 'discrete', extra_metadata = em)
    pk5.create_xml()
    assert len(model.uuids(obj_type = 'PropertyKind')) == 4
    pk6 = rqp.PropertyKind(model, title = 'my kind of property', parent_property_kind = 'continuous')
    pk6.create_xml()
    assert len(model.uuids(obj_type = 'PropertyKind')) == 4
    assert pk6 == pk4


def test_string_lookup():
    model = rq.Model(create_basics = True)
    d = {1: 'peaty', 2: 'muddy', 3: 'sandy', 4: 'granite'}
    sl = rqp.StringLookup(model, int_to_str_dict = d, title = 'stargazing')
    sl.create_xml()
    assert sl.length() == 5 if sl.stored_as_list else 4
    assert sl.get_string(3) == 'sandy'
    assert sl.get_index_for_string('muddy') == 2
    d2 = {0: 'nothing', 27: 'something', 173: 'amazing', 1072: 'brilliant'}
    sl2 = rqp.StringLookup(model, int_to_str_dict = d2, title = 'head in the clouds')
    assert not sl2.stored_as_list
    assert sl2.length() == 4
    assert sl2.get_string(1072) == 'brilliant'
    assert sl2.get_string(555) is None
    assert sl2.get_index_for_string('amazing') == 173
    sl2.create_xml()
    assert set(model.titles(obj_type = 'StringTableLookup')) == set(['stargazing', 'head in the clouds'])
    assert sl != sl2


def test_constant_array_expansion(tmp_path):
    epc = os.path.join(tmp_path, 'boring.epc')
    model = rq.new_model(epc)
    grid = grr.RegularGrid(model, extent_kji = (2, 3, 4))
    grid.write_hdf5()
    grid.create_xml()
    p1 = rqp.Property.from_array(model,
                                 cached_array = None,
                                 source_info = 'test',
                                 keyword = 'constant pi',
                                 support_uuid = grid.uuid,
                                 property_kind = 'continuous',
                                 indexable_element = 'cells',
                                 const_value = maths.pi,
                                 expand_const_arrays = True)
    p2 = rqp.Property.from_array(model,
                                 cached_array = None,
                                 source_info = 'test',
                                 keyword = 'constant three',
                                 support_uuid = grid.uuid,
                                 property_kind = 'discrete',
                                 discrete = True,
                                 indexable_element = 'cells',
                                 const_value = 3,
                                 expand_const_arrays = False)
    model.store_epc()
    # re-open the model and check that p1 appears as an ordinary array whilst p2 is still a constant
    model = rq.Model(epc)
    p1p = rqp.Property(model, uuid = p1.uuid)
    p2p = rqp.Property(model, uuid = p2.uuid)
    assert p1p.constant_value() is None
    assert p2p.constant_value() == 3
    assert_array_almost_equal(p1p.array_ref(), np.full(grid.extent_kji, maths.pi))
    assert np.all(p2p.array_ref() == 3) and p2p.array_ref().shape == tuple(grid.extent_kji)


def test_property_extra_metadata(tmp_path):
    # set up test model with a grid
    epc = os.path.join(tmp_path, 'em_test.epc')
    model = rq.new_model(epc)
    grid = grr.RegularGrid(model, extent_kji = (2, 3, 4))
    grid.write_hdf5()
    grid.create_xml()
    model.store_epc()
    # add a grid property with some extra metadata
    a = np.arange(24).astype(float).reshape((2, 3, 4))
    em = {'important': 'something', 'also': 'nothing'}
    uuid = rqdm.add_one_grid_property_array(epc, a, 'length', title = 'nonsense', uom = 'm', extra_metadata = em)
    # re-open the model and check that extra metadata has been preserved
    model = rq.Model(epc)
    p = rqp.Property(model, uuid = uuid)
    for item in em.items():
        assert item in p.extra_metadata.items()


def test_points_properties(tmp_path):
    epc = os.path.join(tmp_path, 'points_test.epc')
    model = rq.new_model(epc)

    extent_kji = (2, 3, 4)
    ensemble_size = 5
    faulted_ensemble_size = 2
    time_series_size = 6

    # create a geological time series
    years = [int(t) for t in np.linspace(-252170000, -66000000, num = time_series_size, dtype = int)]
    ts = rqts.GeologicTimeSeries.from_year_list(model, year_list = years, title = 'Mesozoic time series')
    ts.create_xml()
    ts_uuid = ts.uuid
    assert ts.timeframe == 'geologic'

    #  create a simple grid without an explicit geometry and ensure it has a property collection initialised
    grid = grr.RegularGrid(model,
                           extent_kji = extent_kji,
                           origin = (0.0, 0.0, 1000.0),
                           dxyz = (100.0, 100.0, -10.0),
                           title = 'unfaulted grid')
    # patch K direction, even though it won't be stored in xml
    grid.k_direction_is_down = False
    grid.grid_is_right_handed = not grid.grid_is_right_handed
    # generate xml and establish a property collection for the grid
    grid.create_xml()
    if grid.property_collection is None:
        grid.property_collection = rqp.PropertyCollection(support = grid)
    pc = grid.property_collection

    # shape of points property arrays
    points_shape = tuple(list(extent_kji) + [3])

    # create a static points property with multiple realisations
    for r in range(ensemble_size):
        stress = vec.unit_vectors(np.random.random(points_shape) + 0.1)
        pc.add_cached_array_to_imported_list(stress,
                                             'random stress vectors',
                                             'stress direction',
                                             uom = 'm',
                                             property_kind = 'length',
                                             realization = r,
                                             indexable_element = 'cells',
                                             points = True)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    #  create a dynamic points property (indexable cells) related to the geological time series
    for r in range(ensemble_size):
        centres = grid.centre_point().copy()
        for time_index in range(time_series_size):
            pc.add_cached_array_to_imported_list(centres,
                                                 'dynamic cell centres',
                                                 'centres',
                                                 uom = 'm',
                                                 property_kind = 'length',
                                                 realization = r,
                                                 time_index = time_index,
                                                 indexable_element = 'cells',
                                                 points = True)
            centres[..., 2] += 100.0 * (r + 1)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts_uuid)

    #  create a dynamic points property (indexable nodes) related to the geological time series
    # also create a parallel set of active cell properties
    for r in range(ensemble_size):
        nodes = grid.points_ref().copy()
        for time_index in range(time_series_size):
            pc.add_cached_array_to_imported_list(nodes,
                                                 'dynamic nodes',
                                                 'points',
                                                 uom = 'm',
                                                 property_kind = 'length',
                                                 realization = r,
                                                 time_index = time_index,
                                                 indexable_element = 'nodes',
                                                 points = True)
            nodes[..., 2] += 100.0 * (r + 1)
            active_array = np.ones(extent_kji, dtype = bool)
            # de-activate some cells
            if extent_kji[0] > 1 and time_index < time_series_size // 2:
                active_array[extent_kji[0] // 2:] = False
            if extent_kji[2] > 1:
                active_array[:, :, r % extent_kji[2]] = False
            rqp.write_hdf5_and_create_xml_for_active_property(model,
                                                              active_array,
                                                              grid.uuid,
                                                              title = 'ACTIVE',
                                                              realization = r,
                                                              time_series_uuid = ts_uuid,
                                                              time_index = time_index)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts_uuid)

    # store the xml
    model.store_epc()

    # create a second grid being a copy of the first grid but with a fault
    fault_pillar_dict = {'the fault': [(2, 0), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4)]}
    rqdm.add_faults(epc, source_grid = grid, full_pillar_list_dict = fault_pillar_dict, new_grid_title = 'faulted grid')

    # re-open the model and build a dynamic points property for the faulted grid
    # NB. this test implicitly has the oldest geometry as the 'official' grid geometry whereas
    # the RESQML documentation states that the geometry stored for the grid should be 'current' (ie. youngest)
    model = rq.Model(epc)
    f_grid = model.grid(title = 'faulted grid')
    assert f_grid is not None
    pc = f_grid.property_collection
    for r in range(faulted_ensemble_size):
        nodes = f_grid.points_ref().copy()
        for time_index in range(time_series_size):
            pc.add_cached_array_to_imported_list(nodes,
                                                 'faulted dynamic nodes',
                                                 'faulted points',
                                                 uom = 'm',
                                                 property_kind = 'length',
                                                 realization = r,
                                                 time_index = time_index,
                                                 indexable_element = 'nodes',
                                                 points = True)
            nodes[..., 2] += 101.0 * (r + 1)  # add more and more depth for each realisation
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model(time_series_uuid = ts_uuid)

    # store the xml again
    model.store_epc()

    # re-open the model and access the grid properties
    model = rq.Model(epc)
    grid = model.grid(title = 'unfaulted grid')
    pc = grid.property_collection

    # check that the grid as stored has all cells active
    assert grid.inactive is None or np.count_nonzero(grid.inactive) == 0

    # load a static points property for a single realisation
    sample_stress_part = pc.singleton(realization = ensemble_size // 2,
                                      citation_title = 'stress direction',
                                      points = True)
    assert sample_stress_part is not None
    sample_stress = pc.single_array_ref(realization = ensemble_size // 2,
                                        citation_title = 'stress direction',
                                        points = True)
    assert sample_stress is not None and sample_stress.ndim == 4
    assert np.all(sample_stress.shape[:3] == extent_kji)
    assert sample_stress.shape[3] == 3
    assert np.count_nonzero(np.isnan(sample_stress)) == 0
    stress_uuid = pc.uuid_for_part(sample_stress_part)

    # load the same property using the Property class
    stress_p = rqp.Property(model, uuid = stress_uuid)
    assert stress_p is not None
    assert stress_p.is_points()
    sample_stress = stress_p.array_ref()
    assert sample_stress is not None and sample_stress.ndim == 4

    # select the dynamic points properties related to the geological time series and indexable by cells
    cc = rqp.selective_version_of_collection(grid.property_collection,
                                             indexable = 'cells',
                                             points = True,
                                             time_series_uuid = ts_uuid)
    assert cc.number_of_parts() == ensemble_size * time_series_size

    #  check that 5 dimensional numpy arrays can be set up, each covering time indices for a single realisation
    for r in range(ensemble_size):
        rcc = rqp.selective_version_of_collection(cc, realization = r)
        assert rcc.number_of_parts() == time_series_size
        dynamic_nodes = rcc.time_series_array_ref()
        assert dynamic_nodes.ndim == 5
        assert dynamic_nodes.shape[0] == time_series_size
        assert np.all(dynamic_nodes.shape[1:4] == np.array(extent_kji))
        assert dynamic_nodes.shape[-1] == 3
        mean_depth = np.nanmean(dynamic_nodes[..., 2])
        assert mean_depth > 1000.0

    # select the dynamic points properties related to the geological time series and indexable by nodes
    nc = rqp.selective_version_of_collection(grid.property_collection,
                                             indexable = 'nodes',
                                             points = True,
                                             time_series_uuid = ts_uuid)
    assert nc.number_of_parts() == ensemble_size * time_series_size

    # check that the cached points for the grid can be populated from a points property
    grid.make_regular_points_cached()
    r = ensemble_size // 2
    ti = time_series_size // 2
    a = nc.single_array_ref(realization = r, time_index = ti)
    grid.set_cached_points_from_property(property_collection = nc,
                                         realization = r,
                                         time_index = ti,
                                         set_inactive = True,
                                         active_collection = grid.property_collection)
    assert_array_almost_equal(grid.points_cached, nc.single_array_ref(realization = r, time_index = ti))
    # check that grid's time index has been set
    assert grid.time_index == ti
    assert bu.matching_uuids(grid.time_series_uuid, ts_uuid)
    # and that the inactive array now indicates some cells are inactive
    assert grid.inactive is not None and np.count_nonzero(grid.inactive) > 0

    #  check that 5 dimensional numpy arrays can be set up, each covering realisations for a single time index
    for ti in range(time_series_size):
        tnc = rqp.selective_version_of_collection(nc, time_index = ti)
        assert tnc.number_of_parts() == ensemble_size
        dynamic_nodes = tnc.realizations_array_ref()
        assert dynamic_nodes.ndim == 5
        assert dynamic_nodes.shape[0] == ensemble_size
        assert np.all(dynamic_nodes.shape[1:4] == np.array(extent_kji) + 1)
        assert dynamic_nodes.shape[-1] == 3
        mean_depth = np.nanmean(dynamic_nodes[..., 2])
        assert mean_depth > 1000.0

    # check the faulted grid properties
    f_grid = model.grid(title = 'faulted grid')
    assert f_grid is not None
    assert not f_grid.k_direction_is_down
    f_grid.set_k_direction_from_points()
    assert not f_grid.k_direction_is_down

    # compute and cache cell centre points
    older_centres = f_grid.centre_point()
    assert hasattr(f_grid, 'array_centre_point') and f_grid.array_centre_point is not None

    # select the dynamic points properties related to the geological time series and indexable by nodes
    fnc = rqp.selective_version_of_collection(f_grid.property_collection,
                                              indexable = 'nodes',
                                              points = True,
                                              time_series_uuid = ts_uuid)
    assert fnc.number_of_parts() == faulted_ensemble_size * time_series_size

    # check that the cached points for the faulted grid can be populated from a points property
    r = faulted_ensemble_size // 2
    ti = time_series_size - 1
    p_uuid = fnc.uuid_for_part(fnc.singleton(realization = r, time_index = ti))
    assert p_uuid is not None
    f_grid.set_cached_points_from_property(points_property_uuid = p_uuid,
                                           set_grid_time_index = True,
                                           set_inactive = False)
    assert_array_almost_equal(f_grid.points_cached, fnc.single_array_ref(realization = r, time_index = ti))

    # check that grid's time index has been set
    assert f_grid.time_index == ti
    assert bu.matching_uuids(f_grid.time_series_uuid, ts_uuid)
    # and that the centre point cache has been invalidated
    assert not hasattr(f_grid, 'array_centre_point') or f_grid.array_centre_point is None

    # re-compute centre points based on the dynamically loaded geometry
    younger_centres = f_grid.centre_point()
    assert hasattr(f_grid, 'array_centre_point') and f_grid.array_centre_point is not None

    # check that later centres are vertically below the earlier centres
    # (in this example, the depths of all cells are increasing with time)
    assert_array_almost_equal(older_centres[..., :2], younger_centres[..., :2])  # xy
    assert np.all(older_centres[..., 2] < younger_centres[..., 2])  # depths


def test_remove_part_from_dict(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    assert pc is not None
    assert len(pc.parts()) == 8
    part = pc.parts()[0]

    # Act
    pc.remove_part_from_dict(part)

    # Assert
    assert len(pc.parts()) == 7
    assert part not in pc.parts()


def test_part_str(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    assert pc is not None
    part_disc = pc.parts()[0]
    part_cont = pc.parts()[-1]
    part_facet = pc.parts()[4]

    # Act / Assert
    assert pc.part_str(part_disc) == 'discrete (Zone)'
    assert pc.part_str(part_disc, include_citation_title = False) == 'discrete'
    assert pc.part_str(part_cont) == 'saturation: water; timestep: 2 (SW)'
    assert pc.part_str(part_cont, include_citation_title = False) == 'saturation: water; timestep: 2'
    assert pc.part_str(part_facet) == 'permeability rock: J (Perm)'
    assert pc.part_str(part_facet, include_citation_title = False) == 'permeability rock: J'


def test_part_filename(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    assert pc is not None
    part_disc = pc.parts()[0]
    part_cont = pc.parts()[-1]
    part_facet = pc.parts()[4]

    # Act / Assert
    assert pc.part_filename(part_disc) == 'discrete'
    assert pc.part_filename(part_cont) == 'saturation_water_ts_2'
    assert pc.part_filename(part_facet) == 'permeability_rock_J'


def test_grid_for_part(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    assert pc is not None
    part = pc.parts()[0]

    # Act
    grid = pc.grid_for_part(part)

    # Assert
    assert grid == model.grid()


def test_all_discrete(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    assert pc is not None

    # Act / Assert
    assert not pc.all_discrete()

    # Arrange
    for part in pc.parts():
        if pc.continuous_for_part(part):
            pc.remove_part_from_dict(part)

    # Act / Assert
    assert len(pc.parts()) == 4
    assert pc.all_discrete()


def test_h5_slice(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    assert pc is not None

    # Act  / Assert
    part = pc.parts()[0]
    full = pc.cached_part_array_ref(part)

    slice = pc.h5_slice(part, (0, 0))
    assert_array_almost_equal(slice, full[0, 0])

    slice = pc.h5_slice(part, (-1, -1))
    assert_array_almost_equal(slice, full[-1, -1])


def test_h5_overwrite_slice(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    assert pc is not None
    part = pc.parts()[0]

    # Act
    slice = pc.h5_slice(part, (0, 0))
    new_slice = np.zeros(shape = slice.shape)
    pc.h5_overwrite_slice(part, array_slice = new_slice, slice_tuple = (0, 0))

    # Assert
    new_full = pc.cached_part_array_ref(part)
    assert_array_almost_equal(new_slice, new_full[0, 0])


def test_string_lookup_for_part(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    lookup = model.parts_list_of_type('obj_StringTableLookup')[0]
    assert lookup is not None
    facies_part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Facies']
    assert len(facies_part) == 1

    # Act
    lookup_uuid = pc.string_lookup_uuid_for_part(facies_part[0])

    # Assert
    assert bu.matching_uuids(lookup_uuid, model.uuid_for_part(lookup))


def test_establish_has_multiple_realisations(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    # Assert initial model has multiple
    assert pc.establish_has_multiple_realizations()
    assert pc.has_multiple_realizations()
    # Remove parts with realiations
    for part in pc.parts():
        if pc.realization_for_part(part) is not None:
            pc.remove_part_from_dict(part)
    # Assert new model has not got multiple
    assert len(pc.parts()) == 8
    assert not pc.establish_has_multiple_realizations()
    assert not pc.has_multiple_realizations()


def test_establish_has_multiple_realisations_single(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    # Remove parts with realiation is None or 0
    for part in pc.parts():
        if pc.realization_for_part(part) in [None, 0]:
            pc.remove_part_from_dict(part)
    # Assert new model has not got multiple
    assert len(pc.parts()) == 2
    assert not pc.establish_has_multiple_realizations()
    assert not pc.has_multiple_realizations()


def test_establish_time_set_kind(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    # Assert model is not a time set
    assert pc.establish_time_set_kind() == 'not a time set'

    # Remove parts where ts != 2000-01-01Z
    for part in pc.parts():
        if pc.time_index_for_part(part) != 0:
            pc.remove_part_from_dict(part)
    # Assert new model is a single time
    assert len(pc.parts()) == 1
    assert pc.establish_time_set_kind() == 'single time'


def test_discombobulated_combobulated_face_arrays(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    orig = np.array([[[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]])
    pc = model.grid().property_collection

    # Act
    combob = pc.combobulated_face_array(orig)
    assert combob.shape[-1] == 2
    assert combob.shape[-2] == 3
    discombob = pc.discombobulated_face_array(combob)
    assert discombob.shape[-1] == 6

    # Assert
    assert_array_almost_equal(orig, discombob)


def test_time_series_array_ref(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    # Trim the model to only contain properties with timesteps
    for part in pc.parts():
        if pc.time_index_for_part(part) is None:
            pc.remove_part_from_dict(part)
    # Make sure the number of properties is as expected, and they are all the same property
    assert len(pc.parts()) == 3
    assert len(set([pc.citation_title_for_part(part) for part in pc.parts()])) == 1
    # Pull out the full arrays to generate the expected output
    sw1, sw2, sw3 = [pc.cached_part_array_ref(part) for part in pc.parts()]
    expected = np.array([sw1, sw2, sw3])

    # Act
    output = pc.time_series_array_ref()

    # Assert
    assert_array_almost_equal(expected, output)


def test_inherit_parts_from_other_collection(example_model_with_prop_ts_rels):
    # Arrange
    copy_from = example_model_with_prop_ts_rels
    pc_from = copy_from.grid().property_collection

    pc_to = rqp.PropertyCollection()
    pc_to.set_support(model = copy_from, support_uuid = copy_from.grid().uuid)

    orig_from = len(pc_from.parts())

    # Act
    pc_to.inherit_parts_from_other_collection(pc_from)
    # Assert
    assert len(pc_to.parts()) == orig_from


def test_similar_parts_for_time_series_from_other_collection(example_model_with_prop_ts_rels):
    # Arrange
    copy_from = example_model_with_prop_ts_rels
    pc_from = copy_from.grid().property_collection

    pc_to = rqp.PropertyCollection()
    pc_to.set_support(model = copy_from, support_uuid = copy_from.grid().uuid)

    sw_parts = [part for part in pc_from.parts() if pc_from.citation_title_for_part(part) == 'SW']
    example_part = sw_parts[0]

    # Act
    pc_to.inherit_similar_parts_for_time_series_from_other_collection(other = pc_from, example_part = example_part)
    # Assert
    assert len(pc_to.parts()) == len(sw_parts)


def test_similar_parts_for_realizations_from_other_collection(example_model_with_prop_ts_rels):
    # Arrange
    copy_from = example_model_with_prop_ts_rels
    pc_from = copy_from.grid().property_collection

    pc_to = rqp.PropertyCollection(realization = 1)
    pc_to.set_support(model = copy_from)

    rel1_parts = [part for part in pc_from.parts() if pc_from.realization_for_part(part) == 1]
    example_part = rel1_parts[0]

    # Act
    pc_to.inherit_similar_parts_for_realizations_from_other_collection(other = pc_from, example_part = example_part)
    # Assert
    assert len(pc_to.parts()) == len(rel1_parts)


def test_property_over_time_series_from_collection(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection

    sw_parts = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'SW']
    example_part = sw_parts[0]

    # Act
    new_pc = rqp.property_over_time_series_from_collection(collection = pc, example_part = example_part)
    # Assert
    assert len(new_pc.parts()) == len(sw_parts)


def test_property_for_keyword_from_collection(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection

    sw_parts = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'SW']

    # Act
    new_pc = rqp.property_collection_for_keyword(collection = pc, keyword = 'sw')
    # Assert
    assert len(new_pc.parts()) == len(sw_parts)


def test_stringlookup_add_str(example_model_and_crs):
    # Arrange
    model, _ = example_model_and_crs
    lookup = rqp.StringLookup(parent_model = model)
    assert lookup.str_dict == {}
    # Act
    lookup.set_string(0, 'channel')
    # Assert
    assert lookup.str_dict == {0: 'channel'}


def test_create_property_set_xml(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    num_parts = len(pc.parts())

    # Act
    pc.create_property_set_xml('Grid property collection')
    model.store_epc()
    reload = rq.Model(model.epc_file)
    # Assert
    assert len(reload.parts_list_of_type('obj_PropertySet')) == 1

    # Act
    prop_set_root = reload.root_for_part(reload.parts_list_of_type('obj_PropertySet')[0])
    pset = rqp.PropertyCollection()
    pset.set_support(support = model.grid())
    pset.populate_from_property_set(prop_set_root)
    # Assert
    assert len(pset.parts()) == num_parts


def test_override_min_max(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = pc.parts()[0]

    print(pc.dict[part])
    vmin = pc.minimum_value_for_part(part)
    vmax = pc.maximum_value_for_part(part)

    # Act
    pc.override_min_max(part, min_value = vmin - 1, max_value = vmax + 1)
    # Assert
    assert pc.minimum_value_for_part(part) == vmin - 1
    assert pc.maximum_value_for_part(part) == vmax + 1


def test_set_support_mesh(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs

    ni = 5
    nj = 5
    origin = (0, 0, 0)
    di = dj = 50.0

    # make a regular mesh representation
    support = rqs.Mesh(model,
                       crs_uuid = crs.uuid,
                       mesh_flavour = 'regular',
                       ni = ni,
                       nj = nj,
                       origin = origin,
                       dxyz_dij = np.array([[di, 0.0, 0.0], [0.0, dj, 0.0]]),
                       title = 'regular mesh',
                       originator = 'Emma',
                       extra_metadata = {'testing mode': 'automated'})
    assert support is not None
    support.write_hdf5()
    support.create_xml()

    pc = rqp.PropertyCollection()
    pc.set_support(support = support)
    assert pc.support_uuid == support.uuid


# Set up expected arrays for normalized array tests
array1 = np.array([[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]] * 3)

array2 = np.where(array1 == 0, 0.5, array1)

array3 = np.array([[[1, 10, 10, 100, 100], [1, 10, 10, 100, 100], [1, 10, 10, 100, 100], [1, 10, 10, 100, 100],
                    [1, 10, 10, 100, 100]]] * 3)

array4 = np.where(array3 == 1, 0, array3)
array4 = np.where(array4 == 100, 1, array4)
array4 = np.where(array4 == 10, 0.090909, array4)

array5 = np.where(array3 == 1, 0, array3)
array5 = np.where(array5 == 100, 1, array5)
array5 = np.where(array5 == 10, 0.5, array5)

array6 = np.where(array3 == 100, np.nan, array3)
array6 = np.where(array6 == 1, 0, array6)
array6 = np.where(array6 == 10, 1, array6)

array7 = np.where(array3 == 10, np.nan, array3)
array7 = np.where(array7 == 1, 0, array7)
array7 = np.where(array7 == 100, 1, array7)


@pytest.mark.parametrize(
    'name,masked,log,discrete,trust,fix,array,emin,emax',
    [
        ('NTG', False, False, None, False, None, array1, 0, 0.5),  # Simple don't trust minmax
        ('NTG', False, False, None, True, None, array1, 0, 0.5),  # Simple trust minmax
        ('NTG', False, False, None, False, 0.5, array2, -0.5, 0.5),  # Fix 0 at 0.5
        ('Perm', False, False, None, False, None, array4, 1, 100),
        ('Perm', False, True, None, False, None, array5, 0, 2),
    ])  # Logarithmic
def test_norm_array_ref(example_model_with_properties, name, masked, log, discrete, trust, fix, array, emin, emax):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    cont = [part for part in pc.parts() if pc.citation_title_for_part(part) == name][0]

    # Act
    normed, vmin, vmax = pc.normalized_part_array(cont,
                                                  masked = masked,
                                                  use_logarithm = log,
                                                  discrete_cycle = discrete,
                                                  trust_min_max = trust,
                                                  fix_zero_at = fix)

    # Assert
    assert vmin == emin
    assert vmax == emax
    assert_array_almost_equal(normed, array)


def test_norm_array_ref_mask(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    grid = model.grid()
    # Set up a mask in the grid
    minimask = np.array([0, 0, 0, 1, 1])
    layermask = np.array([minimask] * 5)
    mask = np.array([layermask, layermask, layermask], dtype = 'bool')
    grid.inactive = mask
    pc = model.grid().property_collection
    cont = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    # Act
    normed, vmin, vmax = pc.normalized_part_array(cont, masked = True, use_logarithm = False)
    # Assert
    assert vmin == 1
    assert vmax == 10
    assert np.all(np.isclose(np.ma.array(array6, mask = np.isnan(array6)), normed))


def test_norm_array_ref_mask_equal(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    grid = model.grid()
    # Set up a mask in the grid
    minimask = np.array([1, 0, 0, 1, 1])
    layermask = np.array([minimask] * 5)
    mask = np.array([layermask, layermask, layermask], dtype = 'bool')
    grid.inactive = mask
    pc = model.grid().property_collection
    cont = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    # Act
    normed, vmin, vmax = pc.normalized_part_array(cont, masked = True, use_logarithm = False)
    # Assert
    assert vmin == 10
    assert vmax == 10
    assert_array_almost_equal(np.ones(shape = (3, 5, 5)) / 2, normed)


def test_norm_array_ref_log_mask(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    grid = model.grid()
    # Set up a mask in the grid
    minimask = np.array([0, 1, 1, 0, 0])
    layermask = np.array([minimask] * 5)
    mask = np.array([layermask, layermask, layermask], dtype = 'bool')
    grid.inactive = mask
    pc = model.grid().property_collection
    cont = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    # Act
    normed, vmin, vmax = pc.normalized_part_array(cont, masked = True, use_logarithm = True)
    # Assert
    assert vmin == 0
    assert vmax == 2
    assert np.all(np.isclose(np.ma.array(array7, mask = np.isnan(array7)), normed))


def test_normalized_part_array_discrete(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    disc = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Zone'][0]

    # Act
    normed, vmin, vmax = pc.normalized_part_array(disc, discrete_cycle = 3)
    # Assert
    assert vmin == 0
    assert vmax == 2
    assert normed[0, 0, 0] == 0.5
    assert normed[1, 0, 0] == 1
    assert normed[2, 0, 0] == 0


def test_create_xml_minmax_none(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    array = np.ones(shape = (3, 5, 5))
    array[0, 0, 0] = 2
    support_uuid = model.grid().uuid
    ext_uuid = model.h5_uuid()

    p_node = pc.create_xml(ext_uuid = ext_uuid,
                           property_array = array,
                           title = 'Tester',
                           property_kind = 'continuous',
                           support_uuid = support_uuid,
                           p_uuid = bu.new_uuid(),
                           uom = 'Euc',
                           add_min_max = True,
                           min_value = None,
                           max_value = None,
                           indexable_element = 'cells',
                           count = 1)

    assert rqet.find_tag_text(p_node, 'MinimumValue') == '1.0'
    assert rqet.find_tag_text(p_node, 'MaximumValue') == '2.0'


def test_create_xml_minmax_none_discrete(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    array = np.ones(shape = (3, 5, 5))
    array[0, 0, 0] = 2
    support_uuid = model.grid().uuid
    ext_uuid = model.h5_uuid()

    p_node = pc.create_xml(ext_uuid = ext_uuid,
                           property_array = array,
                           title = 'Tester',
                           property_kind = 'discrete',
                           support_uuid = support_uuid,
                           p_uuid = bu.new_uuid(),
                           uom = 'Euc',
                           discrete = True,
                           add_min_max = True,
                           min_value = None,
                           max_value = None,
                           indexable_element = 'cells',
                           count = 1)

    assert rqet.find_tag_text(p_node, 'MinimumValue') == '1'
    assert rqet.find_tag_text(p_node, 'MaximumValue') == '2'


def test_basic_static_property_parts_ntgnone(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act / Assert - Check it finds ntg initially
    ntg, por, permi, permj, permk = pc.basic_static_property_parts()
    assert ntg is not None
    assert pc.citation_title_for_part(ntg) == 'NTG'

    # Arrange - delete ntg
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'NTG'][0]
    pc.remove_part_from_dict(part)

    # Act / Assert - check it now finds nothing
    ntg, por, permi, permj, permk = pc.basic_static_property_parts()
    assert ntg is None


def test_basic_static_property_parts_pornone(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act / Assert - Check it finds por initially
    ntg, por, permi, permj, permk = pc.basic_static_property_parts()
    assert por is not None
    assert pc.citation_title_for_part(por) == 'POR'

    # Arrange - Delete por
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'POR'][0]
    pc.remove_part_from_dict(part)

    # Act / Assert - Check it now finds nothing
    ntg, por, permi, permj, permk = pc.basic_static_property_parts()
    assert por is None


def test_basic_static_property_parts_permnone(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act / Assert - Check it finds permi initially
    ntg, por, permi, permj, permk = pc.basic_static_property_parts()
    assert permi is not None
    assert permj is None
    assert permk is None
    assert pc.citation_title_for_part(permi) == 'Perm'

    # Arrange - Delete permi
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    pc.remove_part_from_dict(part)
    # Act / Assert - Check it now finds nothing
    ntg, por, permi, permj, permk = pc.basic_static_property_parts()
    assert permi is None
    assert permj is None
    assert permk is None


def test_basic_static_property_parts_permshared(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act
    ntg, por, permi, permj, permk = pc.basic_static_property_parts(share_perm_parts = True)

    # Assert
    assert permi is not None
    assert permj is not None
    assert permk is not None
    assert permi == permj == permk
    assert pc.citation_title_for_part(permi) == 'Perm'
    assert pc.facet_for_part(permi) == 'I'


@pytest.mark.parametrize('facet,expected_none', [('J', [True, False, True]), ('K', [True, True, False]),
                                                 ('IJ', [False, False, True]), ('IJK', [False, False, False]),
                                                 ('Invalid', [False, True, True])])
def test_basic_static_property_parts_perm_facet(example_model_with_properties, facet, expected_none):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    pc.remove_part_from_dict(part)

    pc.add_cached_array_to_imported_list(cached_array = array,
                                         source_info = '',
                                         keyword = 'Testfacet',
                                         discrete = False,
                                         property_kind = 'permeability rock',
                                         facet_type = 'direction',
                                         facet = facet)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    _, _, permi, permj, permk = pc.basic_static_property_parts()

    # Assert
    for actual, expected in zip([permi, permj, permk], expected_none):
        if expected:
            assert actual is None
        else:
            assert actual is not None
            assert pc.citation_title_for_part(actual) == 'Testfacet'


@pytest.mark.parametrize('facet_list,expected_none,expected_names',
                         [(['K', 'I'], [False, True, False], ['Testfacet_I', None, 'Testfacet_K']),
                          (['K', 'J'], [True, False, False], [None, 'Testfacet_J', 'Testfacet_K']),
                          (['IJ', 'K'], [False, False, False], ['Testfacet_IJ', 'Testfacet_IJ', 'Testfacet_K']),
                          (['IJK'], [False, False, False], ['Testfacet_IJK', 'Testfacet_IJK', 'Testfacet_IJK']),
                          (['Invalid', 'K'], [True, True, False], [None, None, 'Testfacet_K'])])
def test_basic_static_property_parts_perm_multiple_facet(example_model_with_properties, facet_list, expected_none,
                                                         expected_names):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    pc.remove_part_from_dict(part)
    for facet in facet_list:
        pc.add_cached_array_to_imported_list(cached_array = array,
                                             source_info = '',
                                             keyword = f'Testfacet_{facet}',
                                             discrete = False,
                                             property_kind = 'permeability rock',
                                             facet_type = 'direction',
                                             facet = facet)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    _, _, permi, permj, permk = pc.basic_static_property_parts()

    # Assert
    for actual, expected, name in zip([permi, permj, permk], expected_none, expected_names):
        if expected:
            assert actual is None, f'Expected none for {name}'
        else:
            assert actual is not None
            assert pc.citation_title_for_part(actual) == name


@pytest.mark.parametrize('name_list,expected_none', [(['KI', 'KJ', 'KK'], [False, False, False]),
                                                     (['KX', 'KY', 'KZ'], [False, False, False]),
                                                     (['PERMI', 'PERMJ', 'PERMK'], [False, False, False]),
                                                     (['PERMX', 'PERMY', 'PERMZ'], [False, False, False])])
def test_basic_static_property_parts_perm_multiple_name(example_model_with_properties, name_list, expected_none):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    pc.remove_part_from_dict(part)
    for name in name_list:
        pc.add_cached_array_to_imported_list(cached_array = array,
                                             source_info = '',
                                             keyword = name,
                                             discrete = False,
                                             property_kind = 'permeability rock')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    _, _, permi, permj, permk = pc.basic_static_property_parts()

    # Assert
    for actual, expected, name in zip([permi, permj, permk], expected_none, name_list):
        if expected:
            assert actual is None, f'Expected none for {name}'
        else:
            assert actual is not None
            assert pc.citation_title_for_part(actual) == name


@pytest.mark.parametrize('name_list,facet', [(['KI', 'KX'], 'I'), (['KJ', 'KY'], 'J'), (['KZ', 'KK'], 'K')])
def test_basic_static_property_parts_perm_repeat(example_model_with_properties, name_list, facet):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    pc.remove_part_from_dict(part)
    for name in name_list:
        pc.add_cached_array_to_imported_list(cached_array = array,
                                             source_info = '',
                                             keyword = name,
                                             discrete = False,
                                             facet_type = 'direction',
                                             facet = facet,
                                             property_kind = 'permeability rock')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    _, _, permi, permj, permk = pc.basic_static_property_parts()

    # Assert
    assert permi is None
    assert permj is None
    assert permk is None


def test_basic_static_property_parts_perm_options(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    ntgpart = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'NTG'][0]
    ntgarray = pc.cached_part_array_ref(ntgpart)
    pc.add_cached_array_to_imported_list(cached_array = array,
                                         source_info = '',
                                         keyword = 'PermK',
                                         discrete = False,
                                         facet_type = 'direction',
                                         facet = 'J',
                                         property_kind = 'permeability rock')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    _, _, permi, permj, permk = pc.basic_static_property_parts(perm_k_mode = 'none')
    assert permi is not None
    assert permj is not None
    assert permk is None

    _, _, permi, permj, permk = pc.basic_static_property_parts(perm_k_mode = None)
    assert permi is not None
    assert permj is not None
    assert permk is None

    _, _, permi, permj, permk = pc.basic_static_property_parts(perm_k_mode = 'ratio', perm_k_ratio = 0.5)
    assert permi is not None
    assert permj is not None
    assert permk is not None
    karray = pc.cached_part_array_ref(permk)
    assert_array_almost_equal(karray, array * 0.5)


def test_basic_static_property_parts_perm_options_ntg(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    ntgpart = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'NTG'][0]
    ntgarray = pc.cached_part_array_ref(ntgpart)
    pc.add_cached_array_to_imported_list(cached_array = array,
                                         source_info = '',
                                         keyword = 'PermK',
                                         discrete = False,
                                         facet_type = 'direction',
                                         facet = 'J',
                                         property_kind = 'permeability rock')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    ntg, _, permi, permj, permk = pc.basic_static_property_parts(perm_k_mode = 'ntg', perm_k_ratio = 0.5)
    assert permi is not None
    assert permj is not None
    assert permk is not None
    karray = pc.cached_part_array_ref(permk)
    assert_array_almost_equal(karray, (array / 2) * ntgarray)


def test_basic_static_property_parts_perm_options_ntgsquared(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'Perm'][0]
    array = pc.cached_part_array_ref(part)
    ntgpart = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'NTG'][0]
    ntgarray = pc.cached_part_array_ref(ntgpart)
    pc.add_cached_array_to_imported_list(cached_array = array,
                                         source_info = '',
                                         keyword = 'PermK',
                                         discrete = False,
                                         facet_type = 'direction',
                                         facet = 'J',
                                         property_kind = 'permeability rock')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    _, _, permi, permj, permk = pc.basic_static_property_parts(perm_k_mode = 'ntg squared', perm_k_ratio = 0.5)
    assert permi is not None
    assert permj is not None
    assert permk is not None
    karray = pc.cached_part_array_ref(permk)
    assert_array_almost_equal(karray, (array / 2) * (ntgarray * ntgarray))


@pytest.mark.parametrize('keyword,kind,facet_type,facet', [('bv', 'rock volume', 'netgross', 'gross'),
                                                           ('brv', 'rock volume', 'netgross', 'gross'),
                                                           ('pv', 'pore volume', None, None),
                                                           ('pvr', 'pore volume', None, None),
                                                           ('porv', 'pore volume', None, None),
                                                           ('mdep', 'depth', 'what', 'cell centre'),
                                                           ('depth', 'depth', 'what', 'cell top'),
                                                           ('tops', 'depth', 'what', 'cell top'),
                                                           ('mids', 'depth', 'what', 'cell centre'),
                                                           ('ntg', 'net to gross ratio', None, None),
                                                           ('netgrs', 'net to gross ratio', None, None),
                                                           ('netv', 'rock volume', 'netgross', 'net'),
                                                           ('nrv', 'rock volume', 'netgross', 'net'),
                                                           ('dzc', 'thickness', 'netgross', 'gross'),
                                                           ('dzn', 'thickness', 'netgross', 'net'),
                                                           ('dz', 'thickness', 'netgross', 'gross'),
                                                           ('dznet', 'thickness', 'netgross', 'net'),
                                                           ('dxc', 'cell length', 'direction', 'I'),
                                                           ('dyc', 'cell length', 'direction', 'J'),
                                                           ('dx', 'cell length', 'direction', 'I'),
                                                           ('dy', 'cell length', 'direction', 'J'),
                                                           ('dxaaa', 'length', 'direction', 'X'),
                                                           ('dyaaa', 'length', 'direction', 'Y'),
                                                           ('dzaaa', 'length', 'direction', 'Z'),
                                                           ('por', 'porosity', None, None),
                                                           ('poro', 'porosity', None, None),
                                                           ('porosity', 'porosity', None, None),
                                                           ('kh', 'permeability thickness', None, None),
                                                           ('transx', 'transmissibility', 'direction', 'I'),
                                                           ('tx', 'transmissibility', 'direction', 'I'),
                                                           ('ty', 'transmissibility', 'direction', 'J'),
                                                           ('tz', 'transmissibility', 'direction', 'K'),
                                                           ('ti', 'transmissibility', 'direction', 'I'),
                                                           ('tj', 'transmissibility', 'direction', 'J'),
                                                           ('tk', 'transmissibility', 'direction', 'K'),
                                                           ('p', 'pressure', None, None),
                                                           ('pressure', 'pressure', None, None),
                                                           ('sw', 'saturation', 'what', 'water'),
                                                           ('so', 'saturation', 'what', 'oil'),
                                                           ('sg', 'saturation', 'what', 'gas'),
                                                           ('satw', 'saturation', 'what', 'water'),
                                                           ('sato', 'saturation', 'what', 'oil'),
                                                           ('satg', 'saturation', 'what', 'gas'),
                                                           ('soil', 'saturation', 'what', 'oil'),
                                                           ('swl', 'saturation', 'what', 'water minimum'),
                                                           ('swr', 'saturation', 'what', 'water residual'),
                                                           ('sgl', 'saturation', 'what', 'gas minimum'),
                                                           ('sgr', 'saturation', 'what', 'gas residual'),
                                                           ('swro', 'saturation', 'what', 'water residual to oil'),
                                                           ('sgro', 'saturation', 'what', 'gas residual to oil'),
                                                           ('sor', 'saturation', 'what', 'oil residual'),
                                                           ('swu', 'saturation', 'what', 'water maximum'),
                                                           ('sgu', 'saturation', 'what', 'gas maximum'),
                                                           ('wip', 'fluid volume', 'what', 'water'),
                                                           ('oip', 'fluid volume', 'what', 'oil'),
                                                           ('gip', 'fluid volume', 'what', 'gas'),
                                                           ('mobw', 'fluid volume', 'what', 'water (mobile)'),
                                                           ('mobo', 'fluid volume', 'what', 'oil (mobile)'),
                                                           ('mobg', 'fluid volume', 'what', 'gas (mobile)'),
                                                           ('ocip', 'fluid volume', 'what', 'oil condensate'),
                                                           ('tmx', 'transmissibility multiplier', 'direction', 'I'),
                                                           ('tmy', 'transmissibility multiplier', 'direction', 'J'),
                                                           ('tmz', 'transmissibility multiplier', 'direction', 'K'),
                                                           ('tmflx', 'transmissibility multiplier', 'direction', 'I'),
                                                           ('tmfly', 'transmissibility multiplier', 'direction', 'J'),
                                                           ('tmflz', 'transmissibility multiplier', 'direction', 'K'),
                                                           ('multx', 'transmissibility multiplier', 'direction', 'I'),
                                                           ('multy', 'transmissibility multiplier', 'direction', 'J'),
                                                           ('multz', 'transmissibility multiplier', 'direction', 'K'),
                                                           ('multbv', 'property multiplier', 'what', 'rock volume'),
                                                           ('multpv', 'property multiplier', 'what', 'pore volume'),
                                                           ('rs', 'solution gas-oil ratio', None, None),
                                                           ('rv', 'vapor oil-gas ratio', None, None),
                                                           ('temp', 'thermodynamic temperature', None, None),
                                                           ('temperature', 'thermodynamic temperature', None, None),
                                                           ('dad', 'code', 'what', 'dad'),
                                                           ('kid', 'code', 'what', 'inactive'),
                                                           ('unpack', 'code', 'what', 'unpack'),
                                                           ('deadcell', 'code', 'what', 'inactive'),
                                                           ('inactive', 'code', 'what', 'inactive'),
                                                           ('livecell', 'active', None, None),
                                                           ('act test', 'active', None, None),
                                                           ('ireg', 'region initialization', None, None),
                                                           ('region', 'region initialization', None, None),
                                                           ('cregion', 'region initialization', None, None),
                                                           ('uid', 'index', 'what', 'uid'),
                                                           ('noneoftheabove', None, None, None)])
def test_property_kind_and_facet_from_keyword(keyword, kind, facet_type, facet):
    out_kind, out_type, out_facet = property_kind_and_facet_from_keyword(keyword)
    assert out_kind == kind
    assert out_type == facet_type
    assert out_facet == facet


@pytest.mark.parametrize('expected,xy_uom,z_uom,property_kind,minimum,maximum,facet_type,facet',
                         [('m3', 'm', 'm', 'rock volume', None, None, None, None),
                          ('m3', 'm', 'm', 'volume', None, None, None, None),
                          ('ft3', 'ft', 'ft', 'rock volume', None, None, None, None),
                          ('ft3', 'ft', 'ft', 'volume', None, None, None, None),
                          ('m3', 'm', 'm', 'pore volume', None, None, None, None),
                          ('bbl', 'ft', 'ft', 'pore volume', None, None, None, None),
                          (None, 'm', 'ft', 'volume', None, None, None, None),
                          ('m', 'm', 'm', 'depth', None, None, None, None),
                          ('ft', 'ft', 'ft', 'depth', None, None, None, None),
                          ('m', 'ft', 'm', 'depth', None, None, None, None),
                          ('ft', 'm', 'ft', 'depth', None, None, None, None),
                          ('m', 'm', 'm', 'cell length', None, None, None, None),
                          ('ft', 'ft', 'ft', 'cell length', None, None, None, None),
                          ('%', 'm', 'ft', 'net to gross ratio', None, 50, None, None),
                          (None, 'm', 'm', 'net to gross ratio', None, -1, None, None),
                          ('m3/m3', 'm', 'm', 'net to gross ratio', None, 0.5, None, None),
                          ('ft3/ft3', 'ft', 'ft', 'net to gross ratio', None, 0.5, None, None),
                          ('Euc', 'm', 'ft', 'net to gross ratio', None, None, None, None),
                          ('%', 'm', 'm', 'porosity', None, 50, None, None),
                          ('%', 'm', 'm', 'saturation', None, 50, None, None),
                          ('mD', 'm', 'm', 'permeability rock', None, None, None, None),
                          ('mD', 'm', 'm', 'rock permeability', None, None, None, None),
                          ('mD.m', 'm', 'm', 'permeability thickness', None, None, None, None),
                          ('mD.ft', 'ft', 'ft', 'permeability thickness', None, None, None, None),
                          ('mD.m', 'm', 'm', 'permeability length', None, None, None, None),
                          ('mD.ft', 'ft', 'ft', 'permeability length', None, None, None, None),
                          (None, 'ft', 'm', 'permeability length', None, None, None, None),
                          ('m3', 'm', 'm', 'fluid volume', None, None, None, None),
                          ('1000 ft3', 'ft', 'ft', 'fluid volume', None, None, 'what', 'gas'),
                          ('bbl', 'ft', 'ft', 'fluid volume', None, None, 'what', 'oil'),
                          ('bbl', 'ft', 'ft', 'fluid volume', None, None, None, None),
                          ('m3.cP/(kPa.d)', 'm', 'm', 'transmissibility', None, None, None, None),
                          ('bbl.cP/(psi.d)', 'ft', 'ft', 'transmissibility', None, None, None, None),
                          ('kPa', 'm', 'm', 'pressure', None, None, None, None),
                          ('psi', 'ft', 'ft', 'pressure', None, None, None, None),
                          (None, 'ft', 'm', 'pressure', None, 0, None, None),
                          ('kPa', 'ft', 'm', 'pressure', None, 20000, None, None),
                          ('bar', 'ft', 'm', 'pressure', None, 450, None, None),
                          ('psi', 'ft', 'm', 'pressure', None, 4500, None, None),
                          (None, 'ft', 'm', 'pressure', None, None, None, None),
                          ('m3/m3', 'm', 'm', 'solution gas-oil ratio', None, None, None, None),
                          ('1000 ft3/bbl', 'ft', 'ft', 'solution gas-oil ratio', None, None, None, None),
                          ('m3/m3', 'm', 'm', 'vapor oil-gas ratio', None, None, None, None),
                          ('0.001 bbl/ft3', 'ft', 'ft', 'vapor oil-gas ratio', None, None, None, None),
                          ('Euc', 'm', 'm', 'some kind of multiplier', None, None, None, None),
                          (None, 'm', 'm', 'none of the above', None, None, None, None)])
def test_guess_uom(tmp_model, expected, xy_uom, z_uom, property_kind, minimum, maximum, facet_type, facet):
    model = tmp_model
    crs = Crs(parent_model = model, z_inc_down = True, xy_units = xy_uom, z_units = z_uom)
    crs.create_xml()
    support = grr.RegularGrid(parent_model = model,
                              origin = (0, 0, 0),
                              extent_kji = (3, 5, 5),
                              crs_uuid = rqet.uuid_for_part_root(model.crs_root),
                              set_points_cached = True)
    support.cache_all_geometry_arrays()
    support.write_hdf5_from_caches(file = model.h5_file_name(file_must_exist = False), mode = 'w')

    support.create_xml(ext_uuid = model.h5_uuid(),
                       title = 'grid',
                       write_geometry = True,
                       add_cell_length_properties = False)
    model.store_epc()

    uom = guess_uom(property_kind,
                    minimum = minimum,
                    maximum = maximum,
                    support = support,
                    facet_type = facet_type,
                    facet = facet)
    assert uom == expected


def test_property_kind_list(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act
    element = pc.property_kind_list()

    # Assert
    assert element == ['discrete', 'net to gross ratio', 'permeability rock', 'porosity', 'saturation']


def test_indexable_list(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act
    element = pc.unique_indexable_element_list()

    # Assert
    assert element == ['cells']


def test_facet_type_list(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act
    element = pc.facet_type_list()

    # Assert
    assert element == ['direction']


def test_facet_list(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act
    element = pc.facet_list()

    # Assert
    assert element == ['I']


def test_time_series_uuid_list(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    ts_uuid = [model.uuid_for_part(part) for part in model.parts_list_of_type('obj_TimeSeries')]

    # Act
    element = pc.time_series_uuid_list()

    # Assert
    assert element == ts_uuid


def test_uom_list(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act
    element = pc.uom_list()

    # Assert
    assert element == ['m3/m3', 'mD']


def test_string_lookup_uuid_list(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    pc = model.grid().property_collection
    lookup_uuid = [model.uuid_for_part(part) for part in model.parts_list_of_type('obj_StringTableLookup')]

    # Act
    element = pc.string_lookup_uuid_list()

    # Assert
    assert element == lookup_uuid


def test_shape_and_type_of_part(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection

    # Act / Assert
    for part in pc.parts():
        shape, dtype = pc.shape_and_type_of_part(part)
        assert shape == (3, 5, 5)
        if pc.continuous_for_part(part):
            assert dtype == '<f8'
        else:
            assert dtype == 'int32'


def test_well_interval_property_collection(example_model_and_cellio):
    # Arrange
    model, cellio_file, well_name = example_model_and_cellio
    grid = model.grid()
    bw = rqw.BlockedWell(model, use_face_centres = True)
    bw.import_from_rms_cellio(cellio_file = cellio_file, well_name = well_name, grid = grid)
    bw.write_hdf5()
    bw.create_xml()
    model.store_epc()

    pc = rqp.PropertyCollection(support = bw)

    # Add parts to collection
    zones = np.array([1, 2, 3])
    sw = np.array([0.1, 0.2, 1])
    por = np.array([0, 0.3, 0.24])
    for array, keyword, discrete in zip([zones, sw, por], ['Zone', 'sw', 'por'], [True, False, False]):
        pc.add_cached_array_to_imported_list(array,
                                             source_info = 'testing data',
                                             keyword = keyword,
                                             discrete = discrete)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()
    model.store_epc()

    # Reload the model
    reload = rq.Model(model.epc_file)
    loadbw = rqw.BlockedWell(parent_model = reload, uuid = bw.uuid)

    # Act
    wipc = rqp.WellIntervalPropertyCollection(frame = loadbw)
    df = wipc.to_pandas()

    # Assert
    assert len(list(wipc.logs())) == 3
    assert list(df.columns) == ['Zone', 'sw', 'por']
    assert all(df['Zone'].values == zones)
    assert all(df['por'].values == por)
    assert all(df['sw'].values == sw)

    for log in wipc.logs():
        if log.name == 'Zone':
            assert all(log.values() == zones)
        elif log.name == 'sw':
            assert all(log.values() == sw)
        else:
            assert all(log.values() == por)


def test_load_grid_property_collection(example_model_with_prop_ts_rels):
    # Arrange
    model = example_model_with_prop_ts_rels
    grid = model.grid()

    # Act
    pc = rqp.GridPropertyCollection(grid = grid)

    # Assert
    assert pc is not None
    assert len(pc.parts()) == 12


def test_write_read_nexus_array(example_model_with_prop_ts_rels, tmp_path):
    # Arrange
    model = example_model_with_prop_ts_rels
    grid = model.grid()
    directory = str(tmp_path)

    # Act
    pc = rqp.GridPropertyCollection(grid = grid)
    assert pc is not None
    numparts = len(pc.parts())
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'SW'][0]
    pc.write_nexus_property_generating_filename(part = part, directory = directory)

    outfile = os.path.join(tmp_path, 'SW_what_water_t_0')
    assert os.path.exists(outfile)

    with open(outfile, 'r') as f:
        lines = f.readlines()
        assert lines[1] == '! Extent of array is: [5, 5, 3]\n'
        assert lines[-1] == '1.000\t0.500\t1.000\t0.500\t1.000\n'

    _ = pc.import_nexus_property_to_cache(file_name = outfile, keyword = 'Sw1', discrete = False)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()
    model.store_epc()

    reload = rq.Model(model.epc_file)
    newpc = rqp.GridPropertyCollection(grid = reload.grid())
    assert len(newpc.parts()) == numparts + 1


def test_slice_for_box(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = rqp.GridPropertyCollection(grid = model.grid())
    assert pc is not None
    part = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'SW'][0]

    # Act
    myslice = pc.h5_slice_for_box(part = part, box = np.array([[0, 0, 0], [1, 2, 2]]))

    expected = np.array([[[1, 0.5, 1], [1, 0.5, 1], [1, 0.5, 1]], [[1, 0.5, 1], [1, 0.5, 1], [1, 0.5, 1]]])

    # Assert
    assert_array_almost_equal(expected, myslice)


def test_coarsening_length(example_fine_coarse_model):
    # Arrange
    model, coarse, fine, fc = example_fine_coarse_model
    # Remove existing length parts
    coarse_pc = rqp.GridPropertyCollection(grid = coarse)
    for part in coarse_pc.parts():
        if coarse_pc.citation_title_for_part(part) in ['DX', 'DY', 'DZ']:
            coarse_pc.remove_part_from_dict(part)
    # Check on number of properties
    numc = len(coarse_pc.parts())
    fine_pc = rqp.GridPropertyCollection(grid = fine)
    numf = len(fine_pc.parts())

    # Act
    coarse_pc.extend_imported_list_copying_properties_from_other_grid_collection(other = fine_pc, coarsening = fc)
    coarse_pc.write_hdf5_for_imported_list()
    coarse_pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Assert
    assert len(coarse_pc.parts()) == numc + numf
    all_tens = np.full(shape = (3, 5, 5), fill_value = 10, dtype = 'float')
    for length in ['DX', 'DY', 'DZ']:
        dpart = [part for part in coarse_pc.parts() if coarse_pc.citation_title_for_part(part) == length][0]
        array = coarse_pc.cached_part_array_ref(dpart)
        assert_array_almost_equal(array, all_tens)


def test_coarsening_volume(example_fine_coarse_model):
    # Arrange
    model, coarse, fine, fc = example_fine_coarse_model

    # Set up property collections
    coarse_pc = rqp.GridPropertyCollection(grid = coarse)
    numc = len(coarse_pc.parts())
    fine_pc = rqp.GridPropertyCollection(grid = fine)

    # Add a volume to the fine collection
    inarray = np.full(shape = (6, 10, 10), fill_value = 125)  # fine grid dimensions are 5x5x5 so gross volume of 125
    fine_pc.add_cached_array_to_imported_list(cached_array = inarray,
                                              source_info = '',
                                              keyword = 'brv',
                                              discrete = False,
                                              uom = 'm3/m3',
                                              property_kind = 'rock volume',
                                              facet_type = 'netgross',
                                              facet = 'gross')
    fine_pc.write_hdf5_for_imported_list()
    fine_pc.create_xml_for_imported_list_and_add_parts_to_model()
    numf = len(fine_pc.parts())

    # Act
    coarse_pc.extend_imported_list_copying_properties_from_other_grid_collection(other = fine_pc, coarsening = fc)
    coarse_pc.write_hdf5_for_imported_list()
    coarse_pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Assert
    assert len(coarse_pc.parts()) == numc + numf
    all_thousand = np.full(shape = (3, 5, 5),
                           fill_value = 1000)  # we have coarsened by 2 in all 3 directions, so expected vol of 1000
    vpart = [part for part in coarse_pc.parts() if coarse_pc.citation_title_for_part(part) == 'brv'][0]
    array = coarse_pc.cached_part_array_ref(vpart)
    assert_array_almost_equal(array, all_thousand)


# Array set up for coarsening tests
porarray = np.full(shape = (6, 10, 10), fill_value = 0.3)
porarray[0, :, :] = 0
porarray[5, :, :] = 0

expected_por = np.full(shape = (3, 5, 5), fill_value = 0.3)
expected_por[0, :, :] = 0.15
expected_por[2, :, :] = 0.15

ntgarray = np.full(shape = (6, 10, 10), fill_value = 0.5)
ntgarray[:, :, 0] = 0
ntgarray[:, :, 9] = 0

expected_ntg = np.full(shape = (3, 5, 5), fill_value = 0.5)
expected_ntg[:, :, 0] = 0.25
expected_ntg[:, :, 4] = 0.25

satarray = np.full(shape = (6, 10, 10), fill_value = 0.7)
satarray[:, 0, :] = 1
satarray[:, 9, :] = 1

expected_sat = np.full(shape = (3, 5, 5), fill_value = 0.7)
expected_sat[:, 0, :] = 0.85
expected_sat[:, 4, :] = 0.85

karray = np.full(shape = (6, 10, 10), fill_value = 1000)
karray[:, 0, :] = 100
karray[:, 9, :] = 10

expected_k = np.full(shape = (3, 5, 5), fill_value = 1000)  # simple weighted mean for now
expected_k[:, 0, :] = 550
expected_k[:, 4, :] = 505

single_disc = np.array([[1, 2], [3, 4]])  # Creating a 6x10x10 array with each 'box' of 8 cells numbered 1-8
single_layer = np.tile(np.tile(single_disc, 5).T, 5).T
discarray = np.array([single_layer, single_layer + 4, single_layer, single_layer + 4, single_layer, single_layer + 4])

expected_disc = np.ones(shape = (3, 5, 5))  # for discrete array result is the value of first cell


@pytest.mark.parametrize('inarray,keyword,discrete,kind,facettype,facet,outarray',
                         [(porarray, 'por', False, 'porosity', None, None, expected_por),
                          (ntgarray, 'NTG', False, 'net to gross ratio', None, None, expected_ntg),
                          (porarray, 'por', False, 'porosity', None, None, expected_por),
                          (satarray, 'sw', False, 'saturation', None, None, expected_sat),
                          (karray, 'kx', False, 'permeability rock', 'direction', 'I', expected_k),
                          (discarray, 'zone', True, 'discrete', None, None, expected_disc)])
def test_coarsening_reservoir_properties(example_fine_coarse_model, inarray, keyword, discrete, kind, facettype, facet,
                                         outarray):
    # Arrange
    model, coarse, fine, fc = example_fine_coarse_model

    # Set up property collections
    coarse_pc = rqp.GridPropertyCollection(grid = coarse)
    numc = len(coarse_pc.parts())
    fine_pc = rqp.GridPropertyCollection(grid = fine)

    fine_pc.add_cached_array_to_imported_list(cached_array = inarray,
                                              source_info = '',
                                              keyword = keyword,
                                              discrete = discrete,
                                              property_kind = kind,
                                              facet_type = facettype,
                                              facet = facet)

    fine_pc.write_hdf5_for_imported_list()
    fine_pc.create_xml_for_imported_list_and_add_parts_to_model()
    numf = len(fine_pc.parts())

    # Act
    coarse_pc.extend_imported_list_copying_properties_from_other_grid_collection(other = fine_pc, coarsening = fc)
    coarse_pc.write_hdf5_for_imported_list()
    coarse_pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Assert
    assert len(coarse_pc.parts()) == numc + numf

    newpart = [part for part in coarse_pc.parts() if coarse_pc.citation_title_for_part(part) == keyword][0]
    result = coarse_pc.cached_part_array_ref(newpart)

    assert_array_almost_equal(outarray, result)


def test_coarsening_realization(example_fine_coarse_model):
    # Arrange
    model, coarse, fine, fc = example_fine_coarse_model

    # Set up property collections
    coarse_pc = rqp.GridPropertyCollection(grid = coarse)
    numc = len(coarse_pc.parts())
    fine_pc = rqp.GridPropertyCollection(grid = fine)

    ntg1 = np.ones(shape = (6, 10, 10))
    ntg2 = np.zeros(shape = (6, 10, 10))

    fine_pc.add_cached_array_to_imported_list(cached_array = ntg1,
                                              source_info = '',
                                              keyword = 'ntg',
                                              discrete = False,
                                              property_kind = 'net to gross ratio',
                                              realization = 1)

    fine_pc.add_cached_array_to_imported_list(cached_array = ntg2,
                                              source_info = '',
                                              keyword = 'ntg',
                                              discrete = False,
                                              property_kind = 'net to gross ratio',
                                              realization = 2)
    fine_pc.write_hdf5_for_imported_list()
    fine_pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    coarse_pc.extend_imported_list_copying_properties_from_other_grid_collection(other = fine_pc,
                                                                                 coarsening = fc,
                                                                                 realization = 1)
    coarse_pc.write_hdf5_for_imported_list()
    coarse_pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Assert
    assert len(coarse_pc.parts()) == numc + 1

    newparts = [part for part in coarse_pc.parts() if coarse_pc.citation_title_for_part(part) == 'ntg']
    assert len(newparts) == 1
    result = coarse_pc.cached_part_array_ref(newparts[0])

    assert_array_almost_equal(np.ones(shape = (3, 5, 5)), result)


def test_import_ab_properties(example_model_with_properties, test_data_path):
    # Arrange
    model = example_model_with_properties
    pc = rqp.GridPropertyCollection(grid = model.grid())
    init_num = len(pc.parts())
    ab_facies = os.path.join(test_data_path, 'facies.ib')
    ab_ntg = os.path.join(test_data_path, 'ntg_355.db')

    # Act
    pc.import_ab_property_to_cache(ab_facies, keyword = 'ab_facies', discrete = True, property_kind = 'discrete')

    pc.import_ab_property_to_cache(ab_ntg, keyword = 'ab_ntg', discrete = False, property_kind = 'net to gross ratio')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Assert
    assert len(pc.parts()) == init_num + 2
    # Check NTG array
    ntg = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'ab_ntg'][0]
    ntg_array = pc.cached_part_array_ref(ntg)
    assert pc.continuous_for_part(ntg)
    assert pc.property_kind_for_part(ntg) == 'net to gross ratio'
    assert np.min(ntg_array) > 0.4
    assert np.max(ntg_array) < 0.7
    assert np.allclose(np.mean(ntg_array), 0.550265)

    # Check facies array
    facies = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'ab_facies'][0]
    facies_array = pc.cached_part_array_ref(facies)
    assert not pc.continuous_for_part(facies)
    assert pc.property_kind_for_part(facies) == 'discrete'
    assert np.min(facies_array) == 0
    assert np.max(facies_array) == 5
    assert np.sum(facies_array) == 170


def test_facet_array_ref(example_model_with_properties):
    # Arrange
    model = example_model_with_properties

    pc = model.grid().property_collection
    existing = [part for part in pc.parts() if pc.citation_title_for_part(part) == 'SW'][0]
    pc.remove_part_from_dict(existing)

    swarray = np.full(shape = (3, 5, 5), fill_value = 0.1)
    sgarray = np.full(shape = (3, 5, 5), fill_value = 0.2)
    soarray = np.full(shape = (3, 5, 5), fill_value = 0.7)
    for name, facet, array in zip(['sw', 'sg', 'so'], ['water', 'gas', 'oil'], [swarray, sgarray, soarray]):
        pc.add_cached_array_to_imported_list(cached_array = array,
                                             source_info = '',
                                             keyword = name,
                                             property_kind = 'saturation',
                                             facet_type = 'what',
                                             facet = facet)
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Act
    satpc = rqp.PropertyCollection()
    satpc.set_support(support = model.grid())
    satpc.inherit_parts_selectively_from_other_collection(pc, property_kind = 'saturation')

    # Assert
    assert len(satpc.parts()) == 3  # added 3 parts
    farray = satpc.facets_array_ref()
    assert farray.shape == (3, 3, 5, 5)
    names = [satpc.citation_title_for_part(part) for part in satpc.parts()]
    assert names == ['sw', 'sg', 'so']
    assert_array_almost_equal(farray[:, 0, 0, 0], np.array([0.2, 0.7, 0.1]))  # facets will be sorted so gas, oil, water


def test_copy_imported_from_other(example_model_with_properties):
    # Arrange
    model = example_model_with_properties

    pc = model.grid().property_collection

    array = np.full(shape = (3, 5, 5), fill_value = 0.1)
    pc.add_cached_array_to_imported_list(cached_array = array,
                                         source_info = '',
                                         keyword = 'testimport',
                                         property_kind = 'porosity')
    old_list = pc.imported_list

    # Act
    newpc = rqp.PropertyCollection()
    newpc.set_support(support = model.grid())
    newpc.inherit_imported_list_from_other_collection(pc)

    assert newpc.imported_list == old_list


def test_remove_cached_from_imported_list(example_model_with_properties):
    # Arrange
    model = example_model_with_properties
    pc = model.grid().property_collection
    array = np.full(shape = (3, 5, 5), fill_value = 0.1)
    pc.add_cached_array_to_imported_list(cached_array = array,
                                         source_info = '',
                                         keyword = 'testimport',
                                         property_kind = 'porosity')
    assert pc.imported_list != []
    array_name = pc.imported_list[0][3]
    assert hasattr(pc, array_name)
    # Act
    pc.remove_cached_imported_arrays()
    # Assert
    assert not hasattr(pc, array_name)


def test_mesh_support(example_model_and_crs):
    # Arrange
    model, crs = example_model_and_crs
    # create some random depths
    z = (np.random.random(3 * 3) * 20.0 + 1000.0).reshape((3, 3))
    # Create some properties
    cell_prop = np.full(shape = (2, 2), fill_value = 2)
    cell_prop[:, 0] = 1
    node_prop = np.full(shape = (3, 3), fill_value = 10)
    node_prop[:, 0] = 0

    # make a regular mesh representation
    mesh = rqs.Mesh(model,
                    crs_uuid = crs.uuid,
                    mesh_flavour = 'reg&z',
                    ni = 3,
                    nj = 3,
                    origin = (0, 0, 0),
                    dxyz_dij = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0]]),
                    z_values = z,
                    title = 'random mesh',
                    originator = 'Emma')
    assert mesh is not None
    mesh.write_hdf5()
    mesh.create_xml()

    # Act - make a property collection and set the support to be the mesh
    pc = rqp.PropertyCollection()
    pc.set_support(support = mesh)

    # Assert
    assert pc.support is not None
    assert isinstance(pc.support, rqs.Mesh)
    assert pc.support == mesh
    assert pc.support_uuid == mesh.uuid

    # Act - add different indexable element properties to collection
    pc.add_cached_array_to_imported_list(cell_prop,
                                         source_info = 'cellarray',
                                         keyword = 'TESTcell',
                                         discrete = True,
                                         property_kind = 'discrete',
                                         indexable_element = 'cells')
    pc.add_cached_array_to_imported_list(node_prop,
                                         source_info = 'nodearray',
                                         keyword = 'TESTnode',
                                         discrete = True,
                                         property_kind = 'discrete',
                                         indexable_element = 'nodes')
    pc.write_hdf5_for_imported_list()
    pc.create_xml_for_imported_list_and_add_parts_to_model()

    # Assert
    for part in pc.parts():
        if pc.citation_title_for_part(part) == 'TESTnode':
            assert pc.indexable_for_part(part) == 'nodes'
            shape, _ = pc.shape_and_type_of_part(part)
            assert shape == (3, 3)
            array = pc.cached_part_array_ref(part)
            assert_array_almost_equal(array, node_prop)
        else:
            assert pc.indexable_for_part(part) == 'cells'
            shape, _ = pc.shape_and_type_of_part(part)
            assert shape == (2, 2)
            array = pc.cached_part_array_ref(part)
            assert_array_almost_equal(array, cell_prop)
