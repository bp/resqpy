import os

import numpy as np
import pytest

import resqpy.crs as rqc
import resqpy.grid as grr
import resqpy.model as rq
import resqpy.olio.uuid as bu
import resqpy.olio.write_hdf5 as rwh5
import resqpy.olio.xml_et as rqet
import resqpy.property as rqp
import resqpy.well as rqw


def test_model(tmp_path):

    epc = os.path.join(tmp_path, 'model.epc')
    model = rq.new_model(epc)
    assert model is not None
    crs = rqc.Crs(model)
    crs_root = crs.create_xml()
    model.store_epc()
    assert os.path.exists(epc)
    md_datum_1 = rqw.MdDatum(model, location = (0.0, 0.0, -50.0), crs_uuid = crs.uuid)
    md_datum_1.create_xml(title = 'Datum & 1')
    md_datum_2 = rqw.MdDatum(model, location = (3.0, 0.0, -50.0), crs_uuid = crs.uuid)
    md_datum_2.create_xml(title = 'Datum < 2')
    assert len(model.uuids(obj_type = 'MdDatum')) == 2
    model.store_epc()

    model = rq.Model(epc)
    assert model is not None
    assert len(model.uuids(obj_type = 'MdDatum')) == 2
    datum_part_1 = model.part(obj_type = 'MdDatum', title = '1', title_mode = 'ends')
    datum_part_2 = model.part(obj_type = 'MdDatum', title = '2', title_mode = 'ends')
    assert datum_part_1 is not None and datum_part_2 is not None and datum_part_1 != datum_part_2
    datum_uuid_1 = rqet.uuid_in_part_name(datum_part_1)
    datum_uuid_2 = rqet.uuid_in_part_name(datum_part_2)
    assert not bu.matching_uuids(datum_uuid_1, datum_uuid_2)
    p1 = model.uuid_part_dict[bu.uuid_as_int(datum_uuid_1)]
    p2 = model.uuid_part_dict[bu.uuid_as_int(datum_uuid_2)]
    assert p1 == datum_part_1 and p2 == datum_part_2


def test_model_iterators(example_model_with_well):

    model, well_interp, datum, traj = example_model_with_well

    w = next(model.iter_wellbore_interpretations())
    d = next(model.iter_md_datums())
    t = next(model.iter_trajectories())

    assert bu.matching_uuids(w.uuid, well_interp.uuid)
    assert bu.matching_uuids(d.uuid, datum.uuid)
    assert bu.matching_uuids(t.uuid, traj.uuid)
    assert w == well_interp
    assert d == datum
    assert t == traj


def test_model_iterate_objects(example_model_with_well):

    from resqpy.organize import WellboreFeature, WellboreInterpretation

    model, well_interp, _, _ = example_model_with_well

    # Try iterating over wellbore features

    wells = model.iter_objs(WellboreFeature)
    wells = list(wells)
    w = wells[0]
    assert len(wells) == 1
    assert isinstance(wells[0], WellboreFeature)
    assert wells[0].title == "well A"

    # Try iterating over wellbore interpretations

    interps = model.iter_objs(WellboreInterpretation)
    interps = list(interps)
    assert len(interps) == 1
    assert isinstance(interps[0], WellboreInterpretation)
    assert interps[0] == well_interp


def test_model_iter_crs(example_model_and_crs):

    model, crs_1 = example_model_and_crs

    crs_list = list(model.iter_crs())
    assert len(crs_list) == 1

    crs_2 = crs_list[0]
    assert crs_2 == crs_1


def test_model_iter_crs_empty(tmp_model):

    # Should raise an exception if no CRS exists
    with pytest.raises(StopIteration):
        next(tmp_model.iter_crs())


def test_model_as_graph(example_model_with_well):

    model, well_interp, datum, traj = example_model_with_well
    crs = next(model.iter_crs())

    nodes, edges = model.as_graph()

    # Check nodes
    for obj in [well_interp, datum, traj, crs]:
        assert str(obj.uuid) in nodes.keys()
        if hasattr(obj, 'title'):  # TODO: remove this when all objects have this attribute
            assert obj.title == nodes[str(obj.uuid)]['title']

    # Check edges
    assert frozenset([str(traj.uuid), str(datum.uuid)]) in edges
    assert frozenset([str(traj.uuid), str(crs.uuid)]) in edges
    assert frozenset([str(traj.uuid), str(well_interp.uuid)]) in edges
    assert frozenset([str(datum.uuid), str(traj.uuid)]) in edges

    # Test uuid subset
    nodes, edges = model.as_graph(uuids_subset = [datum.uuid])
    assert len(nodes.keys()) == 1
    assert len(edges) == 0


def test_model_copy_all_parts(example_model_with_properties):

    epc = example_model_with_properties.epc_file
    dir = example_model_with_properties.epc_directory
    copied_epc = os.path.join(dir, 'copied.epc')

    # test copying without consolidation
    original = rq.Model(epc)
    assert original is not None
    copied = rq.new_model(copied_epc)
    copied.copy_all_parts_from_other_model(original, consolidate = False)

    assert set(original.uuids()) == set(copied.uuids())
    assert set(original.parts()) == set(copied.parts())

    # test without consolidation of two crs objects
    copied = rq.new_model(copied_epc)
    new_crs = rqc.Crs(copied)
    new_crs.create_xml()

    copied.copy_all_parts_from_other_model(original, consolidate = False)

    assert len(copied.parts()) == len(original.parts()) + 1
    assert set(original.parts()).issubset(set(copied.parts()))
    assert len(copied.parts(obj_type = 'LocalDepth3dCrs')) == 2

    # test with consolidation of two crs objects
    copied = rq.new_model(copied_epc)
    new_crs = rqc.Crs(copied)
    new_crs.create_xml()

    copied.copy_all_parts_from_other_model(original, consolidate = True)

    assert len(copied.parts()) == len(original.parts())
    assert len(copied.parts(obj_type = 'LocalDepth3dCrs')) == 1

    crs_uuid = copied.uuid(obj_type = 'LocalDepth3dCrs')
    assert (bu.matching_uuids(crs_uuid, new_crs.uuid) or
            bu.matching_uuids(crs_uuid, original.uuid(obj_type = 'LocalDepth3dCrs')))

    # test write and re-load of copied model
    copied.store_epc()
    re_opened = rq.Model(copied_epc)
    assert re_opened is not None

    assert len(copied.parts()) == len(original.parts())

    crs_uuid = re_opened.uuid(obj_type = 'LocalDepth3dCrs')
    assert (bu.matching_uuids(crs_uuid, new_crs.uuid) or
            bu.matching_uuids(crs_uuid, original.uuid(obj_type = 'LocalDepth3dCrs')))


def test_model_copy_all_parts_non_resqpy_hdf5_paths(example_model_with_properties):
    # this test uses low level methods to override the hdf5 internal paths for some new objects

    epc = example_model_with_properties.epc_file
    dir = example_model_with_properties.epc_directory
    copied_epc = os.path.join(dir, 'copied.epc')
    extent_kji = (3, 5, 5)  # needs to match extent of grid in example model

    # add some properties with bespoke internal hdf5 paths
    original = rq.Model(epc)
    assert original is not None
    grid_uuid = original.uuid(obj_type = 'IjkGridRepresentation')
    assert grid_uuid is not None
    data = np.linspace(0.0, 1000.0, num = 75).reshape(extent_kji)

    hdf5_paths = [
        '/RESQML/uuid_waffle/values', '/RESQML/waffle_uuid/unusual_name', 'RESQML/class_uuid_waffle/values0',
        'RESQML/something_interesting/array', '/RESQML/abrupt', 'no_resqml/uuid/values'
    ]

    prop_list = []
    path_list = []  # elements will have 'uuid' replaced with uuid string
    h5_reg = rwh5.H5Register(original)
    for i, hdf5_path in enumerate(hdf5_paths):
        prop = rqp.Property(original, support_uuid = grid_uuid)
        if 'uuid' in hdf5_path:
            path = hdf5_path.replace('uuid', str(prop.uuid))
        else:
            path = hdf5_path
        prop.prepare_import(cached_array = data,
                            source_info = 'test data',
                            keyword = f'TEST{i}',
                            property_kind = 'continuous',
                            uom = 'm')
        for entry in prop.collection.imported_list:  # only one entry
            # override internal hdf5 path
            h5_reg.register_dataset(entry[0], 'values_patch0', prop.collection.__dict__[entry[3]], hdf5_path = path)
        prop_list.append(prop)
        path_list.append(path)
    h5_reg.write(mode = 'a')

    for prop, hdf5_path in zip(prop_list, path_list):
        root_node = prop.create_xml(find_local_property_kind = False)
        assert root_node is not None
        # override the hdf5 internal path in the xml tree
        path_node = rqet.find_nested_tags(root_node, ['PatchOfValues', 'Values', 'Values', 'PathInHdfFile'])
        assert path_node is not None
        path_node.text = hdf5_path

    # rewrite model
    original.store_epc()

    # re-open model and test copying
    original = rq.Model(epc)
    assert original is not None
    copied = rq.new_model(copied_epc)
    copied.copy_all_parts_from_other_model(original, consolidate = False)

    assert set(original.uuids()) == set(copied.uuids())
    assert set(original.parts()) == set(copied.parts())


def test_model_context(tmp_path):

    # Create a new model
    epc_path = str(tmp_path / 'tmp_model.epc')
    model = rq.new_model(epc_path)
    crs = rqc.Crs(parent_model = model, title = 'kuzcotopia')
    crs_uuid = crs.uuid
    crs.create_xml()
    model.store_epc()
    del crs, model

    # Re-open model in read/write mode
    with rq.ModelContext(epc_path, mode = "rw") as model2:

        crs2 = rqc.Crs(model2, uuid = crs_uuid)
        assert len(list(model2.iter_crs())) == 1
        assert crs2.title == 'kuzcotopia'

        # Make a change
        crs2.title = 'wabajam'
        crs2.create_xml(reuse = False)

    # Re-open model in read mode
    with rq.ModelContext(epc_path, mode = "r") as model3:

        # Check model has loaded correctly
        assert len(list(model3.iter_crs())) == 1
        crs3 = rqc.Crs(model3, uuid = crs_uuid)
        assert crs3.title == 'wabajam'

    # Overwrite model
    with rq.ModelContext(epc_path, mode = "create") as model4:
        # Should be empty
        crs_list = list(model4.iter_crs())
        assert len(crs_list) == 0


def test_multiple_epc_sharing_one_hdf5(tmp_path, example_model_with_prop_ts_rels):

    # get some key data from the full model
    full_model = example_model_with_prop_ts_rels
    full_epc_path = full_model.epc_file
    full_count = full_model.number_of_parts()
    assert full_count > 4
    hdf5_path = full_model.h5_file_name(override = 'none')
    assert os.path.exists(hdf5_path)
    ext_uuid = full_model.h5_uuid()
    assert ext_uuid is not None
    grid_uuid = full_model.uuid(obj_type = 'IjkGridRepresentation')
    assert grid_uuid is not None
    grid = full_model.grid(uuid = grid_uuid)
    pc = grid.property_collection
    assert pc is not None and pc.number_of_parts() > 7
    ts_uuid = full_model.uuid(obj_type = 'TimeSeries')
    assert ts_uuid is not None
    discrete_prop_uuid_list = full_model.uuids(obj_type = 'DiscreteProperty')
    assert len(discrete_prop_uuid_list) == 3

    # create a couple of sub model file names, one for each of two realisations
    epc_0 = os.path.join(tmp_path, 'model_0.epc')
    epc_1 = os.path.join(tmp_path, 'model_1.epc')

    # for each sub model...
    for i, epc_path in enumerate((epc_0, epc_1)):

        # create a new, empty, model without the usual new hdf5 file and external part
        model = rq.Model(epc_file = epc_path, new_epc = True, create_basics = True, create_hdf5_ext = False)

        # switch off the default hdf5 filename override
        model.h5_set_default_override('none')

        # create an hdf5 external part referring to the full model's hdf5 file, preserving the ext uuid
        ext_node = model.create_hdf5_ext(file_name = hdf5_path, uuid = ext_uuid)
        assert ext_node is not None

        # check that the correct hdf5 path will be returned for the sub model when not overriding
        sub_hdf5_path = model.h5_file_name(override = 'none')
        assert sub_hdf5_path, 'hdf5 path not established for sub model'
        assert os.path.samefile(hdf5_path, sub_hdf5_path)

        # copy some common parts into the sub model (will include referenced crs)
        for uuid in [grid_uuid, ts_uuid] + discrete_prop_uuid_list:
            model.copy_part_from_other_model(full_model,
                                             full_model.part(uuid = uuid),
                                             self_h5_file_name = hdf5_path,
                                             other_h5_file_name = hdf5_path)

        # copy some realisation specific properties
        for pk in ('net to gross ratio', 'porosity'):
            property_part = pc.singleton(property_kind = pk, realization = i)
            assert property_part is not None
            model.copy_part_from_other_model(full_model,
                                             property_part,
                                             self_h5_file_name = hdf5_path,
                                             other_h5_file_name = hdf5_path)

        # save the epc for the sub model
        model.store_epc()

        # check the number of parts in the sub model is as expected (includes crs and ext)
        assert model.number_of_parts() == 6 + len(discrete_prop_uuid_list)

    # re-open all 3 models and see how they shape up
    full_model = rq.Model(full_epc_path)
    model_0 = rq.Model(epc_0)
    model_1 = rq.Model(epc_1)
    assert full_model.number_of_parts() == full_count
    full_uuids = full_model.uuids()
    assert len(full_uuids) == full_count - 1  # adjusting for ext part which is not included in uuids()

    # for each sub model...
    for model in (model_0, model_1):
        # switch off hdf5 filename override
        model.h5_set_default_override('none')
        # check that the number of parts in the sub model is still as expected
        assert model.number_of_parts() == 6 + len(discrete_prop_uuid_list)
        # check that all sub model uuids exist in the full model, and have the same type
        for uuid in model.uuids():
            found = False
            for full_uuid in full_uuids:
                if bu.matching_uuids(uuid, full_uuid):
                    found = True
                    break
            assert found, 'new uuid found in sub model'
            assert model.type_of_uuid(uuid) == full_model.type_of_uuid(uuid)

    # establish some grid objects with property collections
    grid = full_model.grid()
    grid_0 = model_0.grid()
    assert grid == grid_0, 'grid equivalence based on uuid failed'
    grid_1 = model_1.grid()
    assert grid == grid_1, 'grid equivalence based on uuid failed'

    # check that realisation-specific properties are different in the two sub models
    for pk in ('net to gross ratio', 'porosity'):
        prop_0 = grid_0.property_collection.single_array_ref(property_kind = pk)
        assert prop_0 is not None
        prop_1 = grid_1.property_collection.single_array_ref(property_kind = pk)
        assert prop_1 is not None
        assert not np.any(np.isclose(prop_0, prop_1))


def test_one_epc_using_multiple_hdf5(tmp_path, example_model_with_prop_ts_rels):

    model = example_model_with_prop_ts_rels
    epc = model.epc_file
    model.h5_set_default_override('dir')
    grid = model.grid()
    assert grid is not None and grid.property_collection is not None
    pc = grid.property_collection

    # add a couple of new hdf5 parts
    jitter_ext_files = []
    jitter_ext_nodes = []
    jitter_ext_uuids = []
    for i in range(2):
        filename = f'jitter_{i}.h5'
        jitter_ext_files.append(filename)
        jitter_ext_nodes.append(model.create_hdf5_ext(file_name = filename))
        jitter_ext_uuids.append(bu.uuid_from_string(jitter_ext_nodes[-1].attrib['uuid']))
    assert None not in jitter_ext_nodes and None not in jitter_ext_uuids

    # generate a couple of new permeability arrays
    k_original = pc.single_array_ref(property_kind = 'rock permeability', indexable = 'cells')
    assert k_original is not None
    k_new = []
    for i in range(2):
        k_new.append(k_original + 1.0 + 5.0 * float(i) + np.random.random(k_original.shape))

    # add the new arrays to the property collection but writing to the new, distinct hdf5 files
    for i, (k_a, hdf5_file, ext_uuid) in enumerate(zip(k_new, jitter_ext_files, jitter_ext_uuids)):
        pc.add_cached_array_to_imported_list(k_a,
                                             source_info = 'testing multi hdf5',
                                             keyword = 'Jittery Permeability',
                                             discrete = False,
                                             uom = 'mD',
                                             property_kind = 'permeability rock',
                                             facet_type = 'direction',
                                             facet = 'IJK',
                                             realization = i,
                                             indexable_element = 'cells')
        pc.write_hdf5_for_imported_list(file_name = jitter_ext_files[i])
        pc.create_xml_for_imported_list_and_add_parts_to_model(ext_uuid = ext_uuid)

    # store the updated epc
    model.store_epc()

    # check that the new hdf5 files exist and are the expected size
    for filename in jitter_ext_files:
        file_path = os.path.join(model.epc_directory, filename)
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) >= 8 * grid.cell_count()

    # re-open the model and check that multiple ext parts are present
    model = rq.Model(epc)
    assert len(model.external_parts_list()) == model.parts_count_by_type('EpcExternalPartReference')[0][1] == 3

    # check that we can access the stored arrays
    model.h5_set_default_override('none')
    grid = model.grid()
    assert grid is not None
    pc = grid.property_collection
    assert pc is not None
    jitter_pc = rqp.selective_version_of_collection(pc, citation_title = 'Jittery Permeability')
    assert jitter_pc.number_of_parts() == 2
    k_0 = jitter_pc.single_array_ref(realization = 0)
    k_1 = jitter_pc.single_array_ref(realization = 1)
    assert k_0 is not None and k_1 is not None
    assert np.mean(k_1) >= 4.0 + np.mean(k_0)


def test_root_for_time_series(example_model_with_prop_ts_rels):
    model = example_model_with_prop_ts_rels
    ts_root = model.root_for_time_series()
    assert ts_root is not None
    assert rqet.node_type(ts_root, strip_obj = True) == 'TimeSeries'


def test_grid_list(example_model_and_crs):
    model, crs = example_model_and_crs
    # create some grid objects
    grid_a = grr.RegularGrid(model, extent_kji = (2, 2, 2), crs_uuid = crs.uuid, title = 'GRID A')
    grid_a.write_hdf5()
    grid_a.create_xml(write_active = False, add_cell_length_properties = False)
    grid_b = grr.RegularGrid(model, extent_kji = (3, 3, 3), crs_uuid = crs.uuid, title = 'GRID B')
    grid_b.write_hdf5()
    grid_b.create_xml(write_active = False, add_cell_length_properties = False)
    grid_c = grr.RegularGrid(model, extent_kji = (4, 4, 4), crs_uuid = crs.uuid, title = 'GRID C')
    grid_c.write_hdf5()
    grid_c.create_xml(write_active = False, add_cell_length_properties = False)
    # access a grid part
    grid_1 = model.grid(title = 'GRID C')
    assert grid_1 is not None
    assert grid_1 == grid_c, 'grid comparison based on uuids failed'
    assert len(model.parts(obj_type = 'IjkGridRepresentation')) == 3
    # check that the call to grid() has added the returned grid to the cache list
    assert len(model.grid_list_uuid_list()) == len(model.grid_list) == 1
    assert model.grid_list[0] == grid_c, 'grid comparison based on uuids failed'
    assert model.grid_list[0] is grid_1
    # access another grid by uuid and check that it is added to cached list
    grid_2 = model.grid(uuid = grid_a.uuid)
    assert grid_2 == grid_a
    assert len(model.grid_list_uuid_list()) == len(model.grid_list) == 2
    # add all 3 grids to the grid cache list, checking for duplicates
    model.add_grid(grid_a, check_for_duplicates = True)
    model.add_grid(grid_b, check_for_duplicates = True)
    model.add_grid(grid_c, check_for_duplicates = True)
    assert len(model.parts(obj_type = 'IjkGridRepresentation')) == 3
    assert len(model.grid_list_uuid_list()) == 3
    # check use of cached grids
    grid_3a = model.grid_for_uuid_from_grid_list(grid_b.uuid)
    assert grid_3a == grid_b, 'grid comparison based on uuids failed'
    assert grid_3a is grid_b
    grid_3b = model.grid_for_uuid_from_grid_list(grid_b.uuid)
    assert grid_3a is grid_3b
    assert tuple(grid_3a.extent_kji) == (3, 3, 3)
