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
import resqpy.time_series as rqts
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
    # check that hdf5 ext parts have distinct uuids
    ext_uuids = [model.h5_uuid_list(model.root(uuid = model.uuid_for_part(p))) for p in jitter_pc.parts()]
    assert len(ext_uuids) == 2 and np.all(ul is not None and len(ul) == 1 for ul in ext_uuids)
    assert ext_uuids[0][0] is not None and ext_uuids[1][0] is not None
    assert not bu.matching_uuids(ext_uuids[0][0], ext_uuids[1][0])


def test_root_for_time_series(example_model_with_prop_ts_rels):
    # test when model has only one time series
    model = example_model_with_prop_ts_rels
    ts_root = model.root_for_time_series()
    assert ts_root is not None
    assert rqet.node_type(ts_root, strip_obj = True) == 'TimeSeries'
    assert model.resolve_time_series_root(ts_root) is ts_root
    assert model.resolve_time_series_root(None) is ts_root
    # test when model has multiple time series
    model = rq.Model(model.epc_file)
    assert model is not None
    oldest_ts_uuid = model.uuid(obj_type = 'TimeSeries')
    assert oldest_ts_uuid is not None
    for first_timestamp in ['2022-01-01Z', '2023-01-01Z', '2024-01-01Z']:
        ts = rqts.TimeSeries(parent_model = model, first_timestamp = '2000-01-01Z')
        for _ in range(3):
            ts.extend_by_days(90)
        ts.create_xml()
        newest_ts_uuid = ts.uuid
    # check that oldest series by creation date is returned by default
    ts_root = model.root_for_time_series()
    assert ts_root is model.root(uuid = oldest_ts_uuid)
    # check that correct time series root is returned when uuid given
    ts_root = model.root_for_time_series(uuid = newest_ts_uuid)
    assert ts_root is not None and ts_root is model.root(uuid = newest_ts_uuid)


def test_grid_list(example_model_and_crs):
    model, crs = example_model_and_crs
    # create some grid objects
    grid_a, grid_b, grid_c = add_grids(model, crs, False)
    # access a grid part
    grid_1 = model.grid(title = 'GRID C')
    assert grid_1 is not None
    assert bu.matching_uuids(grid_1.uuid, grid_c.uuid)
    assert len(model.parts(obj_type = 'IjkGridRepresentation')) == 3
    # check that the call to grid() has added the returned grid to the cache list
    assert len(model.grid_list_uuid_list()) == len(model.grid_list) == 1
    assert bu.matching_uuids(model.grid_list[0].uuid, grid_c.uuid)
    assert model.grid_list[0] is grid_1
    # access another grid by uuid and check that it is added to cached list
    grid_2 = model.grid(uuid = grid_a.uuid)
    assert bu.matching_uuids(grid_2.uuid, grid_a.uuid)
    assert len(model.grid_list_uuid_list()) == len(model.grid_list) == 2
    # add all 3 grids to the grid cache list, checking for duplicates
    model.add_grid(grid_a, check_for_duplicates = True)
    model.add_grid(grid_b, check_for_duplicates = True)
    model.add_grid(grid_c, check_for_duplicates = True)
    assert len(model.parts(obj_type = 'IjkGridRepresentation')) == 3
    assert len(model.grid_list_uuid_list()) == 3
    # check use of cached grids
    grid_3a = model.grid_for_uuid_from_grid_list(grid_b.uuid)
    assert bu.matching_uuids(grid_3a.uuid, grid_b.uuid)
    assert grid_3a is grid_b
    grid_3b = model.grid_for_uuid_from_grid_list(grid_b.uuid)
    assert grid_3a is grid_3b
    assert tuple(grid_3a.extent_kji) == (3, 3, 3)


def test_catalogue_functions(example_model_and_crs):
    model, crs = example_model_and_crs
    # create some grid objects with some boring properties
    grid_a, grid_b, grid_c = add_grids(model, crs, True)
    # test parts() method with various options
    all_parts = model.parts()
    assert isinstance(all_parts, list)
    assert len(all_parts) >= 13
    assert all([isinstance(p, str) for p in all_parts])
    # test obj_type filtering
    grid_parts = model.parts(obj_type = 'IjkGridRepresentation')
    assert len(grid_parts) == 3
    pcbt = model.parts_count_by_type('obj_IjkGridRepresentation')
    assert isinstance(pcbt, list) and len(pcbt) == 1 and pcbt[0] == ('IjkGridRepresentation', 3)
    # test single part selection with multiple handling options
    oldest_grid_part = model.part(obj_type = 'IjkGridRepresentation', multiple_handling = 'oldest')
    assert oldest_grid_part is not None and isinstance(oldest_grid_part, str)
    none_part = model.part(obj_type = 'IjkGridRepresentation', multiple_handling = 'none')
    assert none_part is None
    # test type_of_part()
    assert all([(model.type_of_part(p, strip_obj = True) == 'IjkGridRepresentation') for p in grid_parts])
    # test filtering with title mode and case sensitivity options
    grid_b_part = model.part(parts_list = grid_parts, title = 'b', title_mode = 'ends')
    assert grid_b_part is not None
    assert model.citation_title_for_part(grid_b_part).endswith('B')
    no_grid_b_part = model.part(parts_list = grid_parts, title = 'b', title_mode = 'ends', title_case_sensitive = True)
    assert no_grid_b_part is None
    grid_not_b_titles = model.parts(parts_list = grid_parts, title = 'grid b', title_mode = 'is not')
    assert len(grid_not_b_titles) == 2 and 'GRID B' not in grid_not_b_titles
    none_root = model.root(parts_list = grid_parts,
                           title = 'GRID',
                           title_mode = 'does not start',
                           title_case_sensitive = True)
    assert none_root is None
    all_grid_uuids = model.uuids(parts_list = grid_parts, title = 'QWERTY', title_mode = 'does not contain')
    assert len(all_grid_uuids) == 3
    two_parts = model.parts(parts_list = grid_parts, title = 'A', title_mode = 'does not end')
    assert len(two_parts) == 2
    # test uuids() with relationship filtering
    grid_b_rels_uuids = model.uuids(related_uuid = model.uuid_for_part(grid_b_part), sort_by = 'uuid')
    assert grid_b_rels_uuids is not None and len(grid_b_rels_uuids) >= 4
    assert uuid_in_list(crs.uuid, grid_b_rels_uuids)
    # test parts_list_related_to_uuid_of_type()
    grid_b_rels_crs_part = model.parts_list_related_to_uuid_of_type(model.uuid_for_part(grid_b_part),
                                                                    'obj_LocalDepth3dCrs')
    assert isinstance(grid_b_rels_crs_part, list) and len(grid_b_rels_crs_part) == 1
    assert grid_b_rels_crs_part[0] == crs.part
    grid_b_rels_uuids_ints = [bu.uuid_as_int(u) for u in grid_b_rels_uuids]
    assert all(a < b for a, b in zip(grid_b_rels_uuids_ints[:-1], grid_b_rels_uuids_ints[1:]))
    # test parts_list_of_type() with uuid specified
    singleton_list = model.parts_list_of_type('obj_IjkGridRepresentation', uuid = model.uuid_for_part(grid_b_part))
    assert isinstance(singleton_list, list) and len(singleton_list) == 1
    assert singleton_list[0] == grid_b_part
    empty_list = model.parts_list_of_type('obj_IjkGridRepresentation', uuid = crs.uuid)
    assert isinstance(empty_list, list) and len(empty_list) == 0
    empty_list = model.parts_list_of_type('obj_IjkGridRepresentation', uuid = bu.new_uuid())
    assert isinstance(empty_list, list) and len(empty_list) == 0
    # test sorting
    grid_b_props_titles = model.titles(obj_type = 'ContinuousProperty',
                                       parts_list = [model.part_for_uuid(uuid) for uuid in grid_b_rels_uuids],
                                       sort_by = 'title')
    assert len(grid_b_props_titles) == 3
    assert all([a < b for (a, b) in zip(grid_b_props_titles[:-1], grid_b_props_titles[1:])])
    # test filtering by extra metadata
    set_extra_metadata(grid_b, 'em_test', 'chai')
    grid_b.create_xml()
    set_extra_metadata(grid_c, 'em_test', 'oolong')
    grid_c.create_xml()
    assert model.root(extra = {'em_test': 'espresso'}) is None
    assert bu.matching_uuids(grid_c.uuid, model.uuid(extra = {'em_test': 'oolong'}))
    # test list_of_parts()
    obj_parts = model.list_of_parts()
    all_parts = model.list_of_parts(only_objects = False)
    assert len(all_parts) > len(obj_parts)
    assert all([p in all_parts for p in obj_parts])
    assert not all([p in obj_parts for p in all_parts])
    # check exception is raised when multiple parts match criteria
    with pytest.raises(ValueError) as excinfo:
        part = model.part(obj_type = 'IjkGridRepresentation')


def test_supporting_representation_change(example_model_and_crs):
    model, crs = example_model_and_crs
    # create some grid objects with some boring properties
    grid_ap, grid_bp, grid_cp = add_grids(model, crs, True)
    # create some more grid objects without those properties
    grid_anp, grid_bnp, grid_cnp = add_grids(model, crs, True)
    assert len(model.parts_list_of_type('obj_IjkGridRepresentation')) == 6
    pc = grid_bp.property_collection
    assert pc.number_of_parts() > 0
    bnp_count = grid_bnp.property_collection.number_of_parts()
    prop_part = pc.parts()[-1]
    sr_uuid = model.supporting_representation_for_part(prop_part)
    assert bu.matching_uuids(sr_uuid, grid_bp.uuid)
    assert model.change_uuid_in_supporting_representation_reference(model.root(uuid = pc.uuid_for_part(prop_part)),
                                                                    old_uuid = sr_uuid,
                                                                    new_uuid = grid_bnp.uuid)
    sr_uuid = model.supporting_representation_for_part(prop_part)
    assert not bu.matching_uuids(sr_uuid, grid_bp.uuid)
    assert bu.matching_uuids(sr_uuid, grid_bnp.uuid)
    grid_bnp.property_collection = None
    grid_bnp.extract_property_collection()
    assert grid_bnp.property_collection.number_of_parts() == bnp_count + 1


def test_without_full_load(example_model_with_prop_ts_rels):
    epc = example_model_with_prop_ts_rels.epc_file
    uuid_list = example_model_with_prop_ts_rels.uuids()
    assert len(uuid_list) > 0
    del example_model_with_prop_ts_rels
    # open model with minimum loading of xml
    model = rq.Model(epc_file = epc, full_load = False, create_basics = False, create_hdf5_ext = False)
    assert model is not None
    assert len(model.parts_forest) >= len(uuid_list)
    # check that xml for parts has not been loaded but part names and uuids are catalogued
    assert np.all([
        p_type is not None and uuid is not None and tree is None
        for (p_type, uuid, tree) in model.parts_forest.values()
    ])
    # see if parts are searchable
    cp_parts = model.parts(obj_type = 'ContinuousProperty')
    assert cp_parts is not None and len(cp_parts) > 1
    # see if xml is loaded on demand
    cp_tree = model.tree_for_part(cp_parts[0])
    assert cp_tree is not None
    crs_root = model.root(obj_type = 'LocalDepth3dCrs')
    assert crs_root is not None
    assert rqet.find_tag(crs_root, 'VerticalUom') is not None


def test_forestry(example_model_with_prop_ts_rels):
    model = example_model_with_prop_ts_rels
    full_parts_list = model.parts()
    dp_parts_list = model.parts(obj_type = 'DiscreteProperty')
    assert len(dp_parts_list) > 1
    # remove an individual part
    model.remove_part(dp_parts_list[0])
    # corrupt some forest dictionary entries and test tidy up
    for part in dp_parts_list[1:]:
        model.parts_forest[part] = (None, None, None)
    model.tidy_up_forests()
    assert len(model.parts()) + len(dp_parts_list) == len(full_parts_list)
    assert all(p not in model.parts() for p in dp_parts_list)
    # test patch_root_for_part()
    crs_uuid = model.uuid(obj_type = 'LocalDepth3dCrs')
    crs_part = model.part_for_uuid(crs_uuid)
    assert crs_uuid is not None and crs_part is not None
    crs = rqc.Crs(model, uuid = crs_uuid)
    assert crs is not None
    crs.title = 'relativity'
    crs.originator = 'einstein'
    new_crs_node = crs.create_xml(add_as_part = False, reuse = False)
    rqet.find_tag(new_crs_node, 'VerticalUom').text = 'ft[US]'
    model.patch_root_for_part(crs_part, new_crs_node)
    assert rqet.find_tag_text(model.root(uuid = crs_uuid), 'VerticalUom') == 'ft[US]'
    assert model.citation_title_for_part(crs_part) == 'relativity'
    assert model.title(uuid = crs_uuid) == 'relativity'
    assert rqet.find_nested_tags_text(model.root(uuid = crs_uuid), ['Citation', 'Originator']) == 'einstein'
    # rough test of low level fell_part()
    model.fell_part(crs_part)
    assert len(model.parts()) + len(dp_parts_list) + 1 == len(full_parts_list)


def test_copy_from(example_model_with_prop_ts_rels):
    original_epc = example_model_with_prop_ts_rels.epc_file
    copied_epc = original_epc[:-4] + '_copy.epc'
    parts_list = example_model_with_prop_ts_rels.parts(sort_by = 'oldest')
    assert len(parts_list) > 0
    del example_model_with_prop_ts_rels
    model = rq.Model(copied_epc, copy_from = original_epc)
    assert model.parts(sort_by = 'oldest') == parts_list


def test_h5_array_element(example_model_with_properties):
    model = example_model_with_properties
    zone_root = model.root(obj_type = 'DiscreteProperty', title = 'Zone')
    assert zone_root is not None
    key_pair = model.h5_uuid_and_path_for_node(rqet.find_nested_tags(zone_root, ['PatchOfValues', 'Values']))
    assert key_pair is not None and all([x is not None for x in key_pair])
    # check full array is expected size and shape
    shape, dtype = model.h5_array_shape_and_type(key_pair)
    assert shape == (3, 5, 5)
    assert str(dtype)[0] == 'i'
    # test single element access
    zone = model.h5_array_element(key_pair, index = (1, 2, 2), dtype = int)
    assert zone == 2
    # test single element access with required shape
    zone = model.h5_array_element(key_pair, index = (1, 7), dtype = int, required_shape = (3, 25))
    assert zone == 2


def add_grids(model, crs, add_lengths):
    grid_a = grr.RegularGrid(model,
                             extent_kji = (2, 2, 2),
                             crs_uuid = crs.uuid,
                             title = 'GRID A',
                             set_points_cached = True)
    grid_a.write_hdf5()
    grid_a.create_xml(write_active = False, add_cell_length_properties = add_lengths, write_geometry = True)
    grid_b = grr.RegularGrid(model,
                             extent_kji = (3, 3, 3),
                             crs_uuid = crs.uuid,
                             title = 'GRID B',
                             set_points_cached = True)
    grid_b.write_hdf5()
    grid_b.create_xml(write_active = False, add_cell_length_properties = add_lengths, write_geometry = True)
    grid_c = grr.RegularGrid(model,
                             extent_kji = (4, 4, 4),
                             crs_uuid = crs.uuid,
                             title = 'GRID C',
                             set_points_cached = True)
    grid_c.write_hdf5()
    grid_c.create_xml(write_active = False, add_cell_length_properties = add_lengths, write_geometry = True)
    return (grid_a, grid_b, grid_c)


def set_extra_metadata(obj, key, value):
    if not hasattr(obj, 'extra_metadata') or obj.extra_metadata is None:
        obj.extra_metadata = {}
    obj.extra_metadata[key] = value


def uuid_in_list(uuid, uuid_list):
    for u in uuid_list:
        if bu.matching_uuids(u, uuid):
            return True
    return False
