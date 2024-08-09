import pytest
import math as maths

import resqpy.well as rqw
import resqpy.olio.xml_et as rqet


def test_list_obj_references(example_model_with_well):

    model, well_interp, datum, traj = example_model_with_well

    i_refs = rqet.list_obj_references(well_interp.root)
    d_refs = rqet.list_obj_references(datum.root)
    t_refs = rqet.list_obj_references(traj.root)
    t_refs_with_hdf5 = rqet.list_obj_references(traj.root, skip_hdf5 = False)

    assert len(i_refs) == 1  # feature reference
    assert len(d_refs) == 1  # crs reference
    assert len(t_refs) == 3  # interp, crs & datum references
    assert len(t_refs_with_hdf5) == 5  # as above, plus mds and control points hdf5 references


def test_print_xml_tree(example_model_with_well):
    # note: this test only does a rudimentary check of the number of lines reported

    model, well_interp, datum, traj = example_model_with_well

    lines = rqet.print_xml_tree(datum.root,
                                level = 0,
                                max_level = None,
                                strip_tag_refs = True,
                                to_log = False,
                                log_level = None,
                                max_lines = 0,
                                line_count = 0)
    assert lines == 15
    lines = rqet.print_xml_tree(well_interp.root,
                                level = 0,
                                max_level = 10,
                                strip_tag_refs = True,
                                to_log = False,
                                log_level = None,
                                max_lines = 0,
                                line_count = 0)
    assert lines == 12
    lines = rqet.print_xml_tree(traj.root,
                                level = 0,
                                max_level = 4,
                                strip_tag_refs = False,
                                to_log = True,
                                log_level = 'info',
                                max_lines = 20,
                                line_count = 0)
    assert lines == 21


def test_miscellaneous_xml_et_functions(example_model_with_well):

    model, well_interp, datum, traj = example_model_with_well

    assert rqet.list_of_tag(None, 'Name') is None
    assert rqet.list_of_descendant_tag(None, 'Name') is None
    assert rqet.list_obj_references(None) == []
    assert rqet.count_tag(None, 'ExtraMetadata') is None
    assert rqet.find_metadata_item_node_in_xml(None, 'source') is None

    datetime_str = rqet.find_nested_tags_cast(well_interp.root, ['Citation', 'Creation'], dtype = str)
    assert isinstance(datetime_str, str)
    assert len(datetime_str) == 20
    assert datetime_str.startswith('20')
    assert datetime_str.endswith('Z')

    assert datum.extra_metadata == {}
    model.set_source_for_part(datum.part, 'true')
    datum = rqw.MdDatum(model, uuid = datum.uuid)
    assert len(datum.extra_metadata) == 1
    emv = rqet.find_nested_tags_bool(datum.root, ['ExtraMetadata', 'Value'])
    assert emv is not None
    assert isinstance(emv, bool)
    assert emv is True
    model.set_source_for_part(datum.part, '3.14')
    datum = rqw.MdDatum(model, uuid = datum.uuid)
    assert len(datum.extra_metadata) == 1
    emv = rqet.find_nested_tags_float(datum.root, ['ExtraMetadata', 'Value'])
    assert emv is not None
    assert isinstance(emv, float)
    assert maths.isclose(emv, 3.14)
