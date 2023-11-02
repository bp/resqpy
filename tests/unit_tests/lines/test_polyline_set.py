import os
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.model as rq
import resqpy.lines as rql
import resqpy.olio.xml_et as rqet


def test_polyline_set_mixed_closure(example_model_and_crs):

    model, crs = example_model_and_crs

    # some coordinates for individual polylines
    plc = np.array(
        [[(100.0, 100.0, 0.0), (110.0, 100.0, 0.0),
          (105.0, 110.0, 0.0)], [(200.0, 100.0, 0.0), (210.0, 100.0, 0.0),
                                 (205.0, 110.0, 0.0)], [(300.0, 100.0, 0.0), (310.0, 100.0, 0.0), (305.0, 110.0, 0.0)],
         [(400.0, 100.0, 0.0), (410.0, 100.0, 0.0),
          (405.0, 110.0, 0.0)], [(500.0, 100.0, 0.0), (510.0, 100.0, 0.0), (505.0, 110.0, 0.0)]],
        dtype = float)
    assert len(plc) % 2 == 1  # following code needs an odd number of polylines

    for more_open in [True, False]:

        # create individual polylines with mixed closure
        pl_list = []
        for i in range(len(plc)):
            closed = bool(i % 2)
            if not more_open:
                closed = not closed
            pl = rql.Polyline(model, set_coord = plc[i], set_crs = crs.uuid, is_closed = closed, title = f'pl{i}')
            pl.write_hdf5()
            pl.create_xml()
            pl_list.append(pl)

        # create polyline set from individual polylines
        pl_set = rql.PolylineSet(model, polylines = pl_list, title = f'pl set more open {more_open}')
        pl_set.write_hdf5()
        pl_set.create_xml()

        # check some stuff
        assert pl_set.boolnotconstant
        assert pl_set.boolvalue != more_open
        assert pl_set.indices is not None and len(pl_set.indices) == 2
        assert tuple(pl_set.indices) == (1, 3)

        # also check some xml
        root = pl_set.root
        cpl_node = rqet.find_nested_tags(root, ['LinePatch', 'ClosedPolylines'])
        assert cpl_node is not None
        assert rqet.node_type(cpl_node) == 'BooleanArrayFromIndexArray'
        v = rqet.find_tag_bool(cpl_node, 'IndexIsTrue')  # should be True if majority is open
        assert v is not None and v == more_open

    # check reloading the polyline sets from epc & hdf5
    model.store_epc()
    model = rq.Model(model.epc_file)
    pls_uuids = model.uuids(obj_type = 'PolylineSetRepresentation')
    assert len(pls_uuids) == 2
    for pls_uuid in pls_uuids:
        pl_set_reloaded = rql.PolylineSet(model, uuid = pls_uuid)
        assert pl_set_reloaded is not None
        assert pl_set_reloaded.boolnotconstant
        assert pl_set_reloaded.boolvalue != ('true' in pl_set_reloaded.title.lower())


def test_polyline_set_from_irap_single_open_line(example_model_and_crs):
    model, crs = example_model_and_crs
    irap_file = os.path.join(os.path.dirname(model.epc_file), 'irap_line.txt')
    lines = [
        '123.45  678.90  1010.0', '145.67  689.01  1020.0', '167.89  699.11  1015.0', '181.81  711.22  1025.0',
        '999.00  999.00  999.00'
    ]
    with open(irap_file, 'w') as fp:
        for line in lines:
            fp.write(line + '\n')
    ps = rql.PolylineSet(model, irap_file = irap_file, crs_uuid = crs.uuid, title = 'one open')
    ps.write_hdf5()
    ps.create_xml()
    ps = rql.PolylineSet(model, uuid = ps.uuid)
    assert ps is not None
    assert ps.polys is not None and len(ps.polys) == 1
    assert not ps.polys[0].isclosed
    assert ps.polys[0].coordinates.shape == (4, 3)
    expected = np.array([(123.45, 678.90, 1010.0), (145.67, 689.01, 1020.0), (167.89, 699.11, 1015.0),
                         (181.81, 711.22, 1025.0)],
                        dtype = float)
    assert_array_almost_equal(ps.polys[0].coordinates, expected)


def test_polyline_set_from_irap_single_closed_line(example_model_and_crs):
    model, crs = example_model_and_crs
    irap_file = os.path.join(os.path.dirname(model.epc_file), 'irap_line.txt')
    lines = [
        '123.45  678.90  1010.0', '145.67  689.01  1020.0', '167.89  699.11  1015.0', '181.81  711.22  1025.0',
        '123.45  678.90  1010.0', '999.00  999.00  999.00'
    ]
    with open(irap_file, 'w') as fp:
        for line in lines:
            fp.write(line + '\n')
    ps = rql.PolylineSet(model, irap_file = irap_file, crs_uuid = crs.uuid, title = 'one open')
    ps.write_hdf5()
    ps.create_xml()
    ps = rql.PolylineSet(model, uuid = ps.uuid)
    assert ps is not None
    assert ps.polys is not None and len(ps.polys) == 1
    assert ps.polys[0].isclosed
    assert ps.polys[0].coordinates.shape == (4, 3)
    expected = np.array([(123.45, 678.90, 1010.0), (145.67, 689.01, 1020.0), (167.89, 699.11, 1015.0),
                         (181.81, 711.22, 1025.0)],
                        dtype = float)
    assert_array_almost_equal(ps.polys[0].coordinates, expected)


def test_polyline_set_from_irap_two_closed_lines(example_model_and_crs):
    model, crs = example_model_and_crs
    irap_file = os.path.join(os.path.dirname(model.epc_file), 'irap_line.txt')
    lines = [
        '123.45  678.90  1010.0', '145.67  689.01  1020.0', '167.89  699.11  1015.0', '181.81  711.22  1025.0',
        '123.45  678.90  1010.0', '999.00  999.00  999.00', '321.32  543.21  900.0', '654.32  589.89  910.5',
        '444.44  201.01  932.0', '321.32  543.21  900.0', '999.00  999.00  999.00'
    ]
    with open(irap_file, 'w') as fp:
        for line in lines:
            fp.write(line + '\n')
    ps = rql.PolylineSet(model, irap_file = irap_file, crs_uuid = crs.uuid, title = 'one open')
    ps.write_hdf5()
    ps.create_xml()
    ps = rql.PolylineSet(model, uuid = ps.uuid)
    assert ps is not None
    assert ps.polys is not None and len(ps.polys) == 2
    assert ps.polys[0].isclosed
    assert ps.polys[0].coordinates.shape == (4, 3)
    expected = np.array([(123.45, 678.90, 1010.0), (145.67, 689.01, 1020.0), (167.89, 699.11, 1015.0),
                         (181.81, 711.22, 1025.0)],
                        dtype = float)
    assert_array_almost_equal(ps.polys[0].coordinates, expected)
    assert ps.polys[1].isclosed
    assert ps.polys[1].coordinates.shape == (3, 3)
    expected = np.array([(321.32, 543.21, 900.0), (654.32, 589.89, 910.5), (444.44, 201.01, 932.0)], dtype = float)
    assert_array_almost_equal(ps.polys[1].coordinates, expected)


def test_polyline_set_from_irap_two_lines_one_open_one_closed(example_model_and_crs):
    model, crs = example_model_and_crs
    irap_file = os.path.join(os.path.dirname(model.epc_file), 'irap_line.txt')
    lines = [
        '123.45  678.90  1010.0', '145.67  689.01  1020.0', '167.89  699.11  1015.0', '181.81  711.22  1025.0',
        '999.00  999.00  999.00', '321.32  543.21  900.0', '654.32  589.89  910.5', '444.44  201.01  932.0',
        '321.32  543.21  900.0', '999.00  999.00  999.00'
    ]
    with open(irap_file, 'w') as fp:
        for line in lines:
            fp.write(line + '\n')
    ps = rql.PolylineSet(model, irap_file = irap_file, crs_uuid = crs.uuid, title = 'one open')
    ps.write_hdf5()
    ps.create_xml()
    ps = rql.PolylineSet(model, uuid = ps.uuid)
    assert ps is not None
    assert ps.polys is not None and len(ps.polys) == 2
    assert not ps.polys[0].isclosed
    assert ps.polys[0].coordinates.shape == (4, 3)
    expected = np.array([(123.45, 678.90, 1010.0), (145.67, 689.01, 1020.0), (167.89, 699.11, 1015.0),
                         (181.81, 711.22, 1025.0)],
                        dtype = float)
    assert_array_almost_equal(ps.polys[0].coordinates, expected)
    assert ps.polys[1].isclosed
    assert ps.polys[1].coordinates.shape == (3, 3)
    expected = np.array([(321.32, 543.21, 900.0), (654.32, 589.89, 910.5), (444.44, 201.01, 932.0)], dtype = float)
    assert_array_almost_equal(ps.polys[1].coordinates, expected)
