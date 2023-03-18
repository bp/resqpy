import os
import math as maths
import numpy as np
from numpy.testing import assert_array_almost_equal

import resqpy.lines
import resqpy.model as rq
import resqpy.organize
import resqpy.olio.vector_utilities as vec


def test_lines(example_model_and_crs):

    # Set up a Polyline
    title = 'Nazca'
    model, crs = example_model_and_crs
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = True,
                                 set_coord = np.array([[0, 0, 0], [1, 1, 1]]))
    line.write_hdf5()
    line.create_xml()

    # Add a interpretation
    assert line.rep_int_root is None
    line.create_interpretation_and_feature(kind = 'fault')
    assert line.rep_int_root is not None

    # Check fault can be loaded in again
    model.store_epc()
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.lines.Polyline(parent_model = model, uuid = line.uuid)
    assert reload.citation_title == title

    fault_interp = resqpy.organize.FaultInterpretation(model, uuid = line.rep_int_uuid)
    fault_feature = resqpy.organize.TectonicBoundaryFeature(model, uuid = fault_interp.feature_uuid)

    # Check title matches expected title
    assert fault_feature.feature_name == title


def test_lineset(example_model_and_crs, tmp_path):

    # Set up a PolylineSet
    title = 'Nazcas'
    model, crs = example_model_and_crs
    line1 = resqpy.lines.Polyline(parent_model = model,
                                  title = title,
                                  set_crs = crs.uuid,
                                  is_closed = True,
                                  set_coord = np.array([[0, 0, 0], [1, 1, 1]], dtype = float))

    line2 = resqpy.lines.Polyline(parent_model = model,
                                  title = title,
                                  set_crs = crs.uuid,
                                  is_closed = True,
                                  set_coord = np.array([[2, 2, 2], [3, 3, 3]], dtype = float))

    lines = resqpy.lines.PolylineSet(parent_model = model, title = title, polylines = [line1, line2])

    lines.write_hdf5()
    lines.create_xml()

    # Check lines can be loaded in again
    model.store_epc()
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.lines.PolylineSet(parent_model = model, uuid = lines.uuid)
    assert len(reload.polys) == 2,  \
        f'Expected two polylines in the polylineset, found {len(reload.polys)}'
    assert (reload.count_perpol == [2, 2]).all(),  \
        f'Expected count per polyline to be [2,2], found {reload.count_perpol}'
    pl_list = reload.convert_to_polylines()
    assert len(pl_list) == 2
    pl_set_2 = resqpy.lines.PolylineSet(model)
    pl_set_2.combine_polylines(pl_list)
    assert len(pl_set_2.polys) == 2,  \
        f'Expected two polylines in the polylineset, found {len(pl_set_2.polys)}'
    assert (pl_set_2.count_perpol == [2, 2]).all(),  \
        f'Expected count per polyline to be [2,2], found {pl_set_2.count_perpol}'
    irap_file = os.path.join(tmp_path, 'test_irap.dat')
    pl_set_2.convert_to_irap(irap_file)
    charisma_file = os.path.join(tmp_path, 'test_charisma.dat')
    pl_set_2.convert_to_charisma(charisma_file)
    pl_set_3 = resqpy.lines.PolylineSet(model, charisma_file = charisma_file)
    assert pl_set_3 is not None and len(pl_set_3.polys) == 2
    pl_set_4 = resqpy.lines.PolylineSet(model, irap_file = irap_file)
    assert pl_set_4 is not None and len(pl_set_4.polys) == 2


def test_charisma(example_model_and_crs, test_data_path):
    # Set up a PolylineSet
    model, crs = example_model_and_crs
    charisma_file = test_data_path / "Charisma_example.txt"
    lines = resqpy.lines.PolylineSet(parent_model = model, charisma_file = str(charisma_file))
    lines.write_hdf5()
    lines.create_xml()

    model.store_epc()
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.lines.PolylineSet(parent_model = model, uuid = lines.uuid)

    assert reload.title == 'Charisma_example'
    assert (reload.count_perpol == [4, 5, 4, 5, 5]).all(),  \
        f"Expected count per polyline to be [4,5,4,5,5], found {reload.count_perpol}"
    assert len(reload.coordinates) == 23,  \
        f"Expected length of coordinates to be 23, found {len(reload.coordinates)}"


def test_irap(example_model_and_crs, test_data_path):
    # Set up a PolylineSet
    model, crs = example_model_and_crs
    irap_file = test_data_path / "IRAP_example.txt"
    lines = resqpy.lines.PolylineSet(parent_model = model, irap_file = str(irap_file))
    lines.write_hdf5()
    lines.create_xml()

    model.store_epc()
    model = rq.Model(epc_file = model.epc_file)
    reload = resqpy.lines.PolylineSet(parent_model = model, uuid = lines.uuid)

    assert reload.title == 'IRAP_example'
    assert (reload.count_perpol == [15]).all(),  \
        f"Expected count per polyline to be [15], found {reload.count_perpol}"
    assert len(reload.coordinates) == 15,  \
        f"Expected length of coordinates to be 15, found {len(reload.coordinates)}"


def test_is_clockwise(example_model_and_crs):
    # Set up a Polyline
    title = 'diamond'
    model, crs = example_model_and_crs
    line = resqpy.lines.Polyline(
        parent_model = model,
        title = title,
        set_crs = crs.uuid,
        is_closed = True,
        set_coord = np.array([
            (3.0, 2.0, 4.3),  # z values are ignored for clockwise test
            (2.0, 1.0, -13.5),
            (1.0, 2.5, 7.8),
            (2.1, 3.9, -2.0)
        ]))
    assert line is not None
    for trust in [False, True]:
        assert line.is_clockwise(trust_metadata = trust)
    # reverse the order of the coordinates
    line.coordinates = np.flip(line.coordinates, axis = 0)
    assert line.is_clockwise(trust_metadata = True)
    assert not line.is_clockwise(trust_metadata = False)  # should reset metadata
    assert not line.is_clockwise(trust_metadata = True)


def test_is_convex(example_model_and_crs):
    # Set up a Polyline
    title = 'pentagon'
    model, crs = example_model_and_crs
    line = resqpy.lines.Polyline(
        parent_model = model,
        title = title,
        set_crs = crs.uuid,
        is_closed = True,
        set_coord = np.array([
            (3.0, 2.0, 4.3),  # z values are ignored for convexity
            (2.0, 1.0, -13.5),
            (1.0, 2.5, 7.8),
            (1.5, 3.0, -2.0),
            (2.5, 3.0, 23.4)
        ]))
    assert line is not None
    for trust in [False, True]:
        assert line.is_convex(trust_metadata = trust)
    # adjust one point to make shape concave
    line.coordinates[4, 1] = 1.9
    assert line.is_convex(trust_metadata = True)
    assert not line.is_convex(trust_metadata = False)  # should reset metadata
    assert not line.is_convex(trust_metadata = True)


def test_from_scaled_polyline(example_model_and_crs):
    # Set up an original Polyline
    title = 'rectangle'
    model, crs = example_model_and_crs
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = True,
                                 set_coord = np.array([(0.0, 0.0, 10.0), (0.0, 3.0, 10.0), (12.0, 3.0, 10.0),
                                                       (12.0, 0.0, 10.0)]))
    assert line is not None

    # test scaling up to a larger polyline
    bigger = resqpy.lines.Polyline.from_scaled_polyline(line, 1.5, originator = 'testing')
    expected = np.array([(-3.0, -0.75, 10.0), (-3.0, 3.75, 10.0), (15.0, 3.75, 10.0), (15.0, -0.75, 10.0)])
    assert bigger.isclosed
    assert bigger.is_convex()
    assert bigger.title == 'rectangle'
    assert bigger.originator == 'testing'
    assert_array_almost_equal(bigger.coordinates, expected)

    # test scaling to a smaller polyline
    smaller = resqpy.lines.Polyline.from_scaled_polyline(line, 2.0 / 3.0, title = 'small')
    expected = np.array([(2.0, 0.5, 10.0), (2.0, 2.5, 10.0), (10.0, 2.5, 10.0), (10.0, 0.5, 10.0)])
    assert smaller.title == 'small'
    assert_array_almost_equal(smaller.coordinates, expected)


def test_point_is_inside_and_balanced_centre_and_segment_normal(example_model_and_crs):
    # Set up an original Polyline
    title = 'sigma'
    model, crs = example_model_and_crs
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = True,
                                 set_coord = np.array([(0.0, 0.0, 10.0), (0.0, 3.0, 10.0), (4.0, 3.0, 10.0),
                                                       (2.0, 1.5, 10.0), (4.0, 0.0, 10.0)]))
    assert line is not None
    assert not line.is_convex()
    assert line.is_clockwise(trust_metadata = False)
    centre = line.balanced_centre(in_xy = True)
    for mode in ['crossing', 'winding']:
        assert line.point_is_inside_xy((1.0, 1.0, 1.0), mode = mode)
        assert not line.point_is_inside_xy((2.5, 1.4, 1.0), mode = mode)
        assert line.point_is_inside_xy(centre, mode = mode)
    # check a couple of normal vectors
    assert_array_almost_equal(line.segment_normal(1), (0.0, 1.0, 0.0))
    assert_array_almost_equal(line.segment_normal(4), (0.0, -1.0, 0.0))


def test_segment_methods(example_model_and_crs):
    # Set up an original Polyline
    model, crs = example_model_and_crs
    line = __zig_zag(model, crs)
    coords = line.coordinates
    assert maths.isclose(line.segment_length(0), maths.sqrt(27.0))
    assert maths.isclose(line.segment_length(0, in_xy = True), maths.sqrt(18.0))
    assert_array_almost_equal(line.segment_midpoint(1), (3.0, 8.0, 13.5))
    for in_xy in [False, True]:
        d = 2 if in_xy else 3
        length = 0.0
        for seg in range(len(coords) - 1):
            length += maths.sqrt(
                np.sum((coords[seg + 1, :d] - coords[seg, :d]) * (coords[seg + 1, :d] - coords[seg, :d])))
        assert maths.isclose(line.full_length(in_xy = in_xy), length)
    # check interpolation method at end points
    assert_array_almost_equal(line.interpolated_point(0.0), coords[0])
    assert_array_almost_equal(line.interpolated_point(1.0), coords[-1])
    # create a simple 3 segment polyline
    coords = np.array([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 5.0, 3.0), (10.0, 10.0, 3.0)])
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = 'angled',
                                 set_crs = crs.uuid,
                                 is_closed = False,
                                 set_coord = coords)
    # check midpoint
    assert_array_almost_equal(line.interpolated_point(0.5), (7.5, 2.5, 1.5))
    # create some equidistant points along the line
    ep = line.equidistant_points(7)
    # check some of the equidistant points
    assert_array_almost_equal(ep[0], coords[0])
    assert maths.isclose(ep[1, 1], 0.0)
    assert_array_almost_equal(ep[3], (7.5, 2.5, 1.5))
    assert maths.isclose(ep[5, 0], 10.0)
    assert_array_almost_equal(ep[-1], coords[-1])
    # check line segment intersection method
    for half_seg in [False, True]:
        seg, x, y = line.first_line_intersection(5.0, 5.0, 10.0, 0.0, half_segment = half_seg)
        assert seg == 1
        assert maths.isclose(x, 7.5) and maths.isclose(y, 2.5)
        xyz = line.segment_xyz_from_xy(seg, x, y)
        assert_array_almost_equal(xyz, (7.5, 2.5, 1.5))
        seg, x, y = line.first_line_intersection(5.0, 5.0, 0.0, 10.0, half_segment = half_seg)
        assert seg is None and x is None and y is None
        seg, x, y = line.first_line_intersection(0.0, 10.0, 5.0, 5.0, half_segment = half_seg)
        if half_seg:
            assert seg == 1
            assert maths.isclose(x, 7.5) and maths.isclose(y, 2.5)
        else:
            assert seg is None and x is None and y is None


def test_area(example_model_and_crs):
    # create an octagonal polyline
    model, crs = example_model_and_crs
    line = __octagon(model, crs)
    assert maths.isclose(line.area(), 14.0)


def test_concave_area(example_model_and_crs):
    # create an octagonal polyline with some concavities
    model, crs = example_model_and_crs
    line = __concave_octagon(model, crs)
    assert maths.isclose(line.area(), 12.0, rel_tol = 0.01)


def test_splined_and_tangent_vectors(example_model_and_crs):
    model, crs = example_model_and_crs
    line = __zig_zag(model, crs)
    s_line = line.splined(title = 'splined zig zag')
    assert s_line is not None
    tans = s_line.tangent_vectors()
    assert tans is not None
    assert len(tans) == len(s_line.coordinates)


def test_normalised_and_denormalised(example_model_and_crs):
    model, crs = example_model_and_crs
    line = __octagon(model, crs)
    for mode in ['square', 'perimeter', 'circle']:
        xn, yn = line.normalised_xy(3.0, 3.0, mode = mode)
        if mode == 'square':
            assert maths.isclose(xn, yn)
            assert 0.0 < xn < 0.5
        elif mode == 'perimeter':
            assert maths.isclose(yn, 0.0) or maths.isclose(yn, 1.0)
        x, y = line.denormalised_xy(xn, yn, mode = mode)
        assert maths.isclose(x, y)
        assert maths.isclose(x, 3.0)
        if mode != 'square':
            xn, yn = line.normalised_xy(4.25, 5.13, mode = mode)
            x, y = line.denormalised_xy(xn, yn, mode = mode)
            assert maths.isclose(x, 4.25) and maths.isclose(y, 5.13)


def test_poly_index_containing_point_in_xy(example_model_and_crs):
    model, crs = example_model_and_crs
    octagon = __octagon(model, crs)
    square = __square(model, crs)
    pl_set = resqpy.lines.PolylineSet(model, polylines = [octagon, square])
    for mode in ['crossing', 'winding']:
        assert pl_set.poly_index_containing_point_in_xy((5.8, 5.8), mode = mode) is None
        assert pl_set.poly_index_containing_point_in_xy((5.1, 5.1), mode = mode) == 0
        assert pl_set.poly_index_containing_point_in_xy((6.2, 4.8, 0.0), mode = mode) == 1


def test_closest_segment_and_distance_to_point_xy(example_model_and_crs):
    model, crs = example_model_and_crs
    c = np.array((123.45, 678.90, 0.0), dtype = float)
    nonagon = resqpy.lines.Polyline.for_regular_polygon(model, 9, 5.7, c, crs.uuid, 'nonagon')
    assert nonagon is not None
    assert len(nonagon.coordinates) == 9
    cp = np.mean(nonagon.coordinates, axis = 0)
    assert_array_almost_equal(cp, c)
    for seg in range(9):
        p = np.array(nonagon.segment_midpoint(seg), dtype = float)
        m_seg, m_d = nonagon.closest_segment_and_distance_to_point_xy(p)
        assert m_seg is not None and m_d is not None
        assert m_seg == seg
        assert maths.isclose(m_d, 0.0, abs_tol = 1.0e-6)
        p_in = (p - c) * 0.2 + c
        p_out = (p - c) * 23.4 + c
        for pp in [p_in, p_out]:
            d = vec.naive_length(p - pp)
            m_seg, m_d = nonagon.closest_segment_and_distance_to_point_xy(pp)
            assert m_seg is not None and m_d is not None
            assert m_seg == seg
            assert maths.isclose(m_d, d, rel_tol = 1.0e-6)
        pv = nonagon.coordinates[seg]
        p_out = (pv - c) * 13.9 + c
        d = vec.naive_length(p_out - pv)
        m_seg, m_d = nonagon.closest_segment_and_distance_to_point_xy(p_out)
        assert m_seg is not None and m_d is not None
        assert m_seg in [seg, (seg - 1) % 9]
        assert maths.isclose(m_d, d, rel_tol = 1.0e-6)


def test_hull(example_model_and_crs):
    model, crs = example_model_and_crs
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 0.0], [7.0, 9.0, 0.0],
                       [10.0, 10.0, 0.0], [11.0, 5.0, 0.0], [10.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
                      dtype = float)
    plo = resqpy.lines.Polyline(model, is_closed = True, set_coord = coords, set_crs = crs.uuid, title = 'plo')
    plo.write_hdf5()
    plo.create_xml()
    hull = resqpy.lines.Polyline.convex_hull_from_closed_polyline(plo, 'hull')
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [11.0, 5.0, 0.0], [10.0, 0.0, 0.0]],
                        dtype = float)
    assert_array_almost_equal(hull.coordinates, expected)


def __zig_zag(model, crs):
    title = 'zig_zag'
    coords = np.array([(4.0, 5.0, 10.0), (1.0, 8.0, 13.0), (5.0, 8.0, 14.0), (1.0, 12.0, 10.0), (5.0, 12.0, 10.0)])
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = False,
                                 set_coord = coords)
    return line


def __octagon(model, crs):
    title = 'octagon'
    coords = np.array([(2.5, 2.5, 0.0), (2.0, 3.0, 0.0), (2.0, 5.0, 0.0), (3.0, 6.0, 0.0), (5.0, 6.0, 0.0),
                       (6.0, 5.0, 0.0), (6.0, 3.0, 0.0), (5.0, 2.0, 0.0), (3.0, 2.0, 0.0)])
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = True,
                                 set_coord = coords)
    return line


def __concave_octagon(model, crs):
    title = 'concave octagon'
    coords = np.array([(2.5, 2.5, 0.0), (2.0, 3.0, 0.0), (2.0, 5.0, 0.0), (3.0, 6.0, 0.0), (5.0, 6.0, 0.0),
                       (6.0, 5.0, 0.0), (5.0, 4.0, 0.0), (6.0, 3.0, 0.0), (5.0, 2.0, 0.0), (4.0, 3.0, 0.0),
                       (3.0, 2.0, 0.0)])
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = True,
                                 set_coord = coords)
    return line


def __square(model, crs):
    title = 'square'
    coords = np.array([(6.0, 3.0, 10.0), (6.0, 5.0, 10.0), (8.0, 5.0, 10.0), (8.0, 3.0, 10.0)])
    line = resqpy.lines.Polyline(parent_model = model,
                                 title = title,
                                 set_crs = crs.uuid,
                                 is_closed = True,
                                 set_coord = coords)
    return line
