import numpy as np
import pytest
import resqpy.olio.utility as u


@pytest.mark.parametrize("test_input,expected_output",
                         [(np.array([2, 5, 3]), np.array([2, 4, 1])), (np.array([-3, 0, -6]), np.array([-7, -1, -4])),
                          (np.array([-193725, 481873, -330798]), np.array([-330799, 481872, -193726]))])
def test_kji0_from_ijk1(test_input, expected_output):
    # Act
    output = u.kji0_from_ijk1(test_input)

    # Assert
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize("test_input,expected_output",
                         [(np.array([2, 4, 1]), np.array([2, 5, 3])), (np.array([-7, -1, -4]), np.array([-3, 0, -6])),
                          (np.array([-330799, 481872, -193726]), np.array([-193725, 481873, -330798]))])
def test_ijk1_from_kji0(test_input, expected_output):
    # Act
    output = u.ijk1_from_kji0(test_input)

    # Assert
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize("test_input,expected_output",
                         [(np.array([2, 5, 3]), np.array([3, 5, 2])), (np.array([-3, 0, -6]), np.array([-6, 0, -3])),
                          (np.array([-193725, 481873, -330798]), np.array([-330798, 481873, -193725]))])
def test_extent_switch_ijk_kji(test_input, expected_output):
    # Act
    output = u.extent_switch_ijk_kji(test_input)

    # Assert
    np.testing.assert_array_equal(output, expected_output)


@pytest.mark.parametrize("test_input,expected_output", [(np.array([2, 5, 3]), 30), ([3, 0, 6], 0),
                                                        ((193, 481, 330), 30634890)])
def test_cell_count_from_extent(test_input, expected_output):
    # Act
    output = u.cell_count_from_extent(test_input)

    # Assert
    assert output == expected_output


@pytest.mark.parametrize("test_input,expected_output",
                         [(np.array([2, 5, 3]), '[4, 6, 3]'), (np.array([-3, 0, -6]), '[-5, 1, -2]'),
                          (np.array([-193725, 481873, -330798]), '[-330797, 481874, -193724]')])
def test_string_ijk1_for_cell_kji0(test_input, expected_output):
    # Act
    output = u.string_ijk1_for_cell_kji0(test_input)

    # Assert
    assert output == expected_output


@pytest.mark.parametrize("test_input,expected_output",
                         [(np.array([2, 5, 3]), '[2, 5, 3]'), (np.array([-3, 0, -6]), '[-3, 0, -6]'),
                          (np.array([-193725, 481873, -330798]), '[-193725, 481873, -330798]')])
def test_string_ijk1_for_cell_ijk1(test_input, expected_output):
    # Act
    output = u.string_ijk1_for_cell_ijk1(test_input)

    # Assert
    assert output == expected_output


@pytest.mark.parametrize("test_input,expected_output",
                         [(np.array([2, 5, 3]), '[3, 5, 2]'), (np.array([-3, 0, -6]), '[-6, 0, -3]'),
                          (np.array([-193725, 481873, -330798]), '[-330798, 481873, -193725]')])
def test_string_ijk_for_extent_kji(test_input, expected_output):
    # Act
    output = u.string_ijk_for_extent_kji(test_input)

    # Assert
    assert output == expected_output


@pytest.mark.parametrize("test_input,expected_output", [(np.array([26.1, 5.0, 32.8]), '(26.10, 5.00, 32.800)'),
                                                        ([327.281, 82.19, 62172.329], '(327.28, 82.19, 62172.329)'),
                                                        ((19323.1, 481.33, 330.201), '(19323.10, 481.33, 330.201)')])
def test_string_xyz(test_input, expected_output):
    # Act
    output = u.string_xyz(test_input)

    # Assert
    assert output == expected_output


@pytest.mark.parametrize("test_input1,test_input2,expected_output", [(3, '+', 3.5), (-1, '-', -1.5), (7, '-', 6.5)])
def test_horizon_float(test_input1, test_input2, expected_output):
    # Act
    output = u.horizon_float(test_input1, test_input2)

    # Assert
    assert output == expected_output
