import resqpy.grid.faults as f
import numpy as np
from resqpy.model import Model


# yapf: disable
def test_find_faults(faulted_grid):
    # Arrange
    expected_i_fault_id = np.array(
        [[0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    expected_j_fault_id = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    # Act
    j_fault_id, i_fault_id = f.find_faults(faulted_grid)

    # Assert
    np.testing.assert_array_equal(j_fault_id, expected_j_fault_id)
    np.testing.assert_array_equal(i_fault_id, expected_i_fault_id)


def test_find_faults_create_organizing_objects_where_needed_true(faulted_grid):
    # Arrange
    expected_i_fault_id = np.array(
        [[0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 2, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])
    expected_j_fault_id = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])
    model = faulted_grid.model
    f_parts = model.parts(obj_type='FaultInterpretation')
    for part in f_parts:
        uuid = model.uuid_for_part(part)
        fault_features = model.parts(obj_type='TectonicBoundaryFeature', related_uuid=uuid)
        assert len(fault_features) == 1
        model.remove_part(part)
        for feature in fault_features:
            model.remove_part(feature)

    # Act
    # todo: need to fix issue #401
    # j_fault_id, i_fault_id = f.find_faults(faulted_grid, set_face_sets=True,
    #                                        create_organizing_objects_where_needed=True)
    counts_after = model.parts_count_by_type(type_of_interest='FaultInterpretation')

    # Assert
    # assert counts_after == 2
    # np.testing.assert_array_equal(j_fault_id, expected_j_fault_id)
    # np.testing.assert_array_equal(i_fault_id, expected_i_fault_id)


def test_fault_throws(faulted_grid):
    # Arrange
    expected_i_fault_throw = np.array([[[0., 0., 0., 1.75, 0., 0., 0.],
                                        [0., 0., 0., 5.25, 0., 0., 0.],
                                        [0., 0., 0., 1.75, 0., 0., 0.],
                                        [0., 0., 0., 3.5, 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0.]],
                                       [[0., 0., 0., 1.75, 0., 0., 0.],
                                        [0., 0., 0., 5.25, 0., 0., 0.],
                                        [0., 0., 0., 1.75, 0., 0., 0.],
                                        [0., 0., 0., 3.5, 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0.]],
                                       [[0., 0., 0., -3.25, 0., 0., 0.],
                                        [0., 0., 0., 0.25, 0., 0., 0.],
                                        [0., 0., 0., -3.25, 0., 0., 0.],
                                        [0., 0., 0., -1.5, 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0.]]])
    expected_j_fault_throw = np.array([[[0., 0., 0., 0., 0., 0., 0., 0.],
                                        [-3.5, -7., -7., -7., -7., -7., -7., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0.]],
                                       [[0., 0., 0., 0., 0., 0., 0., 0.],
                                        [-3.5, -7., -7., -7., -7., -7., -7., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0.]],
                                       [[0., 0., 0., 0., 0., 0., 0., 0.],
                                        [-3.5, -7., -7., -7., -7., -7., -7., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0., 0., 0.]]])

    # Act
    j_fault_throw, i_fault_throw = f.fault_throws(faulted_grid)

    # Assert
    np.testing.assert_array_equal(j_fault_throw, expected_j_fault_throw)
    np.testing.assert_array_equal(i_fault_throw, expected_i_fault_throw)


def test_fault_throws_per_edge_per_column(faulted_grid):
    # Arrange
    expected_fault_throws = np.array(
        [[[[0., 0.], [0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., -3.5]],
          [[0., 0.], [3.5, 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]]],
         [[[-0., 3.5], [0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., -7.]],
          [[-0., 7.], [7., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 3.5], [-0., 0.]]],
         [[[-3.5, 0.], [0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., -7.]],
          [[-7., 0.], [7., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-3.5, 0.], [-0., 0.]]],
         [[[-0., 0.], [0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., -7.]],
          [[-0., 0.], [7., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]]],
         [[[-0., 0.], [0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., -3.5]],
          [[-0., 0.], [3.5, 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]]]])

    # Act
    fault_throws = f.fault_throws_per_edge_per_column(faulted_grid)

    # Assert
    np.testing.assert_array_equal(fault_throws, expected_fault_throws)


def test_fault_throws_per_edge_per_column_axis_polarity_mode_false(faulted_grid):
    # Arrange
    expected_fault_throws = np.array([[[0., 0., 0., 0.], [-0., 0., 0., 0.], [-0., 0., 0., 0.], [-0., 0., -3.5, 0.],
                                       [3.5, 0., 0., 0.], [-0., 0., 0., 0.], [-0., 0., 0., 0.], [-0., 0., 0., 0.]],
                                      [[0., 3.5, 0., -0.], [-0., 7., 0., -0.], [-0., 7., 0., -0.], [-0., 7., -7., -0.],
                                       [7., 7., 0., -0.], [-0., 7., 0., -0.], [-0., 7., 0., -0.], [-0., 3.5, 0., -0.]],
                                      [[0., 0., 0., -3.5], [-0., 0., 0., -7.], [-0., 0., 0., -7.], [-0., 0., -7., -7.],
                                       [7., 0., 0., -7.], [-0., 0., 0., -7.], [-0., 0., 0., -7.], [-0., 0., 0., -3.5]],
                                      [[0., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., -7., -0.],
                                       [7., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., 0., -0.]],
                                      [[0., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., -3.5, -0.],
                                       [3.5, 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., 0., -0.], [-0., 0., 0., -0.]]])

    # Act
    fault_throws = f.fault_throws_per_edge_per_column(faulted_grid, axis_polarity_mode=False)

    # Assert
    np.testing.assert_array_equal(fault_throws, expected_fault_throws)


# todo: check issue #402
def test_fault_throws_per_edge_per_column_mode_mean(faulted_grid):
    # Arrange
    expected_fault_throws = np.array(
        [[[[0., 0.], [0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., -3.5]],
          [[0., 0.], [3.5, 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]]],
         [[[-0., 3.5], [0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., -7.]],
          [[-0., 7.], [7., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 3.5], [-0., 0.]]],
         [[[-3.5, 0.], [0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., -7.]],
          [[-7., 0.], [7., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-3.5, 0.], [-0., 0.]]],
         [[[-0., 0.], [0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., -7.]],
          [[-0., 0.], [7., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]]],
         [[[-0., 0.], [0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., -3.5]],
          [[-0., 0.], [3.5, 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]]]])

    # Act
    fault_throws = f.fault_throws_per_edge_per_column(faulted_grid, mode='mean')

    # Assert
    np.testing.assert_array_equal(fault_throws, expected_fault_throws)


# todo: check issue #402
def test_fault_throws_per_edge_per_column_mode_minimum(faulted_grid):
    # Arrange
    expected_fault_throws = np.array(
        [[[[0., 0.], [0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., -3.5]],
          [[0., 0.], [3.5, 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]],
          [[0., 0.], [-0., 0.]]],
         [[[-0., 3.5], [0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., -7.]],
          [[-0., 7.], [7., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 7.], [-0., 0.]],
          [[-0., 3.5], [-0., 0.]]],
         [[[-3.5, 0.], [0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., -7.]],
          [[-7., 0.], [7., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-7., 0.], [-0., 0.]],
          [[-3.5, 0.], [-0., 0.]]],
         [[[-0., 0.], [0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., -7.]],
          [[-0., 0.], [7., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]]],
         [[[-0., 0.], [0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., -3.5]],
          [[-0., 0.], [3.5, 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]],
          [[-0., 0.], [-0., 0.]]]])

    # Act
    fault_throws = f.fault_throws_per_edge_per_column(faulted_grid, mode='minimum')

    # Assert
    np.testing.assert_array_equal(fault_throws, expected_fault_throws)
