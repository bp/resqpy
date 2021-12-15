import numpy as np
import pytest
from resqpy.olio.grid_functions import left_right_foursome


@pytest.mark.parametrize('previous, next, expected_result', [
    ((1, 0), (0, 1), np.array([[False, True], [True, True]], dtype = bool)),
    ((1, 0), (1, 2), np.array([[False, False], [True, True]], dtype = bool)),
    ((1, 0), (2, 1), np.array([[False, False], [True, False]], dtype = bool)),
    ((1, 2), (0, 1), np.array([[False, True], [False, False]], dtype = bool)),
    ((1, 2), (1, 0), np.array([[True, True], [False, False]], dtype = bool)),
    ((1, 2), (2, 1), np.array([[True, True], [True, False]], dtype = bool)),
    ((0, 1), (1, 0), np.array([[True, False], [False, False]], dtype = bool)),
    ((0, 1), (2, 1), np.array([[True, False], [True, False]], dtype = bool)),
    ((0, 1), (1, 2), np.array([[True, False], [True, True]], dtype = bool)),
    ((2, 1), (1, 0), np.array([[True, True], [False, True]], dtype = bool)),
    ((2, 1), (0, 1), np.array([[False, True], [False, True]], dtype = bool)),
    ((2, 1), (1, 2), np.array([[False, False], [False, True]], dtype = bool)),
])
def test_left_right_foursome(previous, next, expected_result):

    # --------- Arrange ----------
    full_p_list = [previous, (1, 1), next]
    p_index = 1
    # --------- Act ----------
    result = left_right_foursome(full_pillar_list = full_p_list, p_index = p_index)
    # --------- Assert ----------
    np.testing.assert_equal(result, expected_result)


def test_left_right_foursome_error_handling():

    # --------- Arrange ----------
    p_index = 1
    ## arrange for exceptions
    full_p_list_entry_error = [(1, 0), (1, 2), (0, 1)]
    full_p_list_exit_error = [(1, 0), (1, 1), (2, 2)]

    # --------- Act and Assert----------
    with pytest.raises(Exception) as excinfo:
        result_entry_error = left_right_foursome(full_pillar_list = full_p_list_entry_error, p_index = p_index)
        assert "code failure whilst taking entry sides from dubious full pillar list" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo2:
        result_exit_error = left_right_foursome(full_pillar_list = full_p_list_exit_error, p_index = p_index)
        assert "code failure whilst taking exit sides from dubious full pillar list" in str(excinfo.value)
