import pytest
from resqpy.multiprocessing.multiprocessing import function_multiprocessing
from resqpy.multiprocessing.wrappers.grid_surface import find_faces_to_represent_surface_regular_wrapper


@pytest.skip("Function not complete.")
def test_function_multiprocessing_three_calls(tmp_path):
    # Arrange
    recombined_epc = f"{tmp_path}/test.epc"

    kwargs_1 = {}
    kwargs_2 = {}
    kwargs_3 = {}

    kwargs_list = [kwargs_1, kwargs_2, kwargs_3]
    success_list_expected = [True] * 3

    # Act
    success_list = function_multiprocessing(find_faces_to_represent_surface_regular_wrapper,
                                            kwargs_list,
                                            recombined_epc,
                                            processes = 3)

    # Assert
    assert success_list == success_list_expected