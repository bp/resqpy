import os
from inspect import getsourcefile
import numpy as np

import resqpy.model as rq


def write_snapshot(data, title):
    # Method to write out a new snapshot of data if one is needed
    reshaped_array = data.reshape(data.shape[0], -1)
    np.savetxt(f'C:\\Test\\{title}.txt', reshaped_array)


def check_load_snapshot(data, filename):
    # Compare the actual data against the stored expected array
    loaded_array = np.loadtxt(filename)
    expected_array = loaded_array.reshape(loaded_array.shape[0], loaded_array.shape[1] // data.shape[2], data.shape[2])
    np.testing.assert_array_almost_equal(data, expected_array)


def test_check_transmisibility_output(test_data_path):
    current_filename = os.path.split(getsourcefile(lambda: 0))[0]
    base_folder = os.path.dirname(os.path.dirname(current_filename))

    resqml_file_root = base_folder + '/example_data/block.epc'
    grid_model = rq.Model(resqml_file_root)
    resqml_grid = grid_model.grid()
    k, j, i = resqml_grid.transmissibility()

    snapshot_filename = current_filename + "/snapshots/transmissibility/"
    check_load_snapshot(i, f'{snapshot_filename}block_i.txt')
    check_load_snapshot(j, f'{snapshot_filename}block_j.txt')
    check_load_snapshot(k, f'{snapshot_filename}block_k.txt')
