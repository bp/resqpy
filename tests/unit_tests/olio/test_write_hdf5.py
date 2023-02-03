import pytest
import os
import numpy as np
import h5py

import resqpy.grid as grr
import resqpy.model as rq
import resqpy.property as rqp
import resqpy.olio.write_hdf5 as rqwh
import resqpy.olio.uuid as bu


def test_dtype_size(tmp_path):

    filenames = ['dtype_16', 'dtype_32', 'dtype_64']
    byte_sizes = [2, 4, 8]
    dtypes = [np.float16, np.float32, np.float64]
    hdf5_sizes = []

    extent_kji = (1000, 100, 100)
    a = np.random.random(extent_kji)

    for filename, dtype in zip(filenames, dtypes):
        epc = os.path.join(tmp_path, filename + '.epc')
        h5_file = epc[:-4] + '.h5'
        model = rq.new_model(epc)
        grid = grr.RegularGrid(model, extent_kji = extent_kji)
        grid.create_xml()
        pc = rqp.PropertyCollection()
        pc.set_support(support_uuid = grid.uuid, model = model)
        pc.add_cached_array_to_imported_list(cached_array = a,
                                             source_info = 'random',
                                             keyword = 'NTG',
                                             property_kind = 'net to gross ratio',
                                             indexable_element = 'cells',
                                             uom = 'm3/m3')
        pc.write_hdf5_for_imported_list(dtype = dtype)
        model.store_epc()
        model.h5_release()
        hdf5_sizes.append(os.path.getsize(h5_file))

    assert hdf5_sizes[0] < hdf5_sizes[1] < hdf5_sizes[2]
    for i, (byte_size, hdf5_size) in enumerate(zip(byte_sizes, hdf5_sizes)):
        array_size = byte_size * a.size
        # following may need to be modified if using hdf5 compression
        assert array_size < hdf5_size < array_size + 100000


def test_use_int32(tmp_path):

    extent_kji = (11, 23, 37)
    a = np.arange(np.product(extent_kji), dtype = int).reshape(extent_kji)

    for use_option in [True, False, None]:

        epc = os.path.join(tmp_path, f'use_test_{use_option}.epc')
        h5_file = epc[:-4] + '.h5'
        model = rq.new_model(epc)
        uuid = bu.new_uuid()

        # write the array to hdf5 with a particular use_int32 option
        h5_reg = rqwh.H5Register(model)
        h5_reg.register_dataset(uuid, 'tail', a, dtype = None, hdf5_internal_path = None, copy = False)
        h5_reg.write(release_after = False, use_int32 = use_option)
        model.h5_release()

        # read the array from hdf5
        h5_root = model.h5_access()
        internal_path = '/RESQML/' + str(uuid) + '/tail'
        bh5 = h5_root[internal_path]
        assert bh5 is not None
        b = np.array(bh5)
        model.h5_release()
        del model

        # check the array has been read back and has the correct dtype
        assert b is not None
        assert b.shape == extent_kji
        assert np.all(b == a)
        if use_option is False:
            assert b.dtype is np.int64 or str(b.dtype) in ['int', 'int64']
        else:  # covers both True option and None, as built in default is 32 bit for ints
            assert b.dtype is np.int32 or str(b.dtype).endswith('int32')
