# ab_toolbox module
# contains small utility functions related to use of pure binary files

version = '4th September 2020'

# Nexus is a registered trademark of the Halliburton Company

import logging

log = logging.getLogger(__name__)
log.debug('ab_toolbox.py version %s', version)

import numpy as np

ab_dtype_dict = {'.db': np.float64, '.fb': np.float32, '.lb': np.int64, '.ib': np.int32, '.bb': np.int8}


def load_array_from_ab_file(file_name, shape, return_64_bit = False):
   """Loads a pure binary file into a numpy array, optionally converting to 64 bit."""

   count = 1
   for axis in range(len(shape)):
      count *= shape[axis]
   dtype = ab_dtype_dict[file_name[-3:]]
   with open(file_name, 'rb') as fp:
      a = np.fromfile(fp, dtype = dtype, count = count).reshape(tuple(shape))
   try:  # expected to return null
      c = fp.read(1)
      if len(c):
         log.warning('binary file contains more data than expected: ' + file_name)
   except Exception:
      pass
   return a


def cp_binary_filename(file_name, nexus_ordering = True):
   """Returns a version of the file name with extension adjusted to indicate reseq order and pure binary."""

   if file_name[-9:] == '.reseq.db':
      root_name = file_name[:-9]
   elif file_name[-3:] == '.db':
      root_name = file_name[:-3]
   else:
      root_name = file_name
   if nexus_ordering:
      return root_name + '.db'
   else:
      return root_name + '.reseq.db'


def binary_file_extension_and_np_type_for_data_type(data_type):
   """Returns a file extension suitable for a pure binary array (ab) file of given data type."""

   if data_type in ['real', 'float']:
      ab_type = np.dtype('f8')  # 8 byte floating point (-double in ab_* suite)
      extension = '.db'
   elif data_type in ['int', 'integer']:
      ab_type = np.dtype('i8')  # NB: 8 byte integers not currently supported by ab_* suite
      extension = '.lb'
   elif data_type in ['bool', 'boolean']:
      ab_type = np.dtype('?')  # single byte boolean (-byte in ab_* suite)
      extension = '.bb'
   else:
      log.error('Unknown data_type [%s] passed to load_array_from_file()', data_type)
      assert (False)
   return (extension, ab_type)
